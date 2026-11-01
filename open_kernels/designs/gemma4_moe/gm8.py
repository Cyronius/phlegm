r"""gm8: Gemma 4 26B-A4B's routed experts as one dispatch on eight cores, one routed slot each
(OPEN-FAMILY-GEMMA4-MOE, recipes/gemma4.py moe_layout; the 4-core gm.py it replaces is kept beside it).

    core c = routed slot c (compiled as expert c, re-pointed per token by moeroute2):
        gate | up (Q4_0, K = hidden), all 12 bands -> h = w_c * gelu_tanh(gate) * up   (768 rows, 704 real)
        down (Q4_0, K = 768) over the core's own h, all 44 bands -> partial m_c f32 [hidden]

    args: pool (the layer's: experts from byte 0), act (xm2 + weights in, the eight partials out)

gm.py splits every expert's rows over 4 cores, so h goes to DDR and back between the two halves and
each core sums the experts' down bands. Here h never leaves its core's L1 and nothing is summed on
the array: the host's gemma4_combine adds the eight partials (AD_MOE + c * hidden * 4) in slot order.
Each core streams one whole expert (2.43 MB of gate | up, then 1.22 MB of down) on its own column's
shim channel, so all eight columns read DDR where gm.py used four.

The host's gemma4_route writes xm2 = pre_ffw_norm_2(res) as bf16 at AD_XM with the eight routing
weights at byte 2048 of its last x element (AD_W), and patches the fills; the slot and pool laws are
gm.py's, so moeroute2's table is unchanged: slot c's gate | up fill at c * EXPERT_UG, its down fill
at POOL_DOWN + c * UP_BYTES.

Build: OPEN_KERNELS_SPEC=<gemma4 26B spec> python build_design.py designs/gemma4_moe/gm8.py <out>
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes import gemma4 as G4  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM  # noqa: E402

SPEC = current_spec()
M = G4.moe_layout(SPEC)
L_POOL = max(G4.layout(SPEC, kind=k).POOL_BYTES for k in G4.KINDS)
HID, FF, TOPK = M.HID, M.FF, M.TOPK
N_CORES = TOPK                       # one routed slot a core
UP_BANDS, DOWN_BANDS = FF // BAND_ROWS, HID // BAND_ROWS      # 12, 44
H_ELEMS = M.H_STRIDE // ELEM
OS = ["-Os"]
YB = BAND_ROWS * 4
STACK = 0x1800
assert N_CORES <= 8 and H_ELEMS == 1, (N_CORES, H_ELEMS)


def per_band(K):
    """chunks in one 64-row band at K (two 32-row halves per k-tile)"""
    return 2 * (K // 256)


def n_groups(K):
    return per_band(K) // M.PER_CALL


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gm8_moe(pool: In, act: InOut, *, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(M.CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(M.TAB_BYTES,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(M.MS_FLOATS,), np.dtype[np.float32]]
    wv_ty = np.ndarray[(8,), np.dtype[np.float32]]
    pool_ty = np.ndarray[(L_POOL,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(M.AD_BYTES,), np.dtype[np.uint8]]
    i32 = np.int32
    inc = include_dirs() + [str(GEMV)]

    def ef(sym, args):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=OS)

    f_gms = ef("gm_gms", [elem, tab_ty, ms_ty, i32, i32, i32])
    f_act = ef("gm8_act", [ms_ty, x_ty, wv_ty, i32, i32])
    f_dn = ef("gm8_dn", [elem, tab_ty, y_ty, i32, i32])
    f_wgrab = ef("gm_wgrab", [x_ty, wv_ty])
    f_prep = ef("gm_prep", [x_ty, tab_ty, i32, i32])
    f_prepf = ef("gm_prepf", [x_ty, tab_ty, i32, i32])

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=M.XM_ELEMS)
    NG_H, PB_H = n_groups(HID), per_band(HID)
    NG_F, PB_F = n_groups(FF), per_band(FF)

    def make_body(slot):
        def main_body(win, xin, yout, tab, ms, hb, wv, gms, act, dn, wgrab, prep, prepf):
            # xm2 and, in its last element, the routing weights
            xe = xin.acquire(M.XM_ELEMS)
            for i in range(M.XM_ELEMS):
                prep(xe[i], tab, HID, i)
            wgrab(xe[M.XM_ELEMS - 1], wv)
            xin.release(M.XM_ELEMS)
            # this slot's expert: [gate | up] per band -> h band in L1 (scaled by w_slot)
            for b in range_(UP_BANDS):
                for g in range_(NG_H):
                    we = win.acquire(1)
                    gms(we, tab, ms, g, PB_H, M.MS_G)
                    win.release(1)
                for g in range_(NG_H):
                    we = win.acquire(1)
                    gms(we, tab, ms, g, PB_H, M.MS_U)
                    win.release(1)
                act(ms, hb, wv, slot, b)
            # h's table over xm2's, then every down band of this expert straight out
            prepf(hb, tab, FF, 0)
            for _ in range_(DOWN_BANDS):
                ye = yout.acquire(1)
                for g in range_(NG_F):
                    we = win.acquire(1)
                    dn(we, tab, ye, g, PB_F)
                    win.release(1)
                yout.release(1)
        return main_body

    workers = [Worker(make_body(c), fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                             Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                             Buffer(x_ty, name=f"hb{c}"), Buffer(wv_ty, name=f"wv{c}"),
                                             f_gms, f_act, f_dn, f_wgrab, f_prep, f_prepf],
                      tile=Tile(c, 2), stack_size=STACK)
               for c in range(N_CORES)]

    def sequence(a_pool, a_act, w_prods, x_prod, y_conss):
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        # 1. xm2 (+ the routing weights) to every core
        px.fill(x_prod, a_act, bt(M.AD_BYTES, M.AD_XM, M.XM_ELEMS * ELEM))
        # 2. each slot's partial m, one drain a core
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, bt(M.AD_BYTES, M.AD_MOE + c * HID * 4, HID * 4))
        # 3. slot c's gate | up, then its down -- compiled as expert c, re-pointed by moeroute2
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L_POOL, c * M.EXPERT_UG, M.EXPERT_UG))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L_POOL, M.POOL_DOWN + c * M.UP_BYTES, M.UP_BYTES))
        py.finish()
        pw.finish()
        px.finish()

    rt = Runtime(sequence, [pool_ty, act_ty,
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(0, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = gm8_moe
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + [(HERE.parent.parent / "recipes" / "gemma4.py").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
