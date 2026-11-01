r"""gm: Gemma 4 26B-A4B's routed experts as one dispatch over dx.py's main-core fabric
(OPEN-FAMILY-GEMMA4-MOE, recipes/gemma4.py moe_layout).

    per routed slot e (compiled as expert e, re-pointed per token by moeroute2):
        gate | up (Q4_0, K = hidden) -> h_e = w_e * gelu_tanh(gate) * up   (768 rows, 704 real)
    down (Q4_0, K = 768) over the eight h_e, summed in each core's accumulator -> m f32 [hidden]

    args: pool (the layer's: experts from byte 0), act (xm2 + weights in, h scratch, m out)

The host's gemma4_route writes xm2 = pre_ffw_norm_2(res) as bf16 at AD_XM with the eight routing
weights (renormalised, times per_expert_scale) at byte 2048 of its last x element, and patches the
expert fills; gemma4_combine reads m back. N_CORES main cores (Tile(c, 2)) on w (two 4608-B chunks
an element) / x (4 KB broadcast) / y (64 floats) streams, as dx.py's. No norm helper: the norms
around the block are the host's.

Per core: UP_PC gate | up bands of every expert ([gate band b | up band b] in the pool, so one fill
per expert), then DOWN_PC down bands of every expert into `acc`, handed out band by band at the end.

Build: OPEN_KERNELS_SPEC=<gemma4 26B spec> python build_design.py designs/gemma4_moe/gm.py <out>
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
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

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
HID, FF, N_CORES, TOPK = M.HID, M.FF, M.N_CORES, M.TOPK
OS = ["-Os"]
YB = BAND_ROWS * 4
STACK = 0x1800


def per_band(K):
    """chunks in one 64-row band at K (two 32-row halves per k-tile)"""
    return 2 * (K // 256)


def n_groups(K):
    return per_band(K) // M.PER_CALL


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def parts(col0: int = 0, x_col: int | None = None):
    """gm's workers, its shim endpoints and its runtime sequence, on main cores Tile(col0 + c, 2) with
    their w / y on shim column col0 + c and x on shim column x_col (col0 + N_CORES by default).
    gm_moe below is the standalone image (col0 0); dxm.py puts the same cores into dx.py's layer
    image on its free columns 4-7 (OPEN-FAMILY-GEMMA4-MOE, one context a layer type).
    Returns (workers, runtime fn_args [w_prods, x_prod, y_conss], sequence(a_pool, a_act, *those))."""
    x_col = col0 + N_CORES if x_col is None else x_col
    elem = np.ndarray[(M.CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(M.TAB_BYTES,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(M.MS_FLOATS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(M.DOWN_PC * BAND_ROWS,), np.dtype[np.float32]]
    wv_ty = np.ndarray[(8,), np.dtype[np.float32]]
    i32 = np.int32
    inc = include_dirs() + [str(GEMV)]

    def ef(sym, args):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=OS)

    f_gms = ef("gm_gms", [elem, tab_ty, ms_ty, i32, i32, i32])
    f_gacc = ef("gm_gacc", [elem, tab_ty, ms_ty, acc_ty, i32, i32, i32, i32, i32])
    f_act = ef("gm_act", [ms_ty, y_ty, wv_ty, i32])
    f_wgrab = ef("gm_wgrab", [x_ty, wv_ty])
    f_copy = ef("gm_copy", [acc_ty, y_ty, i32])
    f_prep = ef("gm_prep", [x_ty, tab_ty, i32, i32])
    f_prepf = ef("gm_prepf", [x_ty, tab_ty, i32, i32])

    pfx = "" if col0 == 0 else "gm_"            # the fifo names must not collide with dx.py's w / x / y
    of_w = [ObjectFifo(elem, name=f"{pfx}w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"{pfx}y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name=f"{pfx}x", depth=M.XM_ELEMS)
    NG_H, PB_H = n_groups(HID), per_band(HID)
    NG_F, PB_F = n_groups(FF), per_band(FF)
    H_ELEMS = M.H_STRIDE // ELEM

    def main_body(win, xin, yout, tab, ms, acc, wv, gms, gacc, act, wgrab, copy, prep, prepf):
        # xm2 and, in its last element, the routing weights
        xe = xin.acquire(M.XM_ELEMS)
        for i in range(M.XM_ELEMS):
            prep(xe[i], tab, HID, i)
        wgrab(xe[M.XM_ELEMS - 1], wv)
        xin.release(M.XM_ELEMS)
        # per expert: [gate | up] per band -> h_e band (scaled by w_e)
        for e in range_(TOPK):
            for _ in range_(M.UP_PC):
                for g in range_(NG_H):
                    we = win.acquire(1)
                    gms(we, tab, ms, g, PB_H, M.MS_G)
                    win.release(1)
                for g in range_(NG_H):
                    we = win.acquire(1)
                    gms(we, tab, ms, g, PB_H, M.MS_U)
                    win.release(1)
                ye = yout.acquire(1)
                act(ms, ye, wv, e)
                yout.release(1)
        # down: h_e's table, then slot e's bands added onto the accumulator
        for e in range_(TOPK):
            for i in range_(H_ELEMS):
                he = xin.acquire(1)
                prepf(he, tab, FF, i)
                xin.release(1)
            for b in range_(M.DOWN_PC):
                for g in range_(NG_F):
                    we = win.acquire(1)
                    gacc(we, tab, ms, acc, g, PB_F, b, e, NG_F)
                    win.release(1)
        for b in range_(M.DOWN_PC):
            ye = yout.acquire(1)
            copy(acc, ye, b)
            yout.release(1)

    workers = []
    for c in range(N_CORES):
        workers.append(Worker(main_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                   Buffer(tab_ty, name=f"{pfx}tab{c}"), Buffer(ms_ty, name=f"{pfx}ms{c}"),
                                                   Buffer(acc_ty, name=f"{pfx}acc{c}"), Buffer(wv_ty, name=f"{pfx}wv{c}"),
                                                   f_gms, f_gacc, f_act, f_wgrab, f_copy, f_prep, f_prepf],
                              tile=Tile(col0 + c, 2), stack_size=STACK))

    def sequence(a_pool, a_act, w_prods, x_prod, y_conss):
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        # 1. xm2 (+ the routing weights) to every core
        px.fill(x_prod, a_act, bt(M.AD_BYTES, M.AD_XM, M.XM_ELEMS * ELEM))
        # 2. gate | up: each core's h bands of all the experts in ONE drain (slot e's at AD_H + e * H_STRIDE),
        #    then slot e's fills -- compiled as expert e, re-pointed by moeroute2
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, TensorAccessPattern((1, M.AD_BYTES), M.AD_H + c * M.UP_PC * YB,
                                                            [1, 1, TOPK, M.UP_PC * YB], [0, 0, M.H_STRIDE, 1]))
        for e in range(TOPK):
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L_POOL, e * M.EXPERT_UG + c * M.UP_PC * 2 * M.BB_H,
                                               M.UP_PC * 2 * M.BB_H))
        py.finish()                                               # every h_e is in DDR
        # 3. down: h_e, then slot e's down bands; the cores hold the sum
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, bt(M.AD_BYTES, M.AD_MOE + c * M.DOWN_PC * YB, M.DOWN_PC * YB))
        for e in range(TOPK):
            px.fill(x_prod, a_act, bt(M.AD_BYTES, M.AD_H + e * M.H_STRIDE, M.H_STRIDE))
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L_POOL, M.POOL_DOWN + e * M.UP_BYTES + c * M.DOWN_PC * M.BB_F,
                                               M.DOWN_PC * M.BB_F))
        py.finish()                                               # m is in DDR
        pw.finish()
        px.finish()

    fn_args = [[of_w[c].prod(tile=Tile(col0 + c, 0)) for c in range(N_CORES)],
               of_x.prod(tile=Tile(x_col, 0)),
               [of_y[c].cons(tile=Tile(col0 + c, 0)) for c in range(N_CORES)]]
    return workers, fn_args, sequence


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gm_moe(pool: In, act: InOut, *, srchash: CompileTime[int] = 0):
    pool_ty = np.ndarray[(L_POOL,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(M.AD_BYTES,), np.dtype[np.uint8]]
    workers, fn_args, sequence = parts()
    rt = Runtime(sequence, [pool_ty, act_ty, *fn_args])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = gm_moe
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + [(HERE.parent.parent / "recipes" / "gemma4.py").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
