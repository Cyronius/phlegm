r"""xg: Xing 4.0's NPU dispatches (OPEN-FAMILY-XING, recipes/xing.py) -- the weight streams of a layer; the host
does everything between them (xing-runtime.patch): mHC, the norms, RoPE, the absorbed MLA attention, the router.

ONE core program for every dispatch (R-0003: every layer stream is a stream of one image; only the head changes
context). It is Nemotron-H's one-image expert program (designs/nemotronh/nh.py unified_program) with a GATED up
phase, its loop counts runtime parameters:

  phase A   slots_up slots: per band of upc, the gate band (-> ms @64) then the up band (-> ms @0), each ng_up
            groups of a Q4_0-native 64-row band; h band = w_e silu(g) u -> y (the slot's routing weight from xm)
  phase B   slots_dn slots: an f32 x of h_elems elements (h_e, or the GEMV's activation at AD_XF), then dpc bands
            of the down projection accumulated in each core's f32 accumulator over the slots
  phase C   the dpc accumulated bands -> y

  XG_PART=qkva | qb | o    a GEMV: no phase A, one phase-B slot whose x is the f32 at AD_XF (q_a | kv_a_mqa
                           [1344, 3584], q_b [6144, 768], o [3584, 4096])
  XG_PART=moe | dense      the FFN: `slots` slots of gate | up then down over the slots (moe: the top 4 + the shared
                           expert, slot j compiled as pool expert j and re-pointed by the host, `nhroute`; dense:
                           the dense MLP's nine 1024-wide slices). xm (bf16) carries the slots' weights f32[16] at
                           byte W_OFF of its last element.

Every stream writes each core's [slots_up, upc, slots_dn, dpc, h_elems, K_dn, pb_dn, ng_dn, K_up, pb_up, ng_up].
Eight main cores (Tile(c, 2)) on w (two 4608-B chunks an element) / x (4 KB elements, broadcast) / y (64 floats)
streams, lm_head_q8's uneven band split over the cores. No norm helper, no attention core.

Build: OPEN_KERNELS_SPEC=<xing spec> XG_PART=qkva|qb|o|moe|dense python build_design.py designs/xing/xg.py <out>
"""
from __future__ import annotations

import hashlib
import os
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
from recipes import xing as XR  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM, tab_bytes  # noqa: E402

SPEC = current_spec()
R = XR.recipe(SPEC)
L, G = R.layout, R.geo
PART = os.environ.get("XG_PART", "o")
assert PART in (*XR.GEMVS, "moe", "dense"), f"XG_PART={PART!r} (qkva | qb | o | moe | dense)"
N_CORES = XR.N_CORES
OS = ["-Os"]
YB = BAND_ROWS * 4
NULL = ["-DGEMV_NULL"] if os.environ.get("XG_NULL") else []


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def _ef(sym, args, flags=OS):
    return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args,
                            include_dirs=include_dirs() + [str(GEMV)], compile_flags=flags)


def program(part: str):
    HID, FF = G.HID, G.FF
    shapes = XR.gemv_shapes(SPEC)
    K = max(HID, FF, G.QL, G.O_K)
    SLOTS_MAX = XR.slots_max(G)
    elem = np.ndarray[(XR.PER_CALL * XR.CH,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(tab_bytes(K),), np.dtype[np.uint8]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    acc_bands = max([max(XR.split_bands(sh["n"] // BAND_ROWS)) for sh in shapes.values()] +
                    [max(XR.split_bands(HID // BAND_ROWS))])
    ms_ty = np.ndarray[(192,), np.dtype[np.float32]]               # u @0 | g @64 | the down band @128
    acc_ty = np.ndarray[(acc_bands * BAND_ROWS,), np.dtype[np.float32]]
    wv_ty = np.ndarray[(XR.WV_SLOTS,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(XR.RTP_N,), np.dtype[np.int32]]
    i32 = np.int32
    f_gms = _ef("xg_gms", [elem, tab_ty, ms_ty, i32, i32, i32], OS + NULL)
    f_gacc = _ef("xg_gacc", [elem, tab_ty, ms_ty, acc_ty, i32, i32, i32, i32, i32], OS + NULL)
    f_act = _ef("xg_act", [ms_ty, y_ty, wv_ty, i32])
    f_wgrab = _ef("xg_wgrab", [x_ty, wv_ty], OS + [f"-DXG_W_OFF={L.W_OFF}"])
    f_copy = _ef("xg_copy", [acc_ty, y_ty, i32])
    f_prep = _ef("xg_prep", [x_ty, tab_ty, i32, i32])
    f_prepf = _ef("xg_prepf", [x_ty, tab_ty, i32, i32])
    XM = L.XM_ELEMS
    rtps = [Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(XR.RTP_N, dtype=np.int32), use_write_rtp=True)
            for c in range(N_CORES)]
    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=XM)

    def body(win, xin, yout, tab, ms, acc, wv, rtp, gms, gacc, act, wgrab, copy, prep, prepf):
        xe = xin.acquire(XM)
        # the counts, read after the first acquire (it completes only once the stream issued the fill, after its
        # rtp writes: gemm_q4_prefill's ordering)
        s_up, upc, s_dn, dpc, h_el = rtp[0], rtp[1], rtp[2], rtp[3], rtp[4]
        k_dn, pb_dn, ng_dn, k_up, pb_up, ng_up = rtp[5], rtp[6], rtp[7], rtp[8], rtp[9], rtp[10]
        for i in range(XM):
            prep(xe[i], tab, k_up, i)
        wgrab(xe[XM - 1], wv)
        xin.release(XM)
        for e in range_(s_up):
            for _ in range_(upc):
                for g in range_(ng_up):            # the gate band -> ms @64
                    we = win.acquire(1)
                    gms(we, tab, ms, g, pb_up, 64)
                    win.release(1)
                for g in range_(ng_up):            # the up band -> ms @0
                    we = win.acquire(1)
                    gms(we, tab, ms, g, pb_up, 0)
                    win.release(1)
                ye = yout.acquire(1)
                act(ms, ye, wv, e)
                yout.release(1)
        for e in range_(s_dn):
            for i in range_(h_el):
                he = xin.acquire(1)
                prepf(he, tab, k_dn, i)
                xin.release(1)
            for b in range_(dpc):
                for g in range_(ng_dn):
                    we = win.acquire(1)
                    gacc(we, tab, ms, acc, g, pb_dn, b, e, ng_dn)
                    win.release(1)
        for b in range_(dpc):
            ye = yout.acquire(1)
            copy(acc, ye, b)
            yout.release(1)

    workers = [Worker(body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), Buffer(tab_ty, name=f"tab{c}"),
                                     Buffer(ms_ty, name=f"ms{c}"), Buffer(acc_ty, name=f"acc{c}"),
                                     Buffer(wv_ty, name=f"wv{c}"), rtps[c],
                                     f_gms, f_gacc, f_act, f_wgrab, f_copy, f_prep, f_prepf],
                      tile=Tile(c, 2), stack_size=XR.STACK)
               for c in range(N_CORES)]

    def set_rtp(c, vals):
        for i, v in enumerate(vals):
            rtps[c][i] = int(v)

    def sequence(a_pool, a_act, w_prods, x_prod, y_conss):
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        pb_h, pb_f = HID // 128, FF // 128
        if part in ("moe", "dense"):
            sh = XR.ffn_shape(SPEC, part)
            S, pdown = sh["slots"], sh["pool_down"]
            assert S <= SLOTS_MAX
            up_counts, dn_counts = XR.split_bands(FF // BAND_ROWS), XR.split_bands(HID // BAND_ROWS)
            up_offs, dn_offs = XR.offsets(up_counts), XR.offsets(dn_counts)
            bb_h, bb_f = XR.band_bytes(HID), XR.band_bytes(FF)
            for c in range(N_CORES):
                set_rtp(c, [S, up_counts[c], S, dn_counts[c], L.H_STRIDE // ELEM, FF, pb_f, pb_f // 2,
                            HID, pb_h, pb_h // 2, 0])
            px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XM, XM * ELEM))
            # up: each core's h bands of every slot in ONE drain (slot e's at AD_H + e H_STRIDE), then slot e's
            #     gate | up fills (2 bands per output band, contiguous) -- compiled as pool expert e
            for c in range(N_CORES):
                py.drain(y_conss[c], a_act, TensorAccessPattern((1, L.AD_BYTES), L.AD_H + up_offs[c] * YB,
                                                                [1, 1, S, up_counts[c] * YB], [0, 0, L.H_STRIDE, 1]))
            for e in range(S):
                for c in range(N_CORES):
                    pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_EXP + e * L.UPS + 2 * up_offs[c] * bb_h,
                                                   2 * up_counts[c] * bb_h))
            py.finish()                                           # every h_e is in DDR
            # down: h_e, then slot e's down bands; the cores hold the sum
            for c in range(N_CORES):
                py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_MOE + dn_offs[c] * YB, dn_counts[c] * YB))
            for e in range(S):
                px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_H + e * L.H_STRIDE, L.H_STRIDE))
                for c in range(N_CORES):
                    pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, pdown + e * L.DNS + dn_offs[c] * bb_f,
                                                   dn_counts[c] * bb_f))
        else:
            sh = shapes[part]
            Kp, N = sh["k"], sh["n"]
            counts = XR.split_bands(N // BAND_ROWS)
            offs, bb = XR.offsets(counts), XR.band_bytes(Kp)
            h_el = (Kp * 4 + ELEM - 1) // ELEM
            for c in range(N_CORES):
                set_rtp(c, [0, 0, 1, counts[c], h_el, Kp, Kp // 128, Kp // 256, HID, pb_h, pb_h // 2, 0])
            px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XM, XM * ELEM))          # the program's leading xm (unused)
            for c in range(N_CORES):
                py.drain(y_conss[c], a_act, bt(L.AD_BYTES, sh["ay"] + offs[c] * YB, counts[c] * YB))
            px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XF, h_el * ELEM))
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, sh["pool"] + offs[c] * bb, counts[c] * bb))
        py.finish()
        pw.finish()
        px.finish()

    fn_args = [[of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)], of_x.prod(tile=Tile(N_CORES - 1, 0)),
               [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]]
    rt = Runtime(sequence, [pool_ty, act_ty, *fn_args])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def xg(pool: In, act: InOut, *, part: CompileTime[str] = "o", srchash: CompileTime[int] = 0):
    return program(part)


DESIGN = xg
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + [(HERE.parent.parent / "recipes" / "xing.py").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
