r"""nh: Nemotron-H's NPU dispatches (OPEN-FAMILY-NEMOTRONH, recipes/nemotronh.py) -- the weight streams of a block;
the host does everything between them (nemotronh-runtime.patch).

  NH_PART=in | qkv | out   one q4 GEMV y[N] = W[N, K] x[K] over a layer pool region: x (bf16, K padded to whole
                           k-tiles with zeros) from act, y (f32) to act. in: the Mamba in_proj [10304, 2816];
                           qkv: q | k | v [4608, 2816]; out: the Mamba out_proj / the attention o [2688, 4096]
                           (the same offset in both pools). args: pool, act
  NH_PART=moe              the routed experts and the shared expert's two halves as 8 slots: per slot, the up
                           projection's bands -> h = w_e relu(up)^2 (one f32 element a band), then the down
                           projection over the 8 h's, summed in each core's accumulator -> m f32 [hidden].
                           Slot j is compiled as pool expert j (up at j UPS, down at POOL_DOWN + j DNS); the host's
                           nh_route writes xm (bf16) with the 8 weights at AD_W and re-points the slots (`nhroute`).
                           args: pool, act

Eight main cores (Tile(c, 2)) on w (two 4608-B Q4_0-native chunks an element) / x (4 KB elements, broadcast) / y
(64 floats) streams, lm_head_q8's uneven band split over the cores, gm.py's expert program. No norm helper, no
attention core: the norms, the SSM, the attention and the router are the host's.

Build: OPEN_KERNELS_SPEC=<nemotronh spec> NH_PART=in|qkv|out|moe python build_design.py designs/nemotronh/nh.py <out>
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
from recipes import nemotronh as NR  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM, tab_bytes  # noqa: E402

SPEC = current_spec()
R = NR.recipe(SPEC)
L, G = R.layout, R.geo
PART = os.environ.get("NH_PART", "out")
assert PART in ("in", "qkv", "out", "moe"), f"NH_PART={PART!r} (in | qkv | out | moe)"
N_CORES = NR.N_CORES
OS = ["-Os"]
YB = BAND_ROWS * 4
NULL = ["-DGEMV_NULL"] if os.environ.get("NH_NULL") else []
SHARED = NR.shared_gemv()        # the three GEMVs as streams of one image (runtime-parameter shapes)


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def offsets(counts):
    out, o = [], 0
    for n in counts:
        out.append(o)
        o += n
    return out


def _ef(sym, args, flags=OS):
    return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args,
                            include_dirs=include_dirs() + [str(GEMV)], compile_flags=flags)


def _types(K):
    return dict(
        elem=np.ndarray[(NR.PER_CALL * NR.CH,), np.dtype[np.uint8]],
        x_ty=np.ndarray[(ELEM // 2,), np.dtype[bfloat16]],
        y_ty=np.ndarray[(BAND_ROWS,), np.dtype[np.float32]],
        tab_ty=np.ndarray[(tab_bytes(K),), np.dtype[np.uint8]],
        act_ty=np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]],
    )


def gemv_program():
    sh = NR.gemv_shapes(SPEC)[PART]
    N, K, POOL, AX, AY, PB_BYTES = sh["n"], sh["k"], sh["pool"], sh["ax"], sh["ay"], sh["pool_bytes"]
    assert N % BAND_ROWS == 0 and K % 256 == 0
    per_band = K // 128                       # chunks a 64-row band: K/256 k-tiles x 2 row halves
    ngroups = per_band // NR.PER_CALL
    bb = NR.band_bytes(K)
    xe_n = (K * 2 + ELEM - 1) // ELEM         # x elements the activation spans
    counts = NR.split_bands(N // BAND_ROWS)
    offs = offsets(counts)
    if SHARED:
        # one core program for every projection (R-0003: the three GEMVs are streams of ONE image, no context
        # change between them): K, the band law and each core's band count arrive as runtime parameters, x is
        # always XE elements (the widest operand's), the table always sized for the widest K
        K_T, xe_n = NR.geometry(SPEC).OUT_K, L.XE
        assert K <= K_T and (K * 2 + ELEM - 1) // ELEM <= xe_n
        pool_ty = np.ndarray[(L.POOL_MIX_BYTES,), np.dtype[np.uint8]]
        PB_BYTES = L.POOL_MIX_BYTES
    else:
        K_T = K
    T = _types(K_T)
    elem, x_ty, y_ty, tab_ty, act_ty = T["elem"], T["x_ty"], T["y_ty"], T["tab_ty"], T["act_ty"]
    pool_ty = np.ndarray[(PB_BYTES,), np.dtype[np.uint8]]
    i32 = np.int32
    f_gy = _ef("nh_gy", [elem, tab_ty, y_ty, i32, i32, i32], OS + NULL)
    f_prep = _ef("nh_prep", [x_ty, tab_ty, i32, i32])

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=xe_n)
    # [bands, K, chunks a band, groups a band] per core, written by each stream (zeros in the image: a
    # shape-dependent initializer would be exactly the bytes that keep two streams' images apart)
    rtps = [Buffer(np.ndarray[(4,), np.dtype[np.int32]], name=f"rtp{c}", initial_value=np.zeros(4, dtype=np.int32),
                   use_write_rtp=True) for c in range(N_CORES)] if SHARED else []

    def make_body(nb):
        def body(win, xin, yout, tab, gy, prep, *rtp):
            xe = xin.acquire(xe_n)
            xs = [xe] if xe_n == 1 else xe
            if SHARED:
                # read after the x acquire: it completes only once the stream has issued the fill, which it
                # does after the rtp writes (gemm_q4_prefill's ordering, no runtime barrier)
                r = rtp[0]
                n_b, k_r, pb_r, ng_r = r[0], r[1], r[2], r[3]
            else:
                n_b, k_r, pb_r, ng_r = nb, K, per_band, ngroups
            for i in range(xe_n):
                prep(xs[i], tab, k_r, i)
            xin.release(xe_n)
            for _ in range_(n_b):
                ye = yout.acquire(1)
                for g in range_(ng_r):
                    we = win.acquire(1)
                    gy(we, tab, ye, g, pb_r, 2)
                    win.release(1)
                yout.release(1)
        return body

    workers = [Worker(make_body(counts[c]),
                      fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), Buffer(tab_ty, name=f"tab{c}"), f_gy, f_prep,
                               *([rtps[c]] if SHARED else [])],
                      tile=Tile(c, 2), stack_size=0x1000)
               for c in range(N_CORES)]

    def sequence(a_pool, a_act, w_prods, x_prod, y_conss):
        if SHARED:
            for c in range(N_CORES):
                rtps[c][0] = counts[c]
                rtps[c][1] = K
                rtps[c][2] = per_band
                rtps[c][3] = ngroups
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        px.fill(x_prod, a_act, bt(L.AD_BYTES, AX, xe_n * ELEM))
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, AY + offs[c] * YB, counts[c] * YB))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(PB_BYTES, POOL + offs[c] * bb, counts[c] * bb))
        py.finish()
        pw.finish()
        px.finish()

    fn_args = [[of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)], of_x.prod(tile=Tile(N_CORES - 1, 0)),
               [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]]
    rt = Runtime(sequence, [pool_ty, act_ty, *fn_args])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def moe_program():
    HIDP, FFP, SLOTS = G.HIDP, G.FFP, G.SLOTS
    K = max(HIDP, FFP)
    T = _types(K)
    elem, x_ty, y_ty, tab_ty, act_ty = T["elem"], T["x_ty"], T["y_ty"], T["tab_ty"], T["act_ty"]
    pool_ty = np.ndarray[(L.POOL_MOE_BYTES,), np.dtype[np.uint8]]
    up_counts = NR.split_bands(G.FF // BAND_ROWS)
    dn_counts = NR.split_bands(G.HID // BAND_ROWS)
    up_offs, dn_offs = offsets(up_counts), offsets(dn_counts)
    ms_ty = np.ndarray[(192,), np.dtype[np.float32]]               # u band @0 | the down band @128
    acc_ty = np.ndarray[(max(dn_counts) * BAND_ROWS,), np.dtype[np.float32]]
    wv_ty = np.ndarray[(8,), np.dtype[np.float32]]
    i32 = np.int32
    f_gms = _ef("nh_gms", [elem, tab_ty, ms_ty, i32, i32, i32], OS + NULL)
    f_gacc = _ef("nh_gacc", [elem, tab_ty, ms_ty, acc_ty, i32, i32, i32, i32, i32], OS + NULL)
    f_act = _ef("nh_act", [ms_ty, y_ty, wv_ty, i32])
    f_wgrab = _ef("nh_wgrab", [x_ty, wv_ty])
    f_copy = _ef("nh_copy", [acc_ty, y_ty, i32])
    f_prep = _ef("nh_prep", [x_ty, tab_ty, i32, i32])
    f_prepf = _ef("nh_prepf", [x_ty, tab_ty, i32, i32])
    pb_h, pb_f = HIDP // 128, FFP // 128
    ng_h, ng_f = pb_h // NR.PER_CALL, pb_f // NR.PER_CALL
    bb_h, bb_f = NR.band_bytes(HIDP), NR.band_bytes(FFP)
    h_elems = L.H_STRIDE // ELEM
    XM = L.XM_ELEMS

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=XM)

    def make_body(upc, dpc):
        def body(win, xin, yout, tab, ms, acc, wv, gms, gacc, act, wgrab, copy, prep, prepf):
            # xm and, in its last element, the slots' weights
            xe = xin.acquire(XM)
            for i in range(XM):
                prep(xe[i], tab, HIDP, i)
            wgrab(xe[XM - 1], wv)
            xin.release(XM)
            # per slot: each up band -> h band = w_e relu(u)^2
            for e in range_(SLOTS):
                for _ in range_(upc):
                    for g in range_(ng_h):
                        we = win.acquire(1)
                        gms(we, tab, ms, g, pb_h, 0)
                        win.release(1)
                    ye = yout.acquire(1)
                    act(ms, ye, wv, e)
                    yout.release(1)
            # down: h_e's table, then slot e's bands onto the accumulator
            for e in range_(SLOTS):
                for i in range_(h_elems):
                    he = xin.acquire(1)
                    prepf(he, tab, FFP, i)
                    xin.release(1)
                for b in range_(dpc):
                    for g in range_(ng_f):
                        we = win.acquire(1)
                        gacc(we, tab, ms, acc, g, pb_f, b, e, ng_f)
                        win.release(1)
            for b in range_(dpc):
                ye = yout.acquire(1)
                copy(acc, ye, b)
                yout.release(1)
        return body

    workers = [Worker(make_body(up_counts[c], dn_counts[c]),
                      fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), Buffer(tab_ty, name=f"tab{c}"),
                               Buffer(ms_ty, name=f"ms{c}"), Buffer(acc_ty, name=f"acc{c}"), Buffer(wv_ty, name=f"wv{c}"),
                               f_gms, f_gacc, f_act, f_wgrab, f_copy, f_prep, f_prepf],
                      tile=Tile(c, 2), stack_size=NR.STACK)
               for c in range(N_CORES)]

    def sequence(a_pool, a_act, w_prods, x_prod, y_conss):
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        # 1. xm (+ the weights) to every core
        px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XM, XM * ELEM))
        # 2. up: each core's h bands of every slot in ONE drain (slot e's at AD_H + e H_STRIDE), then slot e's
        #    fills -- compiled as pool expert e, re-pointed by the host (nhroute)
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, TensorAccessPattern((1, L.AD_BYTES), L.AD_H + up_offs[c] * YB,
                                                            [1, 1, SLOTS, up_counts[c] * YB], [0, 0, L.H_STRIDE, 1]))
        for e in range(SLOTS):
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_MOE_BYTES, e * L.UPS + up_offs[c] * bb_h, up_counts[c] * bb_h))
        py.finish()                                               # every h_e is in DDR
        # 3. down: h_e, then slot e's down bands; the cores hold the sum
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_MOE + dn_offs[c] * YB, dn_counts[c] * YB))
        for e in range(SLOTS):
            px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_H + e * L.H_STRIDE, L.H_STRIDE))
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_MOE_BYTES, L.POOL_DOWN + e * L.DNS + dn_offs[c] * bb_f,
                                               dn_counts[c] * bb_f))
        py.finish()                                               # m is in DDR
        pw.finish()
        px.finish()

    fn_args = [[of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)], of_x.prod(tile=Tile(N_CORES - 1, 0)),
               [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]]
    rt = Runtime(sequence, [pool_ty, act_ty, *fn_args])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


RTP_N = 12


def unified_program(part: str):
    """NEMOTRONH_ONE=1: ONE core program for every dispatch of a block (R-0003: the image changes only for the
    head). It is the expert program with runtime-parameter loop counts: phase A (slots_up slots of upc up bands ->
    w relu^2 -> y), phase B (slots_dn slots: an f32 x of h_elems elements, then dpc bands accumulated), phase C (the
    dpc accumulated bands -> y). The experts run it as nh_moe does; a projection GEMV runs it with no phase A and
    one phase-B slot whose x is the f32 activation at AD_XF (prepf rounds it to bf16 as prep does: the same table).
    Every stream writes each core's [slots_up, upc, slots_dn, dpc, h_elems, K_dn, pb_dn, ng_dn, K_up, pb_up, ng_up]."""
    HIDP, FFP, SLOTS = G.HIDP, G.FFP, G.SLOTS
    shapes = NR.gemv_shapes(SPEC)
    K = max(HIDP, FFP, G.OUT_K)
    T = _types(K)
    elem, x_ty, y_ty, tab_ty, act_ty = T["elem"], T["x_ty"], T["y_ty"], T["tab_ty"], T["act_ty"]
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    acc_bands = max([max(NR.split_bands(sh["n"] // BAND_ROWS)) for sh in shapes.values()] +
                    [max(NR.split_bands(G.HID // BAND_ROWS))])
    ms_ty = np.ndarray[(192,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(acc_bands * BAND_ROWS,), np.dtype[np.float32]]
    wv_ty = np.ndarray[(8,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(RTP_N,), np.dtype[np.int32]]
    i32 = np.int32
    f_gms = _ef("nh_gms", [elem, tab_ty, ms_ty, i32, i32, i32], OS + NULL)
    f_gacc = _ef("nh_gacc", [elem, tab_ty, ms_ty, acc_ty, i32, i32, i32, i32, i32], OS + NULL)
    f_act = _ef("nh_act", [ms_ty, y_ty, wv_ty, i32])
    f_wgrab = _ef("nh_wgrab", [x_ty, wv_ty])
    f_copy = _ef("nh_copy", [acc_ty, y_ty, i32])
    f_prep = _ef("nh_prep", [x_ty, tab_ty, i32, i32])
    f_prepf = _ef("nh_prepf", [x_ty, tab_ty, i32, i32])
    XM = L.XM_ELEMS
    rtps = [Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(RTP_N, dtype=np.int32), use_write_rtp=True)
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
                for g in range_(ng_up):
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
                      tile=Tile(c, 2), stack_size=NR.STACK)
               for c in range(N_CORES)]

    def set_rtp(c, vals):
        for i, v in enumerate(vals):
            rtps[c][i] = int(v)

    def sequence(a_pool, a_act, w_prods, x_prod, y_conss):
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        pb_h, pb_f = HIDP // 128, FFP // 128
        if part == "moe":
            up_counts, dn_counts = NR.split_bands(G.FF // BAND_ROWS), NR.split_bands(G.HID // BAND_ROWS)
            up_offs, dn_offs = offsets(up_counts), offsets(dn_counts)
            bb_h, bb_f = NR.band_bytes(HIDP), NR.band_bytes(FFP)
            for c in range(N_CORES):
                set_rtp(c, [SLOTS, up_counts[c], SLOTS, dn_counts[c], L.H_STRIDE // ELEM, FFP, pb_f, pb_f // 2,
                            HIDP, pb_h, pb_h // 2, 0])
            px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XM, XM * ELEM))
            for c in range(N_CORES):
                py.drain(y_conss[c], a_act, TensorAccessPattern((1, L.AD_BYTES), L.AD_H + up_offs[c] * YB,
                                                                [1, 1, SLOTS, up_counts[c] * YB], [0, 0, L.H_STRIDE, 1]))
            for e in range(SLOTS):
                for c in range(N_CORES):
                    pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, e * L.UPS + up_offs[c] * bb_h, up_counts[c] * bb_h))
            py.finish()
            for c in range(N_CORES):
                py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_MOE + dn_offs[c] * YB, dn_counts[c] * YB))
            for e in range(SLOTS):
                px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_H + e * L.H_STRIDE, L.H_STRIDE))
                for c in range(N_CORES):
                    pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_DOWN + e * L.DNS + dn_offs[c] * bb_f,
                                                   dn_counts[c] * bb_f))
        else:
            sh = shapes[part]
            Kp, N = sh["k"], sh["n"]
            counts = NR.split_bands(N // BAND_ROWS)
            offs, bb = offsets(counts), NR.band_bytes(Kp)
            h_el = (Kp * 4 + ELEM - 1) // ELEM
            for c in range(N_CORES):
                set_rtp(c, [0, 0, 1, counts[c], h_el, Kp, Kp // 128, Kp // 256, HIDP, pb_h, pb_h // 2, 0])
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
def nh(pool: In, act: InOut, *, part: CompileTime[str] = "out", srchash: CompileTime[int] = 0):
    if NR.one_image():
        return unified_program(part)
    return moe_program() if part == "moe" else gemv_program()


DESIGN = nh
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + [(HERE.parent.parent / "recipes" / "nemotronh.py").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src + (b"shared" if SHARED else b"") +
                                                      (b"one" if NR.one_image() else b"")).hexdigest()[:8], 16)}
