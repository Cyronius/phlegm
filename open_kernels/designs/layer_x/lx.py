r"""lx: a whole linear-attention layer (attention block + MoE block) in ONE xclbin
context (phase 2 "whole-layer context", .claude/plans/open-kernels-phase2-whole-layer.md):

    ln -> gemv qkv | z -> glue -> [DeltaNet: its own context, for now] -> post -> gemv out
       -> ln (+residual) -> router -> MoE (8 routed experts + shared + combine)

The 8 main cores (one per column, Tile(c, 2)) run every GEMV and the MoE in
one core program fed by three streams each: w (10 KB elements from the shim:
weights, the MoE header, experts), x (4 KB elements broadcast from the shim:
xn, og, xm, the expert hidden h) and y (256 B elements to the shim: band
results, the hidden parts, the block output). Helper cores: ln + router
(Tile(0, 3)), post (Tile(1, 3)), glue (Tile(2, 3)). Shim budget: 13 fills,
11 drains. Cores do not know about dispatch boundaries -- they block on the
next element -- so one xclbin serves THREE instruction streams (CompileTime
`part`): 0 = ln -> qkv|z -> glue (the DeltaNet step runs in between, in
designs/deltanet's context, on `act`), 1 = post -> out -> ln -> router, 2 = the
MoE (the driver's `moeroute2` patches the routed slots' fills from the router
output between parts 1 and 2). Build all three; they share part 0's xclbin.

Geometry: the recipe's (open_kernels/recipes/qwen36moe.py) for the spec named
by OPEN_KERNELS_SPEC (else the checked-in 27B) -- layout.py for the byte
layouts, xcommon.py for the main-core streams, `R.linear` for this layer type.

Args (layout.py): pool u8[POOL_BYTES] (qkv, z, experts at their pool offsets),
xres f32[HID] (in: the layer input; out: the layer output), consts (per layer),
state (conv state + S, in place), act (scratch; vec/o for DeltaNet).
Build (WSL): for p in 0 1 2: LX_PART=$p python build_design.py designs/layer_x/lx.py designs/layer_x/build_lx$p
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_O, A_OG, A_OUT, A_QKV, A_RES, A_ROUT, A_VEC, A_XM, A_XN, A_Z, A_HP,  # noqa: E402
                    C_BYTES, C_LNW, C_NW, C_POSTLN, C_RW, C_SGW, C_SIDE, C_WOUT, GLUE_SIDE_BYTES, POOL_BYTES,
                    POOL_QKV, POOL_Z, STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES, R, SPEC)
import xcommon as X  # noqa: E402

D = R.linear
if D is None:
    sys.exit("lx.py: the spec has no linear-attention layers")
HID = X.HID
N_CORES = X.N_CORES
ELEM = X.ELEM
QKV_PC, Z_PC, OUT_PC = D.QKV_PC, D.Z_PC, D.OUT_PC      # bands per core: qkv (K = HID), z (K = HID), out (K = VW)
VW, OUT_K = D.VW, D.OUT_K
# dn_glue / dn_post
NCH, NHEAD = D.NCH, D.NHEAD
TILE, NT = D.TILE, D.NT
AB_ELEMS = D.AB_ELEMS
G, NG = D.G, D.NG
CONV_ROWS = SPEC.conv_kernel - 1                        # conv state rows (the taps before the new one)
KEY_TILES = D.VALUE_TILE0                               # tiles of the two key groups; the value tiles follow
CONVW_ELEMS = SPEC.conv_kernel * TILE * 2 // ELEM       # 4 KB side elements holding one tile's conv taps
PART = int(os.environ.get("LX_PART", 0))
STOP = int(os.environ.get("LX_STOP", 99))     # debug: truncate part 0 after the glue (1) / DeltaNet (2)


def rows3(t: int):
    """Tile t (1024 bf16 = 2048 B) of each of the conv-state rows, in BYTES of the state BO
    (the conv state is its first STATE_S_OFF bytes; S follows)."""
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, STATE_BYTES), t * TILE * 2, [1, 1, CONV_ROWS, TILE * 2], [0, 0, NCH * 2, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lx(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, *, part: CompileTime[int] = 0,
       stop: CompileTime[int] = 99, srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    u8_4k, u8_2k = tl["u8_4k"], np.ndarray[(2048,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(C_BYTES,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_BYTES,), np.dtype[np.uint8]]      # [conv state | S (in place)]
    act_ty = np.ndarray[(A_BYTES,), np.dtype[np.uint8]]
    nw_ty = np.ndarray[(SPEC.lin_value_dim,), np.dtype[bfloat16]]
    f32 = np.ndarray[(NHEAD,), np.dtype[np.float32]]
    fqk = np.ndarray[(2 * D.KEY_WIDTH,), np.dtype[np.float32]]
    fvt = np.ndarray[(TILE,), np.dtype[np.float32]]
    fxn = np.ndarray[(HID,), np.dtype[bfloat16]]

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(X.LN), str(X.RT), str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    f_ab = ExternalFunction("glue_ab", source_file=str(GLUE / "glue_ab.cc"), arg_types=[u8_4k, fxn, f32, np.int32], include_dirs=inc)
    f_small = ExternalFunction("glue_small_fn", source_file=str(GLUE / "glue_small.cc"), arg_types=[u8_4k, f32, f32, f32, f32], include_dirs=inc)
    f_conv = ExternalFunction("glue_conv", source_file=str(GLUE / "glue_conv.cc"),
                              arg_types=[u8_2k, u8_2k, u8_2k, u8_2k, u8_2k, u8_4k, u8_4k, u8_2k, u8_2k, u8_2k, fqk, fvt, np.int32, np.int32],
                              include_dirs=inc)
    f_emit = ExternalFunction("glue_emit_fn", source_file=str(GLUE / "glue_emit.cc"), arg_types=[fqk, fvt, f32, f32, u8_2k, np.int32, np.int32], include_dirs=inc)
    f_copy = ExternalFunction("glue_copy_xn", source_file=str(GLUE / "glue_copy.cc"), arg_types=[u8_4k, fxn], include_dirs=inc)
    post_fn = ExternalFunction("post_fn", source_file=str(POST / "post.cc"), arg_types=[u8_4k, u8_4k, nw_ty, u8_2k], include_dirs=inc)
    post_copy = ExternalFunction("post_copy_nw", source_file=str(POST / "post_copy.cc"), arg_types=[u8_4k, nw_ty], include_dirs=inc)

    # ---- fifos
    of_w = [ObjectFifo(t["elem"], name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(t["y"], name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(t["x"], name="x", depth=2)           # broadcast; og is acquired as 2 elements
    of_lni = ObjectFifo(u8_4k, name="lni", depth=5)        # [x0 x1 w] | [x0 x1 w a0 a1] | W x256
    of_lno = ObjectFifo(u8_4k, name="lno", depth=3)        # [xn] | [y0 y1 xm] | [rout]
    of_side = ObjectFifo(u8_4k, name="side", depth=2)
    of_gact = ObjectFifo(u8_2k, name="gact", depth=5)
    of_gout = ObjectFifo(u8_2k, name="gout", depth=3)
    of_pin = ObjectFifo(u8_4k, name="pin", depth=2)        # [nw][o g][z g]...
    of_pout = ObjectFifo(u8_2k, name="pout", depth=2)      # og per group

    # ---- cores
    def main_body(win, xin, yout, *args):
        B, K = X.unpack_args(args)
        tab = B["tab"]
        # part 0: qkv | z against xn, then this core's DeltaNet heads
        xe = xin.acquire(1)
        K["prep2048"](xe, tab)
        X.gemv_bands(win, yout, tab, K["gy"], QKV_PC + Z_PC, X.n_groups(HID), X.per_band(HID), 2)
        xin.release(1)
        X.dn_body(win, yout, B, K)
        # (still part 0) out against og (two 4 KB elements, K = VW)
        oe = xin.acquire(2)
        K["prep4096a"](oe[0], tab)
        K["prep4096b"](oe[1], tab)
        X.gemv_bands(win, yout, tab, K["gy"], OUT_PC, X.n_groups(OUT_K), X.per_band(OUT_K), 2)
        xin.release(2)
        # part 1: the MoE block
        X.moe_body(win, xin, yout, B, K)

    def glue_body(sin, ain, oout, acc_a, acc_b, decay, beta, qk, vt, xn, fab, fsmall, fconv, femit, fcopy):
        e0 = sin.acquire(1)
        fcopy(e0, xn)
        sin.release(1)
        for acc in (acc_a, acc_b):
            for tile in range_(AB_ELEMS):
                ww = sin.acquire(1)
                fab(ww, xn, acc, tile)
                sin.release(1)
        sm = sin.acquire(1)
        fsmall(sm, acc_a, acc_b, decay, beta)
        sin.release(1)
        for base in (0, KEY_TILES):
            for tt in range_(KEY_TILES):
                ww = sin.acquire(CONVW_ELEMS)
                e = ain.acquire(2 + CONV_ROWS)
                o = oout.acquire(CONV_ROWS)
                fconv(e[0], e[1], e[2], e[3], e[4], ww[0], ww[1], o[0], o[1], o[2], qk, vt, tt, base)
                oout.release(CONV_ROWS)
                ain.release(2 + CONV_ROWS)
                sin.release(CONVW_ELEMS)
                if base == KEY_TILES:
                    for i in range_(D.HEADS_PER_TILE):
                        r = oout.acquire(1)
                        femit(qk, vt, decay, beta, r, tt, i)
                        oout.release(1)

    def post_body(ain, aout, nwb, f, fc):
        e = ain.acquire(1)
        fc(e, nwb)
        ain.release(1)
        for _ in range_(NG):
            e = ain.acquire(2)
            r = aout.acquire(1)
            f(e[0], e[1], nwb, r)
            aout.release(1)
            ain.release(2)

    workers = [Worker(X.ln_router_body,
                      fn_args=[of_lni.cons(), of_lno.prod(), Buffer(tl["xb"], name="rxs"), Buffer(tl["racc"], name="racc"),
                               L["ln_nr"], L["ln"], L["rcopy"], L["racc"], L["rfin"]],
                      tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body,
                              fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), *X.worker_args(X.core_buffers(t, c), K)],
                              tile=Tile(c, 2), stack_size=0x1800))
    workers.append(Worker(post_body, fn_args=[of_pin.cons(), of_pout.prod(), Buffer(nw_ty, name="nwb"), post_fn, post_copy],
                          tile=Tile(1, 3), stack_size=0x1800))
    workers.append(Worker(glue_body,
                          fn_args=[of_side.cons(), of_gact.cons(), of_gout.prod(),
                                   Buffer(f32, name="acc_a"), Buffer(f32, name="acc_b"), Buffer(f32, name="decay"),
                                   Buffer(f32, name="beta"), Buffer(fqk, name="qk"), Buffer(fvt, name="vt"), Buffer(fxn, name="xnb"),
                                   f_ab, f_small, f_conv, f_emit, f_copy],
                          tile=Tile(2, 3), stack_size=0x1800))

    bt = X.bt
    BB_HID, BB_OUT = X.band_bytes(HID), X.band_bytes(OUT_K)
    YB = X.BAND_ROWS * 4                                   # one band's y bytes

    # ---- host sequences (one per instruction stream)
    def sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss, side_p, gact_p, gout_c, pin_p, pout_c):
        if part == 0:
            # 1. layer-entry norm: xn -> act[A_XN]
            tg_ln = TaskGroup()
            lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
            lni.fill(a_consts, tap=bt(C_BYTES, C_LNW, ELEM), wait=True, group=tg_ln)
            lno.drain(a_act, tap=bt(A_BYTES, A_XN, ELEM), wait=True, group=tg_ln)
            # 2. qkv | z GEMV: weights now, x after the norm
            pw, py = Pipeline(3), Pipeline(3)
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
                py.drain(y_conss[c], a_act, bt(A_BYTES, A_QKV + c * QKV_PC * YB, QKV_PC * YB))
                py.drain(y_conss[c], a_act, bt(A_BYTES, A_Z + c * Z_PC * YB, Z_PC * YB))
            tg_ln.finish()                                   # xn is in DDR
            px = Pipeline(3)
            px.fill(x_prod, a_act, bt(A_BYTES, A_XN, ELEM))
            tg_s = TaskGroup()
            side_p.fill(a_act, tap=bt(A_BYTES, A_XN, ELEM), wait=True, group=tg_s)
            side_p.fill(a_consts, tap=bt(C_BYTES, C_SIDE, GLUE_SIDE_BYTES), wait=True, group=tg_s)
            py.finish()                                      # qkv, z are in DDR
            # 3. glue: conv state updated in place, DeltaNet records -> act[A_VEC]
            pipe = Pipeline(3)
            for tt in range(NT):
                pipe.drain(gout_c, a_state, rows3(tt))
                if tt >= KEY_TILES:
                    pipe.drain(gout_c, a_act, bt(A_BYTES, A_VEC + (tt - KEY_TILES) * D.HEADS_PER_TILE * D.RECORD_BYTES,
                                                 D.HEADS_PER_TILE * D.RECORD_BYTES))
                pipe.fill(gact_p, a_act, bt(A_BYTES, A_QKV + tt * TILE * 4, TILE * 4))
                pipe.fill(gact_p, a_state, rows3(tt))
            pipe.finish()                                    # the records are in DDR
            tg_s.finish()
            if STOP == 1:
                pw.finish()
                return
            # 4. DeltaNet on the main cores: S in place, o -> act[A_O]
            X.dn_sequence(pw, py, a_state, a_act, w_prods, y_conss, A_BYTES, A_VEC, A_O, STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES)
            py.finish()                                      # o is in DDR
            if STOP == 2:
                pw.finish()
                return
            # 5. post: og -> act[A_OG] (z from act, o from DeltaNet)
            pipe = Pipeline(3)
            pipe.fill(pin_p, a_consts, bt(C_BYTES, C_NW, ELEM))
            for g in range(NG):
                pipe.drain(pout_c, a_act, bt(A_BYTES, A_OG + g * G * 2, G * 2))
                pipe.fill(pin_p, a_act, bt(A_BYTES, A_O + g * G * 4, G * 4))
                pipe.fill(pin_p, a_act, bt(A_BYTES, A_Z + g * G * 4, G * 4))
            pipe.finish()                                    # og is in DDR
            # 6. out projection (weights in consts) against og
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_consts, bt(C_BYTES, C_WOUT + c * OUT_PC * BB_OUT, OUT_PC * BB_OUT))
                py.drain(y_conss[c], a_act, bt(A_BYTES, A_OUT + c * OUT_PC * YB, OUT_PC * YB))
            px.fill(x_prod, a_act, bt(A_BYTES, A_OG, VW * 2))
            py.finish()                                      # out is in DDR
            # 7. residual + post-attention norm, then the router
            tg_ln = TaskGroup()
            lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
            lni.fill(a_consts, tap=bt(C_BYTES, C_POSTLN, ELEM), wait=True, group=tg_ln)
            lni.fill(a_act, tap=bt(A_BYTES, A_OUT, HID * 4), wait=True, group=tg_ln)
            lno.drain(a_act, tap=bt(A_BYTES, A_RES, HID * 4), wait=True, group=tg_ln)
            lno.drain(a_act, tap=bt(A_BYTES, A_XM, ELEM), wait=True, group=tg_ln)
            tg_ln.finish()
            tg_r = TaskGroup()
            lni.fill(a_consts, tap=bt(C_BYTES, C_RW, X.W_ELEMS * ELEM), wait=True, group=tg_r)
            lno.drain(a_act, tap=bt(A_BYTES, A_ROUT, ELEM), wait=True, group=tg_r)
            tg_r.finish()
            pw.finish()
            px.finish()
        else:
            # 8. the MoE block (moeroute2 has pointed the routed slots' fills at the router's choice)
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                           A_BYTES, C_BYTES, A_XM, A_ROUT, A_RES, A_HP, C_SGW)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)), of_gout.cons(tile=Tile(2, 0)),
                            of_pin.prod(tile=Tile(4, 0)), of_pout.cons(tile=Tile(1, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = lx
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes()] + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "stop": STOP, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
