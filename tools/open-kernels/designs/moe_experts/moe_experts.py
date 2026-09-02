r"""moe_experts: a whole MoE block (8 routed experts + shared expert + combine)
as ONE dispatch.

    acc  = sum_e w[e] * down_e( bf16( silu(gate_e @ xm) * (up_e @ xm) ) )
    out  = xres + acc + sigmoid(xm . sgw) * down_s( bf16( silu(gate_s @ xm) * (up_s @ xm) ) )

Phase 2 steps 1 + 1b of .claude/plans/open-kernels-phase2-moe-first.md:
replaces the 45 host-driven dispatches (5 per routed expert over 4 xclbin
contexts, 5 for the shared expert + combine) of moe_chain / decode_chain with
one. Weights are the same pool-order chunks make_27b.py slices today,
concatenated per expert `[up 4 stripes | gate 4 stripes | down 16 bands]` =
1,966,080 B x 8, then the shared expert `[share_up | share_gate | share_down]`
(the same 3 x 655,360 B, standard RS=2 layout) as a 9th. (Step 2 replaces the
host slice with the on-device fetch.)

Cores (one per column, Tile(c, 2)), each with exactly its 2 input DMA channels
(w from the shim, h from the memtile) and <= 2 outputs (h part, acc):
  c < 4 : per expert, up band c then gate band c (128 rows each: one 32-chunk
          RS=4 band for the routed experts, two 16-chunk RS=2 bands for the
          shared one) against xm -> h_c = bf16(silu(g)*u) -> joined on a
          memtile into h[512] and broadcast to all 8 cores.
  all 8 : per expert, its 256 rows of the down projection (two 8-chunk RS=4
          bands / four 4-chunk RS=2 bands) against h; routed: acc += w[e]*y in
          the output element; shared: out = xres + acc + gate*y, drained once.
The first element of every core's w stream is the header
[xm | router output | sgw | xres] (see moe_hdr.cc), copied to local buffers.

Args: wexp u8[9 * 1966080], hdr u8[20480], out f32[2048].
Build (WSL): python build_design.py designs/moe_experts/moe_experts.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

NE = 8                                # routed experts
NX = NE + 1                           # + the shared expert, streamed the same way
HID = 2048
FF = 512
TILE = 5120
PER_CALL = 4
CALL_BYTES = PER_CALL * TILE          # one w element (and the header)
STRIPE = 32 * TILE                    # 128 rows x 2048: one up/gate band (RS=4) or two (RS=2)
DOWN_BAND = 8 * TILE                  # 128 rows x 512 (RS=4) or two 64-row bands (RS=2)
UP_BYTES = 4 * STRIPE                 # 655360
EXPERT_BYTES = 3 * UP_BYTES           # up | gate | down
N_CORES = 8
N_UP = 4                              # cores doing up/gate
DOWN_PER_CORE = 2                     # 16 x 128-row down bands / 8 cores
HDR_BYTES = 20480


def tap(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def moe_experts(wexp: In, hdr: In, out: Out, *, srchash: CompileTime[int] = 0):
    w_ty = np.ndarray[(NX * EXPERT_BYTES,), np.dtype[np.uint8]]
    elem_ty = np.ndarray[(CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(HID,), np.dtype[bfloat16]]
    h_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    hp_ty = np.ndarray[(FF // N_UP,), np.dtype[bfloat16]]
    r_ty = np.ndarray[(32,), np.dtype[np.float32]]     # router floats 256..287; [0] = shared gate
    band_ty = np.ndarray[(128,), np.dtype[np.float32]]
    accp_ty = np.ndarray[(DOWN_PER_CORE * 128,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(HID,), np.dtype[np.float32]]

    inc = include_dirs() + [str(GEMV)]
    # routed: up/gate 32-chunk RS=4 bands (8 groups of 4), down 8-chunk RS=4 bands
    # (2 groups) -- gemv_q4's own entry points. Shared: RS=2 wrappers with a
    # runtime group + output offset (two 64-row bands per 128-float buffer).
    k_up = [ExternalFunction(f"gemv_q4_p4b32r4_k{i}", source_file=str(GEMV / f"gemv_q4_p4b32r4_k{i}.cc"),
                             arg_types=[elem_ty, x_ty, band_ty], include_dirs=inc) for i in range(8)]
    k_dn = [ExternalFunction(f"gemv_q4_p4b8r4_k{i}", source_file=str(GEMV / f"gemv_q4_p4b8r4_k{i}.cc"),
                             arg_types=[elem_ty, h_ty, band_ty], include_dirs=inc) for i in range(2)]
    r2x = ExternalFunction("gemv_q4_r2x", source_file=str(HERE / "gemv_q4_r2x.cc"),
                           arg_types=[elem_ty, x_ty, band_ty, np.int32, np.int32], include_dirs=inc)
    r2h = ExternalFunction("gemv_q4_r2h", source_file=str(HERE / "gemv_q4_r2h.cc"),
                           arg_types=[elem_ty, h_ty, band_ty, np.int32, np.int32], include_dirs=inc)
    hdrf = ExternalFunction("moe_hdr", source_file=str(HERE / "moe_hdr.cc"),
                            arg_types=[elem_ty, x_ty, r_ty, accp_ty, np.int32], include_dirs=inc)
    silu = ExternalFunction("moe_silu", source_file=str(HERE / "moe_silu.cc"),
                            arg_types=[band_ty, band_ty, hp_ty], include_dirs=inc)
    accf = ExternalFunction("moe_acc", source_file=str(HERE / "moe_acc.cc"),
                            arg_types=[band_ty, band_ty, r_ty, accp_ty, np.int32], include_dirs=inc)
    finf = ExternalFunction("moe_fin", source_file=str(HERE / "moe_fin.cc"),
                            arg_types=[band_ty, band_ty, r_ty, accp_ty, accp_ty], include_dirs=inc)

    of_w = [ObjectFifo(elem_ty, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_h = ObjectFifo(h_ty, name="h", depth=2)
    of_hp = of_h.prod().join([c * (FF // N_UP) for c in range(N_UP)],
                             obj_types=[hp_ty] * N_UP, names=[f"hp{c}" for c in range(N_UP)],
                             depths=[2] * N_UP)
    of_acc = [ObjectFifo(accp_ty, name=f"acc{c}", depth=1) for c in range(N_CORES)]

    def routed_down(win, hh, y0, y1, kdn):
        for yb in (y0, y1):
            for fn in kdn:
                we = win.acquire(1)
                fn(we, hh, yb)
                win.release(1)

    def shared_down(win, hh, y0, y1):
        for j in range(4):                       # 64-row bands 4c..4c+3 -> [y0 | y1]
            we = win.acquire(1)
            r2h(we, hh, y0 if j < 2 else y1, 0, 64 * (j % 2))
            win.release(1)

    def body_up(win, hout, hin, aout, xb, rb, xr, ub, gb, y0, y1, c, fhdr, fsilu, facc, ffin, fr2x, fr2h, *ks):
        kup, kdn = ks[:8], ks[8:]
        we = win.acquire(1)
        fhdr(we, xb, rb, xr, c)
        win.release(1)
        ae = aout.acquire(1)
        for e in range_(NE):
            for dst in (ub, gb):
                for fn in kup:
                    we = win.acquire(1)
                    fn(we, xb, dst)
                    win.release(1)
            he = hout.acquire(1)
            fsilu(gb, ub, he)
            hout.release(1)
            hh = hin.acquire(1)
            routed_down(win, hh, y0, y1, kdn)
            facc(y0, y1, rb, ae, e)
            hin.release(1)
        # the shared expert: 64-row bands 2c, 2c+1 of up then gate (4 groups each)
        for dst in (ub, gb):
            for b in range(2):
                for g in range(4):
                    we = win.acquire(1)
                    fr2x(we, xb, dst, g, 64 * b)
                    win.release(1)
        he = hout.acquire(1)
        fsilu(gb, ub, he)
        hout.release(1)
        hh = hin.acquire(1)
        shared_down(win, hh, y0, y1)
        ffin(y0, y1, rb, xr, ae)
        hin.release(1)
        aout.release(1)

    def body_dn(win, hin, aout, xb, rb, xr, y0, y1, c, fhdr, facc, ffin, fr2h, *kdn):
        we = win.acquire(1)
        fhdr(we, xb, rb, xr, c)
        win.release(1)
        ae = aout.acquire(1)
        for e in range_(NE):
            hh = hin.acquire(1)
            routed_down(win, hh, y0, y1, kdn)
            facc(y0, y1, rb, ae, e)
            hin.release(1)
        hh = hin.acquire(1)
        shared_down(win, hh, y0, y1)
        ffin(y0, y1, rb, xr, ae)
        hin.release(1)
        aout.release(1)

    workers = []
    for c in range(N_CORES):
        xb = Buffer(x_ty, name=f"x{c}")
        rb = Buffer(r_ty, name=f"r{c}")
        xr = Buffer(accp_ty, name=f"xr{c}")
        y0 = Buffer(band_ty, name=f"y0_{c}")
        y1 = Buffer(band_ty, name=f"y1_{c}")
        if c < N_UP:
            ub = Buffer(band_ty, name=f"u{c}")
            gb = Buffer(band_ty, name=f"g{c}")
            workers.append(Worker(body_up,
                                  fn_args=[of_w[c].cons(), of_hp[c].prod(), of_h.cons(), of_acc[c].prod(),
                                           xb, rb, xr, ub, gb, y0, y1, c, hdrf, silu, accf, finf, r2x, r2h,
                                           *k_up, *k_dn],
                                  tile=Tile(c, 2), stack_size=0x1800))
        else:
            workers.append(Worker(body_dn,
                                  fn_args=[of_w[c].cons(), of_h.cons(), of_acc[c].prod(),
                                           xb, rb, xr, y0, y1, c, hdrf, accf, finf, r2h, *k_dn],
                                  tile=Tile(c, 2), stack_size=0x1800))

    W_TOTAL = NX * EXPERT_BYTES
    acc_taps = [tap(HID, c * DOWN_PER_CORE * 128, DOWN_PER_CORE * 128) for c in range(N_CORES)]

    def sequence(a_w, a_hdr, c_out, w_prods, acc_conss):
        tg_end = TaskGroup()
        for c in range(N_CORES):
            acc_conss[c].drain(c_out, tap=acc_taps[c], wait=True, group=tg_end)
        pipe = Pipeline(3)
        for c in range(N_CORES):
            pipe.fill(w_prods[c], a_hdr, tap(HDR_BYTES, 0, HDR_BYTES))
        # The shared expert's byte layout per core is identical to a routed one's
        # (128 rows of up, of gate, 256 rows of down); only the band law differs.
        for e in range(NX):
            base = e * EXPERT_BYTES
            for c in range(N_CORES):
                if c < N_UP:
                    pipe.fill(w_prods[c], a_w, tap(W_TOTAL, base + c * STRIPE, STRIPE))
                    pipe.fill(w_prods[c], a_w, tap(W_TOTAL, base + UP_BYTES + c * STRIPE, STRIPE))
                pipe.fill(w_prods[c], a_w, tap(W_TOTAL, base + 2 * UP_BYTES + c * DOWN_PER_CORE * DOWN_BAND,
                                                DOWN_PER_CORE * DOWN_BAND))
        pipe.finish()
        tg_end.finish()

    rt = Runtime(sequence, [w_ty, np.ndarray[(HDR_BYTES,), np.dtype[np.uint8]], acc_ty,
                            [f.prod() for f in of_w], [f.cons() for f in of_acc]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = moe_experts
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + [(GEMV / "gemv_q4.h").read_bytes(),
                                                                       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
