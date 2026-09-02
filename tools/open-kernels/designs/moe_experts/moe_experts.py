r"""moe_experts: the 8 routed experts of one MoE block as ONE dispatch.

    acc[2048] = sum_e w[e] * down_e( bf16( silu(gate_e @ xm) * (up_e @ xm) ) )

Phase 2 step 1 of .claude/plans/open-kernels-phase2-moe-first.md: replaces the
40 host-driven dispatches (5 per expert over 4 xclbin contexts) of moe_chain /
decode_chain with one. Weights are the same pool-order chunks make_27b.py
slices today, concatenated per expert: [up 4 stripes | gate 4 stripes | down
16 bands] = 1,966,080 B x 8. (Step 2 replaces the host slice with the on-device
fetch.)

Cores (one per column, Tile(c, 2)), each with exactly its 2 input DMA channels
(w from the shim, h from the memtile) and <= 2 outputs (h part, acc):
  c < 4 : per expert, up band c then gate band c (128 rows each, 32 chunks,
          RS=4) against xm -> h_c = bf16(silu(g)*u) -> joined on a memtile into
          h[512] and broadcast to all 8 cores.
  all 8 : per expert, down bands 2c, 2c+1 (128 rows each, 8 chunks, K = 512)
          against h, then acc[256 rows of core c] += w[e] * y, held in the
          output element across the 8 experts and drained once.
The first element of every core's w stream is the header: xm bf16[2048] at 0
and the router output f32[1024] (w[8] at float 264) at 4096 -- the ln and
router kernels' outputs back to back -- copied to local buffers.

Args: wexp u8[8 * 1966080], hdr u8[20480] (8192 used), acc f32[2048].
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

NE = 8
HID = 2048
FF = 512
TILE = 5120
PER_CALL = 4
CALL_BYTES = PER_CALL * TILE          # one w element (and the header)
STRIPE = 32 * TILE                    # 128 rows x 2048: one up/gate band (RS=4)
DOWN_BAND = 8 * TILE                  # 128 rows x 512
UP_BYTES = 4 * STRIPE                 # 655360
EXPERT_BYTES = 3 * UP_BYTES           # up | gate | down
N_CORES = 8
N_UP = 4                              # cores doing up/gate (4 bands each of up and gate)
DOWN_PER_CORE = 2                     # 16 down bands / 8 cores


def tap(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def moe_experts(wexp: In, hdr: In, acc: Out, *, srchash: CompileTime[int] = 0):
    w_ty = np.ndarray[(NE * EXPERT_BYTES,), np.dtype[np.uint8]]
    elem_ty = np.ndarray[(CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(HID,), np.dtype[bfloat16]]
    h_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    hp_ty = np.ndarray[(FF // N_UP,), np.dtype[bfloat16]]
    r_ty = np.ndarray[(32,), np.dtype[np.float32]]     # router floats 256..287
    band_ty = np.ndarray[(128,), np.dtype[np.float32]]
    accp_ty = np.ndarray[(DOWN_PER_CORE * 128,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(HID,), np.dtype[np.float32]]

    inc = include_dirs() + [str(GEMV)]
    # up/gate: 32-chunk bands (8 groups of 4); down: 8-chunk bands (2 groups).
    # Both RS=4; the .cc entry points are gemv_q4's own (shared gemv_q4_tile body).
    k_up = [ExternalFunction(f"gemv_q4_p4b32r4_k{i}", source_file=str(GEMV / f"gemv_q4_p4b32r4_k{i}.cc"),
                             arg_types=[elem_ty, x_ty, band_ty], include_dirs=inc) for i in range(8)]
    k_dn = [ExternalFunction(f"gemv_q4_p4b8r4_k{i}", source_file=str(GEMV / f"gemv_q4_p4b8r4_k{i}.cc"),
                             arg_types=[elem_ty, h_ty, band_ty], include_dirs=inc) for i in range(2)]
    hdrf = ExternalFunction("moe_hdr", source_file=str(HERE / "moe_hdr.cc"),
                            arg_types=[elem_ty, x_ty, r_ty], include_dirs=inc)
    silu = ExternalFunction("moe_silu", source_file=str(HERE / "moe_silu.cc"),
                            arg_types=[band_ty, band_ty, hp_ty], include_dirs=inc)
    accf = ExternalFunction("moe_acc", source_file=str(HERE / "moe_acc.cc"),
                            arg_types=[band_ty, band_ty, r_ty, accp_ty, np.int32], include_dirs=inc)

    of_w = [ObjectFifo(elem_ty, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_h = ObjectFifo(h_ty, name="h", depth=2)
    of_hp = of_h.prod().join([c * (FF // N_UP) for c in range(N_UP)],
                             obj_types=[hp_ty] * N_UP, names=[f"hp{c}" for c in range(N_UP)],
                             depths=[2] * N_UP)
    of_acc = [ObjectFifo(accp_ty, name=f"acc{c}", depth=1) for c in range(N_CORES)]

    def body_up(win, hout, hin, aout, xb, rb, ub, gb, y0, y1, fhdr, fsilu, facc, *ks):
        kup, kdn = ks[:8], ks[8:]
        we = win.acquire(1)
        fhdr(we, xb, rb)
        win.release(1)
        ae = aout.acquire(1)
        for e in range_(NE):
            for fn in kup:
                we = win.acquire(1)
                fn(we, xb, ub)
                win.release(1)
            for fn in kup:
                we = win.acquire(1)
                fn(we, xb, gb)
                win.release(1)
            he = hout.acquire(1)
            fsilu(gb, ub, he)
            hout.release(1)
            hh = hin.acquire(1)
            for yb in (y0, y1):
                for fn in kdn:
                    we = win.acquire(1)
                    fn(we, hh, yb)
                    win.release(1)
            facc(y0, y1, rb, ae, e)
            hin.release(1)
        aout.release(1)

    def body_dn(win, hin, aout, xb, rb, y0, y1, fhdr, facc, *kdn):
        we = win.acquire(1)
        fhdr(we, xb, rb)
        win.release(1)
        ae = aout.acquire(1)
        for e in range_(NE):
            hh = hin.acquire(1)
            for yb in (y0, y1):
                for fn in kdn:
                    we = win.acquire(1)
                    fn(we, hh, yb)
                    win.release(1)
            facc(y0, y1, rb, ae, e)
            hin.release(1)
        aout.release(1)

    workers = []
    for c in range(N_CORES):
        xb = Buffer(x_ty, name=f"x{c}")
        rb = Buffer(r_ty, name=f"r{c}")
        y0 = Buffer(band_ty, name=f"y0_{c}")
        y1 = Buffer(band_ty, name=f"y1_{c}")
        if c < N_UP:
            ub = Buffer(band_ty, name=f"u{c}")
            gb = Buffer(band_ty, name=f"g{c}")
            workers.append(Worker(body_up,
                                  fn_args=[of_w[c].cons(), of_hp[c].prod(), of_h.cons(), of_acc[c].prod(),
                                           xb, rb, ub, gb, y0, y1, hdrf, silu, accf, *k_up, *k_dn],
                                  tile=Tile(c, 2), stack_size=0x1800))
        else:
            workers.append(Worker(body_dn,
                                  fn_args=[of_w[c].cons(), of_h.cons(), of_acc[c].prod(),
                                           xb, rb, y0, y1, hdrf, accf, *k_dn],
                                  tile=Tile(c, 2), stack_size=0x1800))

    W_TOTAL = NE * EXPERT_BYTES
    acc_taps = [tap(HID, c * DOWN_PER_CORE * 128, DOWN_PER_CORE * 128) for c in range(N_CORES)]

    def sequence(a_w, a_hdr, c_acc, w_prods, acc_conss):
        tg_end = TaskGroup()
        for c in range(N_CORES):
            acc_conss[c].drain(c_acc, tap=acc_taps[c], wait=True, group=tg_end)
        pipe = Pipeline(3)
        for c in range(N_CORES):
            pipe.fill(w_prods[c], a_hdr, tap(CALL_BYTES, 0, CALL_BYTES))
        for e in range(NE):
            base = e * EXPERT_BYTES
            for c in range(N_CORES):
                if c < N_UP:
                    pipe.fill(w_prods[c], a_w, tap(W_TOTAL, base + c * STRIPE, STRIPE))
                    pipe.fill(w_prods[c], a_w, tap(W_TOTAL, base + UP_BYTES + c * STRIPE, STRIPE))
                pipe.fill(w_prods[c], a_w, tap(W_TOTAL, base + 2 * UP_BYTES + c * DOWN_PER_CORE * DOWN_BAND,
                                                DOWN_PER_CORE * DOWN_BAND))
        pipe.finish()
        tg_end.finish()

    rt = Runtime(sequence, [w_ty, elem_ty, acc_ty, [f.prod() for f in of_w], [f.cons() for f in of_acc]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = moe_experts
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + [(GEMV / "gemv_q4.h").read_bytes(),
                                                                       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
