"""Carry GPT-OSS sink-attention online softmax across bounded NPU calls.

The host seeds the state with max=sink, denominator=1, value accumulator=0.
Each invocation consumes 128 or 512 K/V rows and returns state for the next.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
STEP = HERE.parent / "gptoss_attention" / "step.cc"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

HEADS = 64
HEAD_DIM = 64
STATE_FLOATS = 68
QSTATE_BYTES = HEADS * HEAD_DIM * 2 + HEADS * STATE_FLOATS * 4 + 16
ROW_BF16 = 2 * 8 * HEAD_DIM


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_attention_state(q_state: In, kv_rows: In, output_state: Out,
                           *, rows: CompileTime[int], srchash: CompileTime[int] = 0):
    assert rows in (128, 512)
    qs_ty = np.ndarray[(QSTATE_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(rows * ROW_BF16,), np.dtype[bfloat16]]
    state_ty = np.ndarray[(HEADS * STATE_FLOATS,), np.dtype[np.float32]]
    row_ty = np.ndarray[(ROW_BF16,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(HEADS * HEAD_DIM,), np.dtype[np.float32]]
    stats_ty = np.ndarray[(HEADS,), np.dtype[np.float32]]
    head_ty = np.ndarray[(STATE_FLOATS,), np.dtype[np.float32]]
    inc = include_dirs()
    init = ExternalFunction("gptoss_attn_state_init", source_file=str(HERE / "init.cc"),
                            arg_types=[qs_ty, acc_ty, stats_ty, stats_ty], include_dirs=inc)
    step = ExternalFunction("gptoss_attn_state_step", source_file=str(HERE / "state_step.cc"),
                            arg_types=[qs_ty, row_ty, acc_ty, stats_ty, stats_ty, np.int32],
                            include_dirs=inc + [str(STEP.parent)])
    finish = ExternalFunction("gptoss_attn_state_finish", source_file=str(HERE / "finish.cc"),
                              arg_types=[acc_ty, stats_ty, stats_ty, head_ty, np.int32],
                              include_dirs=inc)
    qf = ObjectFifo(qs_ty, name="q_state", depth=1)
    kvf = ObjectFifo(row_ty, name="kv_rows", depth=2)
    of = ObjectFifo(head_ty, name="output_state", depth=1)

    def core_body(qi, ki, yo, acc, mx, den, init_fn, step_fn, finish_fn):
        q = qi.acquire(1)
        init_fn(q, acc, mx, den)
        for row in range_(rows):
            kv = ki.acquire(1)
            step_fn(q, kv, acc, mx, den, row)
            ki.release(1)
        for h in range_(HEADS):
            y = yo.acquire(1)
            finish_fn(acc, mx, den, y, h)
            yo.release(1)
        qi.release(1)

    worker = Worker(core_body,
                    fn_args=[qf.cons(), kvf.cons(), of.prod(),
                             Buffer(acc_ty, name="acc"), Buffer(stats_ty, name="max"),
                             Buffer(stats_ty, name="denom"), init, step, finish],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(q, kv, y, qp, kp, yc):
        qtap = TensorAccessPattern((1, QSTATE_BYTES), 0,
                                   [1, 1, 1, QSTATE_BYTES], [0, 0, 0, 1])
        ktap = TensorAccessPattern((1, rows * ROW_BF16), 0,
                                   [1, 1, rows, ROW_BF16], [0, 0, ROW_BF16, 1])
        ytap = TensorAccessPattern((1, HEADS * STATE_FLOATS), 0,
                                   [1, 1, HEADS, STATE_FLOATS], [0, 0, STATE_FLOATS, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, ytap)
        pipe.fill(qp, q, qtap)
        pipe.fill(kp, kv, ktap)
        pipe.finish()

    runtime = Runtime(sequence, [qs_ty, kv_ty, state_ty,
                                 qf.prod(), kvf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_attention_state
_src = b"".join((HERE / name).read_bytes() for name in ("init.cc", "state_step.cc", "finish.cc")) + \
       STEP.read_bytes() + (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"rows": int(os.environ.get("GPTOSS_ATTN_STATE_ROWS", "512")),
              "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
