"""Rotate GPT-OSS 20B query and key heads using a precomputed YaRN table."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

HEADS = 72
HEAD_DIM = 64


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_rope(qk: In, table: In, output: Out,
                *, srchash: CompileTime[int] = 0):
    qk_ty = np.ndarray[(HEADS * HEAD_DIM,), np.dtype[bfloat16]]
    tab_ty = np.ndarray[(HEAD_DIM,), np.dtype[np.float32]]
    row_ty = np.ndarray[(HEAD_DIM,), np.dtype[bfloat16]]
    fn = ExternalFunction("gptoss_rope_row", source_file=str(HERE / "rope.cc"),
                          arg_types=[row_ty, tab_ty, row_ty], include_dirs=include_dirs())
    qf = ObjectFifo(row_ty, name="qk", depth=2)
    tf = ObjectFifo(tab_ty, name="table", depth=1)
    of = ObjectFifo(row_ty, name="output", depth=2)

    def core_body(qi, ti, yo, rotate):
        tab = ti.acquire(1)
        for _ in range_(HEADS):
            x = qi.acquire(1)
            y = yo.acquire(1)
            rotate(x, tab, y)
            qi.release(1)
            yo.release(1)
        ti.release(1)

    worker = Worker(core_body, fn_args=[qf.cons(), tf.cons(), of.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(q, t, y, qp, tp, yc):
        qtap = TensorAccessPattern((1, HEADS * HEAD_DIM), 0,
                                   [1, 1, HEADS, HEAD_DIM], [0, 0, HEAD_DIM, 1])
        ttap = TensorAccessPattern((1, HEAD_DIM), 0,
                                   [1, 1, 1, HEAD_DIM], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, qtap)
        pipe.fill(qp, q, qtap)
        pipe.fill(tp, t, ttap)
        pipe.finish()

    runtime = Runtime(sequence, [qk_ty, tab_ty, qk_ty,
                                 qf.prod(), tf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_rope
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "rope.cc").read_bytes()).hexdigest()[:8], 16)}
