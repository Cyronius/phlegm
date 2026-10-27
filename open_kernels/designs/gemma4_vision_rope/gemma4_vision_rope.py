"""Gemma 4 axial 2D RoPE for one patch's 12 Q and 12 K heads."""
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


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_vision_rope(qk: In, table: In, output: Out,
                       *, srchash: CompileTime[int] = 0):
    rows, width = 24, 64
    full_ty = np.ndarray[(rows * width,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(width,), np.dtype[bfloat16]]
    table_ty = np.ndarray[(64,), np.dtype[np.float32]]
    fn = ExternalFunction("gemma4_vision_rope_row", source_file=str(HERE / "rope.cc"),
                          arg_types=[row_ty, table_ty, row_ty],
                          include_dirs=include_dirs(), compile_flags=["-ffp-contract=off"])
    xf = ObjectFifo(row_ty, name="qk", depth=2)
    tf = ObjectFifo(table_ty, name="table", depth=1)
    yf = ObjectFifo(row_ty, name="output", depth=2)

    def core_body(xi, ti, yo, rotate):
        tab = ti.acquire(1)
        for _ in range_(rows):
            x = xi.acquire(1)
            y = yo.acquire(1)
            rotate(x, tab, y)
            xi.release(1)
            yo.release(1)
        ti.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), tf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, t, y, xp, tp, yc):
        tap = TensorAccessPattern((1, rows * width), 0,
                                  [1, 1, rows, width], [0, 0, width, 1])
        ttap = TensorAccessPattern((1, 64), 0,
                                   [1, 1, 1, 64], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, tap)
        pipe.fill(xp, x, tap)
        pipe.fill(tp, t, ttap)
        pipe.finish()

    runtime = Runtime(sequence, [full_ty, table_ty, full_ty,
                                 xf.prod(), tf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_vision_rope
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "rope.cc").read_bytes()).hexdigest()[:8], 16)}
