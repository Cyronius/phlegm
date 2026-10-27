"""Scale all eight Gemma 4 audio attention Q/K heads on NPU."""
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

ROWS, DIM = 8 * (12 + 24), 128


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_audio_qk_scale(qk: In, learned: In, output: Out,
                          *, srchash: CompileTime[int] = 0):
    in_row = np.ndarray[(DIM,), np.dtype[bfloat16]]
    out_row = np.ndarray[(DIM,), np.dtype[np.float32]]
    in_panel = np.ndarray[(ROWS * DIM,), np.dtype[bfloat16]]
    out_panel = np.ndarray[(ROWS * DIM,), np.dtype[np.float32]]
    fn = ExternalFunction("gemma4_audio_qk_scale_row", source_file=str(HERE / "scale.cc"),
                          arg_types=[in_row, in_row, out_row, np.int32],
                          include_dirs=include_dirs(), compile_flags=["-ffp-contract=off"])
    qf = ObjectFifo(in_row, name="qk", depth=1)
    lf = ObjectFifo(in_row, name="learned", depth=1)
    of = ObjectFifo(out_row, name="output", depth=1)

    def body(qi, li, oo, scale):
        s = li.acquire(1)
        for row in range_(ROWS):
            x = qi.acquire(1)
            y = oo.acquire(1)
            scale(x, s, y, row)
            qi.release(1)
            oo.release(1)
        li.release(1)

    worker = Worker(body, fn_args=[qf.cons(), lf.cons(), of.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(q, s, y, qp, sp, yc):
        qtap = TensorAccessPattern((1, ROWS * DIM), 0,
                                   [1, 1, ROWS, DIM], [0, 0, DIM, 1])
        stap = TensorAccessPattern((1, DIM), 0, [1, 1, 1, DIM], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, qtap)
        pipe.fill(qp, q, qtap)
        pipe.fill(sp, s, stap)
        pipe.finish()

    runtime = Runtime(sequence, [in_panel, in_row, out_panel,
                                 qf.prod(), lf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_audio_qk_scale
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "scale.cc").read_bytes()).hexdigest()[:8], 16)}
