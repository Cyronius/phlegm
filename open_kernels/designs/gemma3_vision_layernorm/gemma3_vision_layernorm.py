"""SigLIP vision LayerNorm over 256 rows of 1152 BF16 channels."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

ROWS, HIDDEN = 256, 1152


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma3_vision_layernorm(input: In, params: In, output: Out,
                            *, srchash: CompileTime[int] = 0):
    row_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    params_ty = np.ndarray[(2 * HIDDEN,), np.dtype[bfloat16]]
    all_rows = np.ndarray[(ROWS * HIDDEN,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma3_vision_layernorm_row",
                          source_file=str(HERE / "layernorm.cc"),
                          arg_types=[row_ty, params_ty, row_ty],
                          include_dirs=include_dirs(), compile_flags=["-ffp-contract=off"])
    xf = ObjectFifo(row_ty, name="input", depth=1)
    pf = ObjectFifo(params_ty, name="params", depth=1)
    of = ObjectFifo(row_ty, name="output", depth=1)

    def body(xi, pi, yo, norm):
        p = pi.acquire(1)
        for _ in range_(ROWS):
            x = xi.acquire(1)
            y = yo.acquire(1)
            norm(x, p, y)
            xi.release(1)
            yo.release(1)
        pi.release(1)

    worker = Worker(body, fn_args=[xf.cons(), pf.cons(), of.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, p, y, xp, pp, yc):
        xt = TensorAccessPattern((1, ROWS * HIDDEN), 0,
                                 [1, 1, ROWS, HIDDEN], [0, 0, HIDDEN, 1])
        pt = TensorAccessPattern((1, 2 * HIDDEN), 0,
                                 [1, 1, 1, 2 * HIDDEN], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, xt)
        pipe.fill(xp, x, xt)
        pipe.fill(pp, p, pt)
        pipe.finish()

    runtime = Runtime(sequence, [all_rows, params_ty, all_rows,
                                 xf.prod(), pf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma3_vision_layernorm
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "layernorm.cc").read_bytes()).hexdigest()[:8], 16)}
