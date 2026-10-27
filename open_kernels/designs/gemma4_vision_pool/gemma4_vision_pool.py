"""Gemma 4 E2B/E4B 3x3 patch pool into 280 FP32 soft tokens."""
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

TOKENS, PATCHES, HIDDEN = 280, 9, 768


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_vision_pool(groups: In, output: Out, *, srchash: CompileTime[int] = 0):
    input_ty = np.ndarray[(PATCHES * HIDDEN,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(HIDDEN,), np.dtype[np.float32]]
    fn = ExternalFunction("gemma4_vision_pool_group", source_file=str(HERE / "pool.cc"),
                          arg_types=[input_ty, output_ty], include_dirs=include_dirs(),
                          compile_flags=["-ffp-contract=off"])
    xf = ObjectFifo(input_ty, name="groups", depth=1)
    yf = ObjectFifo(output_ty, name="output", depth=1)

    def core_body(xi, yo, pool):
        for _ in range_(TOKENS):
            x = xi.acquire(1)
            y = yo.acquire(1)
            pool(x, y)
            xi.release(1)
            yo.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, y, xp, yc):
        itap = TensorAccessPattern((1, TOKENS * PATCHES * HIDDEN), 0,
                                   [1, 1, TOKENS, PATCHES * HIDDEN],
                                   [0, 0, PATCHES * HIDDEN, 1])
        otap = TensorAccessPattern((1, TOKENS * HIDDEN), 0,
                                   [1, 1, TOKENS, HIDDEN], [0, 0, HIDDEN, 1])
        pipe = Pipeline(2)
        pipe.drain(yc, y, otap)
        pipe.fill(xp, x, itap)
        pipe.finish()

    runtime = Runtime(sequence,
                      [np.ndarray[(TOKENS * PATCHES * HIDDEN,), np.dtype[bfloat16]],
                       np.ndarray[(TOKENS * HIDDEN,), np.dtype[np.float32]],
                       xf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_vision_pool
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "pool.cc").read_bytes()).hexdigest()[:8], 16)}
