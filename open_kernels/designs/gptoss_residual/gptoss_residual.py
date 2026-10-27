"""GPT-OSS 20B output projection bias and residual sum, 2880 FP32 values.

The operation follows the layer reference's order: residual + (projection +
bias). A zero bias also makes this usable after the MoE output mixer.
"""
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

BLOCKS = 45
ROWS = 64


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_residual(pair: In, bias: In, result: Out,
                    *, srchash: CompileTime[int] = 0):
    pair_full = np.ndarray[(BLOCKS * 2 * ROWS,), np.dtype[np.float32]]
    result_full = np.ndarray[(BLOCKS * ROWS,), np.dtype[np.float32]]
    bfull = np.ndarray[(BLOCKS * ROWS,), np.dtype[bfloat16]]
    pair_block = np.ndarray[(2 * ROWS,), np.dtype[np.float32]]
    fblock = np.ndarray[(ROWS,), np.dtype[np.float32]]
    bblock = np.ndarray[(ROWS,), np.dtype[bfloat16]]
    fn = ExternalFunction("gptoss_residual", source_file=str(HERE / "residual.cc"),
                          arg_types=[pair_block, bblock, fblock],
                          include_dirs=include_dirs())
    pf = ObjectFifo(pair_block, name="pair", depth=2)
    bf = ObjectFifo(bblock, name="bias", depth=2)
    yf = ObjectFifo(fblock, name="result", depth=1)

    def core_body(pi, bi, yo, add_fn):
        for _ in range_(BLOCKS):
            pv = pi.acquire(1)
            bv = bi.acquire(1)
            yv = yo.acquire(1)
            add_fn(pv, bv, yv)
            yo.release(1)
            bi.release(1)
            pi.release(1)

    worker = Worker(core_body, fn_args=[pf.cons(), bf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(p, b, y, pp, bp, yc):
        pair_tap = TensorAccessPattern((1, BLOCKS * 2 * ROWS), 0,
                                       [1, 1, BLOCKS, 2 * ROWS], [0, 0, 2 * ROWS, 1])
        tap = TensorAccessPattern((1, BLOCKS * ROWS), 0,
                                   [1, 1, BLOCKS, ROWS], [0, 0, ROWS, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, tap)
        pipe.fill(pp, p, pair_tap)
        pipe.fill(bp, b, tap)
        pipe.finish()

    runtime = Runtime(sequence, [pair_full, bfull, result_full,
                                 pf.prod(), bf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_residual
_src = (HERE / "residual.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
