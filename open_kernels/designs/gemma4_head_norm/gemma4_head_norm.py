"""Gemma 4 text per-head RMSNorm for query, key and value projections."""
from __future__ import annotations

import hashlib
import os
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
def gemma4_head_norm(input: In, scale: In, output: Out,
                     *, heads: CompileTime[int], head_dim: CompileTime[int],
                     srchash: CompileTime[int] = 0):
    assert heads in (1, 2, 8, 16) and head_dim in (256, 512)
    full_ty = np.ndarray[(heads * head_dim,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(head_dim,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma4_head_norm_row", source_file=str(HERE / "norm.cc"),
                          arg_types=[row_ty, row_ty, row_ty], include_dirs=include_dirs(),
                          compile_flags=[f"-DHEAD_DIM={head_dim}", "-ffp-contract=off"])
    xf = ObjectFifo(row_ty, name="input", depth=2)
    wf = ObjectFifo(row_ty, name="scale", depth=1)
    yf = ObjectFifo(row_ty, name="output", depth=2)

    def core_body(xi, wi, yo, norm):
        w = wi.acquire(1)
        for _ in range_(heads):
            x = xi.acquire(1)
            y = yo.acquire(1)
            norm(x, w, y)
            xi.release(1)
            yo.release(1)
        wi.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), wf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, w, y, xp, wp, yc):
        tap = TensorAccessPattern((1, heads * head_dim), 0,
                                  [1, 1, heads, head_dim], [0, 0, head_dim, 1])
        wt = TensorAccessPattern((1, head_dim), 0,
                                 [1, 1, 1, head_dim], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, tap)
        pipe.fill(xp, x, tap)
        pipe.fill(wp, w, wt)
        pipe.finish()

    runtime = Runtime(sequence, [full_ty, row_ty, full_ty,
                                 xf.prod(), wf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_head_norm
_src = (HERE / "norm.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {
    "heads": int(os.environ.get("PHLEGM_HEADS", "8")),
    "head_dim": int(os.environ.get("PHLEGM_HEAD_DIM", "256")),
    "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16),
}
