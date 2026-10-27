"""Gemma 4 text hidden-width RMSNorm followed by BF16 residual addition."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_hidden_norm(value_residual: In, scale: In, output: Out,
                       *, hidden: CompileTime[int], srchash: CompileTime[int] = 0):
    assert hidden in (768, 1024, 1536, 2560, 3840)
    pair_ty = np.ndarray[(2 * hidden,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(hidden,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma4_hidden_norm", source_file=str(HERE / "norm.cc"),
                          arg_types=[pair_ty, row_ty, row_ty], include_dirs=include_dirs(),
                          compile_flags=[f"-DHIDDEN={hidden}", "-ffp-contract=off"])
    xf = ObjectFifo(pair_ty, name="value_residual", depth=1)
    wf = ObjectFifo(row_ty, name="scale", depth=1)
    yf = ObjectFifo(row_ty, name="output", depth=1)

    def core_body(xi, wi, yo, norm):
        x = xi.acquire(1)
        w = wi.acquire(1)
        y = yo.acquire(1)
        norm(x, w, y)
        yo.release(1)
        wi.release(1)
        xi.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), wf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, w, y, xp, wp, yc):
        xt = TensorAccessPattern((1, 2 * hidden), 0,
                                 [1, 1, 1, 2 * hidden], [0, 0, 0, 1])
        wt = TensorAccessPattern((1, hidden), 0,
                                 [1, 1, 1, hidden], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, wt)
        pipe.fill(xp, x, xt)
        pipe.fill(wp, w, wt)
        pipe.finish()

    runtime = Runtime(sequence, [pair_ty, row_ty, row_ty,
                                 xf.prod(), wf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_hidden_norm
_src = (HERE / "norm.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {
    "hidden": int(os.environ.get("PHLEGM_HIDDEN", "1536")),
    "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16),
}
