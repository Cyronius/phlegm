"""GPT-OSS 20B one-token RMSNorm, FP32 residual to BF16 projection input."""
from __future__ import annotations

import hashlib
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

HIDDEN = 2880


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_norm(residual: In, weight: In, result: Out,
                *, srchash: CompileTime[int] = 0):
    x_ty = np.ndarray[(HIDDEN,), np.dtype[np.float32]]
    w_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    fn = ExternalFunction("gptoss_norm", source_file=str(HERE / "norm.cc"),
                          arg_types=[x_ty, w_ty, w_ty], include_dirs=include_dirs(),
                          compile_flags=["-ffp-contract=off"])
    xf = ObjectFifo(x_ty, name="residual", depth=1)
    wf = ObjectFifo(w_ty, name="weight", depth=1)
    of = ObjectFifo(w_ty, name="result", depth=1)

    def core_body(xi, wi, yo, norm):
        x = xi.acquire(1)
        w = wi.acquire(1)
        y = yo.acquire(1)
        norm(x, w, y)
        yo.release(1)
        wi.release(1)
        xi.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), wf.cons(), of.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(x, w, y, xp, wp, yc):
        xtap = TensorAccessPattern((1, HIDDEN), 0,
                                   [1, 1, 1, HIDDEN], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, xtap)
        pipe.fill(xp, x, xtap)
        pipe.fill(wp, w, xtap)
        pipe.finish()

    runtime = Runtime(sequence, [x_ty, w_ty, w_ty,
                                 xf.prod(), wf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_norm
_src = (HERE / "norm.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
