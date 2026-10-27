"""Gemma 4 E2B/E4B per-layer input projection norm and token mix.

Input is BF16[256] context projection followed by BF16[256] token embedding.
The context is scaled by hidden_size**-0.5 before RMSNorm. The token path
combines it with the embedding and scales by 2**-0.5; the no-token variant
returns the normalized context for multimodal inputs.
"""
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

WIDTH = 256


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_ple_input(context_token: In, scale: In, output: Out,
                     *, hidden: CompileTime[int], has_token: CompileTime[int],
                     srchash: CompileTime[int] = 0):
    assert hidden in (1536, 2560) and has_token in (0, 1)
    pair_ty = np.ndarray[(2 * WIDTH,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(WIDTH,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma4_ple_input", source_file=str(HERE / "input.cc"),
                          arg_types=[pair_ty, row_ty, row_ty], include_dirs=include_dirs(),
                          compile_flags=[f"-DHIDDEN={hidden}", f"-DHAS_TOKEN={has_token}",
                                         "-ffp-contract=off"])
    pair = ObjectFifo(pair_ty, name="context_token", depth=1)
    weight = ObjectFifo(row_ty, name="scale", depth=1)
    result = ObjectFifo(row_ty, name="output", depth=1)

    def core_body(pi, wi, yo, compute):
        p = pi.acquire(1)
        w = wi.acquire(1)
        y = yo.acquire(1)
        compute(p, w, y)
        yo.release(1)
        wi.release(1)
        pi.release(1)

    worker = Worker(core_body, fn_args=[pair.cons(), weight.cons(), result.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(p, w, y, pp, wp, yc):
        pair_tap = TensorAccessPattern((1, 2 * WIDTH), 0,
                                      [1, 1, 1, 2 * WIDTH], [0, 0, 0, 1])
        row_tap = TensorAccessPattern((1, WIDTH), 0,
                                     [1, 1, 1, WIDTH], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, row_tap)
        pipe.fill(pp, p, pair_tap)
        pipe.fill(wp, w, row_tap)
        pipe.finish()

    runtime = Runtime(sequence, [pair_ty, row_ty, row_ty,
                                 pair.prod(), weight.prod(), result.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_ple_input
SPECIALIZE = {
    "hidden": int(os.environ.get("PHLEGM_HIDDEN", "1536")),
    "has_token": int(os.environ.get("PHLEGM_HAS_TOKEN", "1")),
    "srchash": int(hashlib.sha1((HERE / "input.cc").read_bytes()).hexdigest()[:8], 16),
}
