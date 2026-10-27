"""Gemma 4 vision/audio 256-row clippable-linear BF16 boundaries."""
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

ROWS = 256


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_clip(values: In, bounds: In, output: Out,
                *, width: CompileTime[int], post_linear: CompileTime[int],
                srchash: CompileTime[int] = 0):
    assert width in (256, 768, 1024, 1536, 2048, 2560, 3072, 3840, 4096)
    assert post_linear in (0, 1)
    input_dtype = np.float32 if post_linear else bfloat16
    in_ty = np.ndarray[(width,), np.dtype[input_dtype]]
    out_ty = np.ndarray[(width,), np.dtype[bfloat16]]
    bounds_ty = np.ndarray[(2,), np.dtype[np.float32]]
    in_panel = np.ndarray[(ROWS * width,), np.dtype[input_dtype]]
    out_panel = np.ndarray[(ROWS * width,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma4_clip_row", source_file=str(HERE / "clip.cc"),
                          arg_types=[in_ty, bounds_ty, out_ty],
                          include_dirs=include_dirs(),
                          compile_flags=[f"-DWIDTH={width}", f"-DPOST_LINEAR={post_linear}",
                                         "-ffp-contract=off"])
    vf = ObjectFifo(in_ty, name="values", depth=1)
    bf = ObjectFifo(bounds_ty, name="bounds", depth=1)
    of = ObjectFifo(out_ty, name="output", depth=1)

    def body(vi, bi, oo, clip):
        b = bi.acquire(1)
        for _ in range_(ROWS):
            v = vi.acquire(1)
            o = oo.acquire(1)
            clip(v, b, o)
            vi.release(1)
            oo.release(1)
        bi.release(1)

    worker = Worker(body, fn_args=[vf.cons(), bf.cons(), of.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(v, b, o, vp, bp, oc):
        itap = TensorAccessPattern((1, ROWS * width), 0,
                                   [1, 1, ROWS, width], [0, 0, width, 1])
        otap = TensorAccessPattern((1, ROWS * width), 0,
                                   [1, 1, ROWS, width], [0, 0, width, 1])
        btap = TensorAccessPattern((1, 2), 0, [1, 1, 1, 2], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(oc, o, otap)
        pipe.fill(vp, v, itap)
        pipe.fill(bp, b, btap)
        pipe.finish()

    runtime = Runtime(sequence, [in_panel, bounds_ty, out_panel,
                                 vf.prod(), bf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_clip
SPECIALIZE = {
    "width": int(os.environ.get("PHLEGM_WIDTH", "1024")),
    "post_linear": int(os.environ.get("PHLEGM_POST_LINEAR", "0")),
    "srchash": int(hashlib.sha1((HERE / "clip.cc").read_bytes()).hexdigest()[:8], 16),
}
