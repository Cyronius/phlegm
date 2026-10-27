"""Gemma 4 audio depthwise valid 5-tap BF16 convolution on packed panels.

One call processes 256 output frames and 1024 channels: 128 packed panels,
each [36,64] input, [5,64] weights, and [32,64] FP32 output. The host packs
time/channel panels with ``open_kernels.gemma4_audio_dw_pack``.
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

PATCHES = 128
IN_ELEMENTS = 36 * 64
WEIGHT_ELEMENTS = 5 * 64
OUT_ELEMENTS = 32 * 64


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_audio_dw(panels: In, weights: In, output: Out, *,
                    srchash: CompileTime[int] = 0):
    in_ty = np.ndarray[(PATCHES * IN_ELEMENTS,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(PATCHES * WEIGHT_ELEMENTS,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(PATCHES * OUT_ELEMENTS,), np.dtype[np.float32]]
    in_patch = np.ndarray[(IN_ELEMENTS,), np.dtype[bfloat16]]
    w_patch = np.ndarray[(WEIGHT_ELEMENTS,), np.dtype[bfloat16]]
    out_patch = np.ndarray[(OUT_ELEMENTS,), np.dtype[np.float32]]
    fn = ExternalFunction("gemma4_audio_dw_step", source_file=str(HERE / "step.cc"),
                          arg_types=[in_patch, w_patch, out_patch],
                          include_dirs=include_dirs(), compile_flags=["-ffp-contract=off"])
    xf = ObjectFifo(in_patch, name="panels", depth=2)
    wf = ObjectFifo(w_patch, name="weights", depth=2)
    yf = ObjectFifo(out_patch, name="output", depth=2)

    def body(xi, wi, yo, step):
        for _ in range_(PATCHES):
            x = xi.acquire(1)
            w = wi.acquire(1)
            y = yo.acquire(1)
            step(x, w, y)
            xi.release(1)
            wi.release(1)
            yo.release(1)

    worker = Worker(body, fn_args=[xf.cons(), wf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, w, y, xp, wp, yc):
        xt = TensorAccessPattern((1, PATCHES * IN_ELEMENTS), 0,
                                 [1, 1, PATCHES, IN_ELEMENTS], [0, 0, IN_ELEMENTS, 1])
        wt = TensorAccessPattern((1, PATCHES * WEIGHT_ELEMENTS), 0,
                                 [1, 1, PATCHES, WEIGHT_ELEMENTS], [0, 0, WEIGHT_ELEMENTS, 1])
        yt = TensorAccessPattern((1, PATCHES * OUT_ELEMENTS), 0,
                                 [1, 1, PATCHES, OUT_ELEMENTS], [0, 0, OUT_ELEMENTS, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, yt)
        pipe.fill(xp, x, xt)
        pipe.fill(wp, w, wt)
        pipe.finish()

    runtime = Runtime(sequence, [in_ty, w_ty, out_ty,
                                 xf.prod(), wf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_audio_dw
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "step.cc").read_bytes()).hexdigest()[:8], 16)}
