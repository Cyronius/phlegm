"""One GPT-OSS MXFP4 expert projection, 2880 input channels to 2880 outputs.

The source is packed as 32-output-row bands by 23 input chunks. The final
64 input lanes are padding. Set GEG_BANDS=1 for a short diagnostic build;
the production geometry is 90 bands.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

BANDS = int(os.environ.get("GEG_BANDS", "90"))
assert BANDS in (1, 90)
K_BLOCKS = 23
CHUNK = 2560


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_expert_gemv(x: In, chunks: In, result: Out,
                       *, bands: CompileTime[int], srchash: CompileTime[int] = 0):
    assert bands in (1, 90)
    x_ty = np.ndarray[(K_BLOCKS * 128,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(bands * K_BLOCKS * CHUNK,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(bands * 32,), np.dtype[np.float32]]
    chunk_ty = np.ndarray[(CHUNK,), np.dtype[np.uint8]]
    band_ty = np.ndarray[(32,), np.dtype[np.float32]]
    bias_ty = np.ndarray[(32,), np.dtype[bfloat16]]
    step = ExternalFunction("gptoss_expert_step", source_file=str(HERE / "step.cc"),
                            arg_types=[chunk_ty, x_ty, band_ty, bias_ty, np.int32],
                            include_dirs=include_dirs())
    xf = ObjectFifo(x_ty, name="x", depth=1)
    wf = ObjectFifo(chunk_ty, name="chunks", depth=2)
    yf = ObjectFifo(band_ty, name="result", depth=1)

    def core_body(xi, wi, yo, bias, fn):
        xv = xi.acquire(1)
        for _ in range_(bands):
            yv = yo.acquire(1)
            for kb in range_(K_BLOCKS):
                wv = wi.acquire(1)
                fn(wv, xv, yv, bias, kb)
                wi.release(1)
            yo.release(1)
        xi.release(1)

    worker = Worker(core_body,
                    fn_args=[xf.cons(), wf.cons(), yf.prod(),
                             Buffer(bias_ty, name="bias"), step],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(x_arg, w_arg, y_arg, xp, wp, yc):
        def tap(n):
            return TensorAccessPattern((1, n), 0, [1, 1, 1, n], [0, 0, 0, 1])
        w_tap = TensorAccessPattern((1, bands * K_BLOCKS * CHUNK), 0,
                                    [1, 1, bands * K_BLOCKS, CHUNK],
                                    [0, 0, CHUNK, 1])
        y_tap = TensorAccessPattern((1, bands * 32), 0,
                                    [1, 1, bands, 32], [0, 0, 32, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y_arg, y_tap)
        pipe.fill(xp, x_arg, tap(K_BLOCKS * 128))
        pipe.fill(wp, w_arg, w_tap)
        pipe.finish()

    runtime = Runtime(sequence, [x_ty, w_ty, y_ty,
                                 xf.prod(), wf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_expert_gemv
_src = (HERE / "step.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"bands": BANDS, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
