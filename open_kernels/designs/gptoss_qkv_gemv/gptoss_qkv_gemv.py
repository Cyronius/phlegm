"""GPT-OSS 20B q4_1 Q/K/V projection, with per-channel bias.

Each chunk is 32 output rows by 128 input channels. The container has 23
chunks per band (2944 channels), with its last 64 inputs padded to zero.
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
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

BANDS = int(os.environ.get("GPTOSS_QKV_BANDS", "16"))
K_BLOCKS = 23
CHUNK = 2560


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_qkv_gemv(x_bias: In, chunks: In, result: Out,
                    *, bands: CompileTime[int], srchash: CompileTime[int] = 0):
    assert bands in (1, 16)
    x_ty = np.ndarray[(2944 + bands * 32,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(bands * K_BLOCKS * CHUNK,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(bands * 32,), np.dtype[np.float32]]
    chunk_ty = np.ndarray[(CHUNK,), np.dtype[np.uint8]]
    band_ty = np.ndarray[(32,), np.dtype[np.float32]]
    step = ExternalFunction("gptoss_qkv_step", source_file=str(HERE / "step.cc"),
                            arg_types=[chunk_ty, x_ty, band_ty, np.int32, np.int32],
                            include_dirs=include_dirs())
    xf = ObjectFifo(x_ty, name="x_bias", depth=1)
    wf = ObjectFifo(chunk_ty, name="chunks", depth=2)
    yf = ObjectFifo(band_ty, name="result", depth=1)

    def core_body(xi, wi, yo, fn):
        xv = xi.acquire(1)
        for band in range_(bands):
            yv = yo.acquire(1)
            for kb in range_(K_BLOCKS):
                wv = wi.acquire(1)
                fn(wv, xv, yv, band, kb)
                wi.release(1)
            yo.release(1)
        xi.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), wf.cons(), yf.prod(), step],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(x, w, y, xp, wp, yc):
        def tap(n):
            return TensorAccessPattern((1, n), 0, [1, 1, 1, n], [0, 0, 0, 1])
        wtap = TensorAccessPattern((1, bands * K_BLOCKS * CHUNK), 0,
                                   [1, 1, bands * K_BLOCKS, CHUNK], [0, 0, CHUNK, 1])
        ytap = TensorAccessPattern((1, bands * 32), 0,
                                   [1, 1, bands, 32], [0, 0, 32, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, ytap)
        pipe.fill(xp, x, tap(2944 + bands * 32))
        pipe.fill(wp, w, wtap)
        pipe.finish()

    runtime = Runtime(sequence, [x_ty, w_ty, y_ty,
                                 xf.prod(), wf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_qkv_gemv
SPECIALIZE = {"bands": BANDS,
              "srchash": int(hashlib.sha1((HERE / "step.cc").read_bytes()).hexdigest()[:8], 16)}
