"""Gemma 4 per-layer BF16 GEMV from FLM's [N/32,K,32] source layout.

This serves inp_gate, per_layer_projection, and per-layer model projection.
It streams one 256-column by 32-row weight block at a time; the activation
stays in one AIE tile and each 32-row output accumulates in FP32.
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

SHAPES = ((256, 1536), (256, 2560), (1536, 256), (2560, 256))
KB = 256
NB = 32


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_bf16_gemv(weight: In, activation: In, output: Out,
                     *, n: CompileTime[int], k: CompileTime[int],
                     srchash: CompileTime[int] = 0):
    assert (n, k) in SHAPES
    blocks, pieces = n // NB, k // KB
    w_ty = np.ndarray[(n * k,), np.dtype[bfloat16]]
    x_ty = np.ndarray[(k,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(n,), np.dtype[np.float32]]
    tile_ty = np.ndarray[(KB * NB,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(NB,), np.dtype[np.float32]]
    fn = ExternalFunction("gemma4_bf16_gemv_step", source_file=str(HERE / "step.cc"),
                          arg_types=[tile_ty, x_ty, out_ty, np.int32],
                          include_dirs=include_dirs(), compile_flags=["-ffp-contract=off"])
    wf = ObjectFifo(tile_ty, name="weight", depth=2)
    xf = ObjectFifo(x_ty, name="activation", depth=1)
    yf = ObjectFifo(out_ty, name="output", depth=2)

    def core_body(wi, xi, yo, step):
        x = xi.acquire(1)
        for _ in range_(blocks):
            y = yo.acquire(1)
            for piece in range_(pieces):
                w = wi.acquire(1)
                step(w, x, y, piece * KB)
                wi.release(1)
            yo.release(1)
        xi.release(1)

    worker = Worker(core_body, fn_args=[wf.cons(), xf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(w, x, y, wp, xp, yc):
        wt = TensorAccessPattern((1, n * k), 0,
                                 [1, 1, blocks * pieces, KB * NB], [0, 0, KB * NB, 1])
        xt = TensorAccessPattern((1, k), 0, [1, 1, 1, k], [0, 0, 0, 1])
        yt = TensorAccessPattern((1, n), 0,
                                 [1, 1, blocks, NB], [0, 0, NB, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, yt)
        pipe.fill(wp, w, wt)
        pipe.fill(xp, x, xt)
        pipe.finish()

    runtime = Runtime(sequence, [w_ty, x_ty, y_ty,
                                 wf.prod(), xf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_bf16_gemv
SPECIALIZE = {
    "n": int(os.environ.get("PHLEGM_N", "256")),
    "k": int(os.environ.get("PHLEGM_K", "1536")),
    "srchash": int(hashlib.sha1((HERE / "step.cc").read_bytes()).hexdigest()[:8], 16),
}
