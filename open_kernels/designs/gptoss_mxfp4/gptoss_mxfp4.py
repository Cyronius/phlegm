"""Decode one GPT-OSS MXFP4 container chunk to row-major fp32 weights.

One 2560-byte chunk covers 32 output rows by 128 input columns. This is a
standalone primitive; expert GEMV tiling and runtime packing are separate.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_mxfp4(chunk: In, values: Out, *, srchash: CompileTime[int] = 0):
    chunk_ty = np.ndarray[(2560,), np.dtype[np.uint8]]
    values_ty = np.ndarray[(32 * 128,), np.dtype[np.float32]]
    fn = ExternalFunction("gptoss_mxfp4", source_file=str(HERE / "gptoss_mxfp4.cc"),
                          arg_types=[chunk_ty, values_ty], include_dirs=include_dirs())
    chunks = ObjectFifo(chunk_ty, name="chunks", depth=1)
    decoded = ObjectFifo(values_ty, name="decoded", depth=1)

    def core_body(inp, out, decode):
        c = inp.acquire(1)
        v = out.acquire(1)
        decode(c, v)
        out.release(1)
        inp.release(1)

    worker = Worker(core_body, fn_args=[chunks.cons(), decoded.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(c, v, inp, out):
        chunk_tap = TensorAccessPattern((1, 2560), 0, [1, 1, 1, 2560], [0, 0, 0, 1])
        values_tap = TensorAccessPattern((1, 4096), 0, [1, 1, 1, 4096], [0, 0, 0, 1])
        pipe = Pipeline(2)
        pipe.drain(out, v, values_tap)
        pipe.fill(inp, c, chunk_tap)
        pipe.finish()

    runtime = Runtime(sequence, [chunk_ty, values_ty, chunks.prod(), decoded.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_mxfp4
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "gptoss_mxfp4.cc").read_bytes()).hexdigest()[:8], 16)}
