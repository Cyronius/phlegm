"""Gemma 4 audio SiLU and GLU BF16 activations at model widths."""
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
def gemma4_audio_act(input: In, output: Out,
                     *, width: CompileTime[int], glu: CompileTime[bool],
                     srchash: CompileTime[int] = 0):
    assert (glu and width == 1024) or (not glu and width in (1024, 4096))
    factor = 2 if glu else 1
    chunks = width // 256
    input_ty = np.ndarray[(256 * factor,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(256,), np.dtype[bfloat16]]
    flags = ["-ffp-contract=off"] + (["-DGEMMA4_AUDIO_GLU"] if glu else [])
    fn = ExternalFunction("gemma4_audio_act_chunk", source_file=str(HERE / "act.cc"),
                          arg_types=[input_ty, output_ty], include_dirs=include_dirs(),
                          compile_flags=flags)
    xf = ObjectFifo(input_ty, name="input", depth=2)
    yf = ObjectFifo(output_ty, name="output", depth=2)

    def core_body(xi, yo, activation):
        for _ in range_(chunks):
            x = xi.acquire(1)
            y = yo.acquire(1)
            activation(x, y)
            xi.release(1)
            yo.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, y, xp, yc):
        itap = TensorAccessPattern((1, width * factor), 0,
                                   [1, 1, chunks, 256 * factor],
                                   [0, 0, 256 * factor, 1])
        otap = TensorAccessPattern((1, width), 0,
                                   [1, 1, chunks, 256], [0, 0, 256, 1])
        pipe = Pipeline(2)
        pipe.drain(yc, y, otap)
        pipe.fill(xp, x, itap)
        pipe.finish()

    runtime = Runtime(sequence,
                      [np.ndarray[(width * factor,), np.dtype[bfloat16]],
                       np.ndarray[(width,), np.dtype[bfloat16]], xf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_audio_act
_src = (HERE / "act.cc").read_bytes() + (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"width": int(os.environ.get("PHLEGM_WIDTH", "1024")),
              "glu": bool(int(os.environ.get("PHLEGM_GLU", "0"))),
              "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
