"""Gemma 4 audio 128/32-channel subsampler LayerNorm and ReLU."""
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
def gemma4_audio_subnorm(packed: In, output: Out,
                         *, channels: CompileTime[int], srchash: CompileTime[int] = 0):
    assert channels in (32, 128)
    rows = 256
    packet = channels * 6
    input_ty = np.ndarray[(packet,), np.dtype[np.uint8]]
    output_ty = np.ndarray[(channels,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma4_audio_subnorm", source_file=str(HERE / "norm.cc"),
                          arg_types=[input_ty, output_ty], include_dirs=include_dirs(),
                          compile_flags=[f"-DCHANNELS={channels}", "-ffp-contract=off"])
    xf = ObjectFifo(input_ty, name="input", depth=2)
    yf = ObjectFifo(output_ty, name="output", depth=2)

    def core_body(xi, yo, norm):
        for _ in range_(rows):
            x = xi.acquire(1)
            y = yo.acquire(1)
            norm(x, y)
            xi.release(1)
            yo.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, y, xp, yc):
        itap = TensorAccessPattern((1, rows * packet), 0,
                                   [1, 1, rows, packet], [0, 0, packet, 1])
        otap = TensorAccessPattern((1, rows * channels), 0,
                                   [1, 1, rows, channels], [0, 0, channels, 1])
        pipe = Pipeline(2)
        pipe.drain(yc, y, otap)
        pipe.fill(xp, x, itap)
        pipe.finish()

    runtime = Runtime(sequence,
                      [np.ndarray[(rows * packet,), np.dtype[np.uint8]],
                       np.ndarray[(rows * channels,), np.dtype[bfloat16]],
                       xf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_audio_subnorm
_src = (HERE / "norm.cc").read_bytes() + (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"channels": int(os.environ.get("PHLEGM_CHANNELS", "128")),
              "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
