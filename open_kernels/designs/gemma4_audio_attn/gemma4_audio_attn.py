"""Gemma 4 audio conformer 8-head 12-query/24-context attention on NPU2."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

INPUT_BYTES = 37664
OUTPUT_FLOATS = 12 * 128
HEADS = 8


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_audio_attn(packed: In, output: Out, *, srchash: CompileTime[int] = 0):
    input_ty = np.ndarray[(INPUT_BYTES,), np.dtype[np.uint8]]
    output_ty = np.ndarray[(OUTPUT_FLOATS,), np.dtype[np.float32]]
    fn = ExternalFunction("gemma4_audio_attention_head", source_file=str(HERE / "attn.cc"),
                          arg_types=[input_ty, output_ty], include_dirs=include_dirs(),
                          compile_flags=["-ffp-contract=off"])
    xf = ObjectFifo(input_ty, name="packed", depth=1)
    yf = ObjectFifo(output_ty, name="output", depth=1)

    def core_body(xi, yo, kernel):
        for _ in range_(HEADS):
            x = xi.acquire(1)
            y = yo.acquire(1)
            kernel(x, y)
            xi.release(1)
            yo.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, y, xp, yc):
        itap = TensorAccessPattern((1, HEADS * INPUT_BYTES), 0,
                                   [1, 1, HEADS, INPUT_BYTES], [0, 0, INPUT_BYTES, 1])
        otap = TensorAccessPattern((1, HEADS * OUTPUT_FLOATS), 0,
                                   [1, 1, HEADS, OUTPUT_FLOATS], [0, 0, OUTPUT_FLOATS, 1])
        pipe = Pipeline(2)
        pipe.drain(yc, y, otap)
        pipe.fill(xp, x, itap)
        pipe.finish()

    runtime = Runtime(sequence,
                      [np.ndarray[(HEADS * INPUT_BYTES,), np.dtype[np.uint8]],
                       np.ndarray[(HEADS * OUTPUT_FLOATS,), np.dtype[np.float32]],
                       xf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_audio_attn
_src = (HERE / "attn.cc").read_bytes() + (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
