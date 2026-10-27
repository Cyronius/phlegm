"""Stream Gemma 4 MLP GELU-tanh(gate) * up in 256-value BF16 chunks."""
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
GATE = HERE.parent / "gemma4_ple_gate" / "gate.cc"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

CHUNK = 256
WIDTHS = (3072, 6144, 10240, 12288, 15360)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_gelu_mul(gate: In, up: In, output: Out,
                    *, width: CompileTime[int], srchash: CompileTime[int] = 0):
    assert width in WIDTHS
    chunks = width // CHUNK
    full_ty = np.ndarray[(width,), np.dtype[bfloat16]]
    chunk_ty = np.ndarray[(CHUNK,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma4_ple_gate", source_file=str(GATE),
                          arg_types=[chunk_ty, chunk_ty, chunk_ty],
                          include_dirs=include_dirs(), compile_flags=["-ffp-contract=off"])
    gf = ObjectFifo(chunk_ty, name="gate", depth=2)
    uf = ObjectFifo(chunk_ty, name="up", depth=2)
    yf = ObjectFifo(chunk_ty, name="output", depth=2)

    def core_body(gi, ui, yo, activation):
        for _ in range_(chunks):
            g = gi.acquire(1)
            u = ui.acquire(1)
            y = yo.acquire(1)
            activation(g, u, y)
            yo.release(1)
            ui.release(1)
            gi.release(1)

    worker = Worker(core_body, fn_args=[gf.cons(), uf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(g, u, y, gp, up, yc):
        tap = TensorAccessPattern((1, width), 0,
                                  [1, 1, chunks, CHUNK], [0, 0, CHUNK, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, tap)
        pipe.fill(gp, g, tap)
        pipe.fill(up, u, tap)
        pipe.finish()

    runtime = Runtime(sequence, [full_ty, full_ty, full_ty,
                                 gf.prod(), uf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_gelu_mul
SPECIALIZE = {
    "width": int(os.environ.get("PHLEGM_WIDTH", "6144")),
    "srchash": int(hashlib.sha1(GATE.read_bytes()).hexdigest()[:8], 16),
}
