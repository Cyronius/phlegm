"""GPT-OSS expert activation in 64-lane blocks, up to a full 2880-lane vector.

The projection biases must already be included in the input vectors.
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
from aie.iron.device import Tile
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N = 64


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_gate(gate: In, up: In, hidden: Out, *, blocks: CompileTime[int],
                srchash: CompileTime[int] = 0):
    assert blocks in (1, 45)
    in_ty = np.ndarray[(blocks * N,), np.dtype[np.float32]]
    out_ty = np.ndarray[(blocks * N,), np.dtype[bfloat16]]
    in_block = np.ndarray[(N,), np.dtype[np.float32]]
    out_block = np.ndarray[(N,), np.dtype[bfloat16]]
    fn = ExternalFunction("gptoss_gate", source_file=str(HERE / "gptoss_gate.cc"),
                          arg_types=[in_block, in_block, out_block], include_dirs=include_dirs())
    gates = ObjectFifo(in_block, name="gates", depth=2)
    ups = ObjectFifo(in_block, name="ups", depth=2)
    outputs = ObjectFifo(out_block, name="outputs", depth=1)

    def core_body(gi, ui, out, gate_fn):
        for _ in range_(blocks):
            g = gi.acquire(1)
            u = ui.acquire(1)
            result = out.acquire(1)
            gate_fn(g, u, result)
            out.release(1)
            ui.release(1)
            gi.release(1)

    worker = Worker(core_body, fn_args=[gates.cons(), ups.cons(), outputs.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(g, u, h, gp, up, out):
        whole = TensorAccessPattern((1, blocks * N), 0,
                                    [1, 1, blocks, N], [0, 0, N, 1])
        pipe = Pipeline(3)
        pipe.drain(out, h, whole)
        pipe.fill(gp, g, whole)
        pipe.fill(up, u, whole)
        pipe.finish()

    runtime = Runtime(sequence, [in_ty, in_ty, out_ty,
                                 gates.prod(), ups.prod(), outputs.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_gate
_src = (HERE / "gptoss_gate.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"blocks": int(os.environ.get("GPTOSS_GATE_BLOCKS", "1")),
              "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
