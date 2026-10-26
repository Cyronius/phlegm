"""Standalone GPT-OSS expert activation: fp32 gate/up[64] to bf16 hidden[64].

The projection biases must already be included in the input vectors.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N = 64


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_gate(gate: In, up: In, hidden: Out, *, srchash: CompileTime[int] = 0):
    in_ty = np.ndarray[(N,), np.dtype[np.float32]]
    out_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    fn = ExternalFunction("gptoss_gate", source_file=str(HERE / "gptoss_gate.cc"),
                          arg_types=[in_ty, in_ty, out_ty], include_dirs=include_dirs())
    inputs = ObjectFifo(in_ty, name="inputs", depth=2)
    outputs = ObjectFifo(out_ty, name="outputs", depth=1)

    def core_body(inp, out, gate_fn):
        pair = inp.acquire(2)
        result = out.acquire(1)
        gate_fn(pair[0], pair[1], result)
        out.release(1)
        inp.release(2)

    worker = Worker(core_body, fn_args=[inputs.cons(), outputs.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(g, u, h, inp, out):
        whole = TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(out, h, whole)
        pipe.fill(inp, g, whole)
        pipe.fill(inp, u, whole)
        pipe.finish()

    runtime = Runtime(sequence, [in_ty, in_ty, out_ty, inputs.prod(), outputs.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_gate
_src = (HERE / "gptoss_gate.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
