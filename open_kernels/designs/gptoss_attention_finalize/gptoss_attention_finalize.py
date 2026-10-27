"""Normalize GPT-OSS long-context attention state for the BF16 output GEMV."""
from __future__ import annotations

import hashlib
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

HEADS = 64
STATE_FLOATS = 68
HEAD_DIM = 64


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_attention_finalize(state: In, output: Out,
                              *, srchash: CompileTime[int] = 0):
    state_head = np.ndarray[(STATE_FLOATS,), np.dtype[np.float32]]
    output_head = np.ndarray[(HEAD_DIM,), np.dtype[bfloat16]]
    state_all = np.ndarray[(HEADS * STATE_FLOATS,), np.dtype[np.float32]]
    output_all = np.ndarray[(HEADS * HEAD_DIM,), np.dtype[bfloat16]]
    finish = ExternalFunction("gptoss_attention_finalize_head",
                              source_file=str(HERE / "finalize.cc"),
                              arg_types=[state_head, output_head],
                              include_dirs=include_dirs(),
                              compile_flags=["-ffp-contract=off"])
    sf = ObjectFifo(state_head, name="state", depth=1)
    of = ObjectFifo(output_head, name="output", depth=1)

    def core_body(si, oo, fn):
        for _ in range_(HEADS):
            s = si.acquire(1)
            o = oo.acquire(1)
            fn(s, o)
            si.release(1)
            oo.release(1)

    worker = Worker(core_body, fn_args=[sf.cons(), of.prod(), finish],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(s, o, sp, oc):
        st = TensorAccessPattern((1, HEADS * STATE_FLOATS), 0,
                                 [1, 1, HEADS, STATE_FLOATS],
                                 [0, 0, STATE_FLOATS, 1])
        ot = TensorAccessPattern((1, HEADS * HEAD_DIM), 0,
                                 [1, 1, HEADS, HEAD_DIM],
                                 [0, 0, HEAD_DIM, 1])
        pipe = Pipeline(3)
        pipe.drain(oc, o, ot)
        pipe.fill(sp, s, st)
        pipe.finish()

    runtime = Runtime(sequence, [state_all, output_all, sf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_attention_finalize
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "finalize.cc").read_bytes()).hexdigest()[:8], 16)}
