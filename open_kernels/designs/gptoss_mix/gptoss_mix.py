"""Mix four routed GPT-OSS expert outputs into one 2880-value FP32 vector.

Input rows are already ordered by the router's int32[4] IDs. The router's
FP32[4] weights occupy bytes 16..31 of its 256-byte output.
"""
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

BLOCKS = 45
ROWS = 64
EXPERTS = 4


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_mix(experts: In, route: In, result: Out,
               *, srchash: CompileTime[int] = 0):
    expert_ty = np.ndarray[(BLOCKS * EXPERTS * ROWS,), np.dtype[np.float32]]
    route_ty = np.ndarray[(256,), np.dtype[np.uint8]]
    result_ty = np.ndarray[(BLOCKS * ROWS,), np.dtype[np.float32]]
    input_block = np.ndarray[(EXPERTS * ROWS,), np.dtype[np.float32]]
    output_block = np.ndarray[(ROWS,), np.dtype[np.float32]]
    fn = ExternalFunction("gptoss_mix", source_file=str(HERE / "mix.cc"),
                          arg_types=[input_block, route_ty, output_block],
                          include_dirs=include_dirs())
    ef = ObjectFifo(input_block, name="experts", depth=2)
    rf = ObjectFifo(route_ty, name="route", depth=1)
    yf = ObjectFifo(output_block, name="result", depth=1)

    def core_body(ei, ri, yo, mix_fn):
        rv = ri.acquire(1)
        for _ in range_(BLOCKS):
            ev = ei.acquire(1)
            yv = yo.acquire(1)
            mix_fn(ev, rv, yv)
            yo.release(1)
            ei.release(1)
        ri.release(1)

    worker = Worker(core_body, fn_args=[ef.cons(), rf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(e, r, y, ep, rp, yc):
        et = TensorAccessPattern((1, BLOCKS * EXPERTS * ROWS), 0,
                                 [1, 1, BLOCKS, EXPERTS * ROWS],
                                 [0, 0, EXPERTS * ROWS, 1])
        rt = TensorAccessPattern((1, 256), 0, [1, 1, 1, 256], [0, 0, 0, 1])
        yt = TensorAccessPattern((1, BLOCKS * ROWS), 0,
                                 [1, 1, BLOCKS, ROWS], [0, 0, ROWS, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, yt)
        pipe.fill(rp, r, rt)
        pipe.fill(ep, e, et)
        pipe.finish()

    runtime = Runtime(sequence, [expert_ty, route_ty, result_ty,
                                 ef.prod(), rf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_mix
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "mix.cc").read_bytes()).hexdigest()[:8], 16)}
