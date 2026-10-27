"""Finish one 256-patch Gemma 3 SigLIP patch-embedding panel on NPU."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

ROWS, PADDED, HIDDEN = 256, 1280, 1152


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma3_vision_patch_finish(projected: In, addends: In, output: Out,
                               *, srchash: CompileTime[int] = 0):
    projected_row = np.ndarray[(PADDED,), np.dtype[np.float32]]
    vision_row = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    addends_row = np.ndarray[(2 * HIDDEN,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma3_vision_patch_finish_row",
                          source_file=str(HERE / "finish.cc"),
                          arg_types=[projected_row, addends_row, vision_row],
                          include_dirs=include_dirs(),
                          compile_flags=["-ffp-contract=off"])
    pf = ObjectFifo(projected_row, name="projected", depth=1)
    af = ObjectFifo(addends_row, name="addends", depth=1)
    of = ObjectFifo(vision_row, name="output", depth=1)

    def body(pi, ai, oo, finish):
        for _ in range_(ROWS):
            p = pi.acquire(1)
            a = ai.acquire(1)
            o = oo.acquire(1)
            finish(p, a, o)
            pi.release(1)
            ai.release(1)
            oo.release(1)

    worker = Worker(body, fn_args=[pf.cons(), af.cons(), of.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(p, a, o, pp, ap, oc):
        pt = TensorAccessPattern((1, ROWS * PADDED), 0,
                                 [1, 1, ROWS, PADDED], [0, 0, PADDED, 1])
        at = TensorAccessPattern((1, ROWS * 2 * HIDDEN), 0,
                                 [1, 1, ROWS, 2 * HIDDEN], [0, 0, 2 * HIDDEN, 1])
        xt = TensorAccessPattern((1, ROWS * HIDDEN), 0,
                                 [1, 1, ROWS, HIDDEN], [0, 0, HIDDEN, 1])
        pipe = Pipeline(3)
        pipe.drain(oc, o, xt)
        pipe.fill(pp, p, pt)
        pipe.fill(ap, a, at)
        pipe.finish()

    runtime = Runtime(sequence, [np.ndarray[(ROWS * PADDED,), np.dtype[np.float32]],
                                 np.ndarray[(ROWS * 2 * HIDDEN,), np.dtype[bfloat16]],
                                 np.ndarray[(ROWS * HIDDEN,), np.dtype[bfloat16]],
                                 pf.prod(), af.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma3_vision_patch_finish
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "finish.cc").read_bytes()).hexdigest()[:8], 16)}
