"""SigLIP 4096-key scaled BF16 softmax for 256 query rows on NPU."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

ROWS, KEYS = 256, 4096


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma3_vision_softmax(scores: In, probabilities: Out,
                          *, srchash: CompileTime[int] = 0):
    score_row = np.ndarray[(KEYS,), np.dtype[np.float32]]
    prob_row = np.ndarray[(KEYS,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma3_vision_softmax_row",
                          source_file=str(HERE / "softmax.cc"),
                          arg_types=[score_row, prob_row],
                          include_dirs=include_dirs(), compile_flags=["-ffp-contract=off"])
    sf = ObjectFifo(score_row, name="scores", depth=1)
    pf = ObjectFifo(prob_row, name="probabilities", depth=1)

    def body(si, po, softmax):
        for _ in range_(ROWS):
            s = si.acquire(1)
            p = po.acquire(1)
            softmax(s, p)
            si.release(1)
            po.release(1)

    worker = Worker(body, fn_args=[sf.cons(), pf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(s, p, sp, pc):
        st = TensorAccessPattern((1, ROWS * KEYS), 0,
                                 [1, 1, ROWS, KEYS], [0, 0, KEYS, 1])
        pipe = Pipeline(2)
        pipe.drain(pc, p, st)
        pipe.fill(sp, s, st)
        pipe.finish()

    runtime = Runtime(sequence,
                      [np.ndarray[(ROWS * KEYS,), np.dtype[np.float32]],
                       np.ndarray[(ROWS * KEYS,), np.dtype[bfloat16]],
                       sf.prod(), pf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma3_vision_softmax
_src = (HERE / "softmax.cc").read_bytes() + (HERE.parent.parent / "include/vecmath.h").read_bytes()
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
