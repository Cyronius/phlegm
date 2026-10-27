"""Gemma 4 final 30*tanh(logits/30) across its 262144-token vocabulary."""
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

VOCAB = 262144
BLOCK = 256


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_logit_cap(logits: In, capped: Out, *, srchash: CompileTime[int] = 0):
    full_ty = np.ndarray[(VOCAB,), np.dtype[np.float32]]
    block_ty = np.ndarray[(BLOCK,), np.dtype[np.float32]]
    fn = ExternalFunction("gemma4_logit_cap_block", source_file=str(HERE / "cap.cc"),
                          arg_types=[block_ty, block_ty], include_dirs=include_dirs(),
                          compile_flags=["-ffp-contract=off"])
    xf = ObjectFifo(block_ty, name="logits", depth=2)
    yf = ObjectFifo(block_ty, name="capped", depth=2)

    def core_body(xi, yo, cap):
        for _ in range_(VOCAB // BLOCK):
            x = xi.acquire(1)
            y = yo.acquire(1)
            cap(x, y)
            xi.release(1)
            yo.release(1)

    worker = Worker(core_body, fn_args=[xf.cons(), yf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(x, y, xp, yc):
        tap = TensorAccessPattern((1, VOCAB), 0,
                                  [1, 1, VOCAB // BLOCK, BLOCK], [0, 0, BLOCK, 1])
        pipe = Pipeline(3)
        pipe.drain(yc, y, tap)
        pipe.fill(xp, x, tap)
        pipe.finish()

    runtime = Runtime(sequence, [full_ty, full_ty, xf.prod(), yf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_logit_cap
_src = (HERE / "cap.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
