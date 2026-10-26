"""GPT-OSS 20B router: BF16 2880-to-32 projection, bias and top-four softmax.

Output bytes: ids int32[4] @0, weights fp32[4] @16, logits fp32[32] @32.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

HIDDEN = 2880
EXPERTS = 32
ROWS = 8


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gptoss_router(x_bias: In, weights: In, result: Out,
                  *, srchash: CompileTime[int] = 0):
    x_ty = np.ndarray[(HIDDEN + EXPERTS,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(HIDDEN * EXPERTS,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(256,), np.dtype[np.uint8]]
    tile_ty = np.ndarray[(ROWS * EXPERTS,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(EXPERTS,), np.dtype[np.float32]]
    inc = include_dirs()
    project = ExternalFunction("gptoss_router_acc", source_file=str(HERE / "acc.cc"),
                               arg_types=[tile_ty, x_ty, acc_ty, np.int32], include_dirs=inc)
    finish = ExternalFunction("gptoss_router_fin", source_file=str(HERE / "fin.cc"),
                              arg_types=[acc_ty, x_ty, out_ty], include_dirs=inc)
    xf = ObjectFifo(x_ty, name="x", depth=1)
    wf = ObjectFifo(tile_ty, name="weights", depth=2)
    rf = ObjectFifo(out_ty, name="result", depth=1)

    def core_body(xi, wi, ro, acc, fa, ff):
        xv = xi.acquire(1)
        for rb in range_(HIDDEN // ROWS):
            wv = wi.acquire(1)
            fa(wv, xv, acc, rb)
            wi.release(1)
        rv = ro.acquire(1)
        ff(acc, xv, rv)
        ro.release(1)
        xi.release(1)

    worker = Worker(core_body,
                    fn_args=[xf.cons(), wf.cons(), rf.prod(),
                             Buffer(acc_ty, name="acc"), project, finish],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(x_arg, w_arg, r_arg, xp, wp, rc):
        def tap(n):
            return TensorAccessPattern((1, n), 0, [1, 1, 1, n], [0, 0, 0, 1])
        weights_tap = TensorAccessPattern((1, HIDDEN * EXPERTS), 0,
                                          [1, 1, HIDDEN // ROWS, ROWS * EXPERTS],
                                          [0, 0, ROWS * EXPERTS, 1])
        pipe = Pipeline(3)
        pipe.drain(rc, r_arg, tap(256))
        pipe.fill(xp, x_arg, tap(HIDDEN + EXPERTS))
        pipe.fill(wp, w_arg, weights_tap)
        pipe.finish()

    runtime = Runtime(sequence, [x_ty, w_ty, out_ty,
                                 xf.prod(), wf.prod(), rf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gptoss_router
_src = b"".join(p.read_bytes() for p in sorted(HERE.glob("*.cc"))) + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
