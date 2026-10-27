"""Gemma 4 E2B/E4B per-layer input GELU gate and elementwise product."""
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

WIDTH = 256


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_ple_gate(projection: In, layer_input: In, gated: Out,
                    *, srchash: CompileTime[int] = 0):
    row_ty = np.ndarray[(WIDTH,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma4_ple_gate", source_file=str(HERE / "gate.cc"),
                          arg_types=[row_ty, row_ty, row_ty], include_dirs=include_dirs(),
                          compile_flags=["-ffp-contract=off"])
    pf = ObjectFifo(row_ty, name="projection", depth=1)
    lf = ObjectFifo(row_ty, name="layer_input", depth=1)
    gf = ObjectFifo(row_ty, name="gated", depth=1)

    def core_body(pi, li, go, gate):
        p = pi.acquire(1)
        l = li.acquire(1)
        g = go.acquire(1)
        gate(p, l, g)
        go.release(1)
        li.release(1)
        pi.release(1)

    worker = Worker(core_body, fn_args=[pf.cons(), lf.cons(), gf.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(p, l, g, pp, lp, gc):
        tap = TensorAccessPattern((1, WIDTH), 0, [1, 1, 1, WIDTH], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(gc, g, tap)
        pipe.fill(pp, p, tap)
        pipe.fill(lp, l, tap)
        pipe.finish()

    runtime = Runtime(sequence, [row_ty, row_ty, row_ty,
                                 pf.prod(), lf.prod(), gf.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_ple_gate
_src = (HERE / "gate.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
