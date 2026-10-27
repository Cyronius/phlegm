"""Gemma 4 vision 256-patch BF16 projected-plus-position NPU image."""
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

ROWS, HIDDEN = 256, 768


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_vision_pos_add(projected: In, positions: In,
                          output: Out, *, srchash: CompileTime[int] = 0):
    row_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    positions_ty = np.ndarray[(2 * HIDDEN,), np.dtype[bfloat16]]
    panel_ty = np.ndarray[(ROWS * HIDDEN,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma4_vision_pos_add_row", source_file=str(HERE / "add.cc"),
                          arg_types=[row_ty, positions_ty, row_ty],
                          include_dirs=include_dirs(), compile_flags=["-ffp-contract=off"])
    pf = ObjectFifo(row_ty, name="projected", depth=2)
    xf = ObjectFifo(positions_ty, name="positions", depth=2)
    of = ObjectFifo(row_ty, name="output", depth=2)

    def body(pi, xi, oo, add):
        for _ in range_(ROWS):
            p = pi.acquire(1)
            x = xi.acquire(1)
            o = oo.acquire(1)
            add(p, x, o)
            pi.release(1)
            xi.release(1)
            oo.release(1)

    worker = Worker(body, fn_args=[pf.cons(), xf.cons(), of.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(p, x, o, pp, xp, oc):
        tap = TensorAccessPattern((1, ROWS * HIDDEN), 0,
                                  [1, 1, ROWS, HIDDEN], [0, 0, HIDDEN, 1])
        xtap = TensorAccessPattern((1, ROWS * 2 * HIDDEN), 0,
                                   [1, 1, ROWS, 2 * HIDDEN], [0, 0, 2 * HIDDEN, 1])
        pipe = Pipeline(3)
        pipe.drain(oc, o, tap)
        pipe.fill(pp, p, tap)
        pipe.fill(xp, x, xtap)
        pipe.finish()

    positions_panel_ty = np.ndarray[(ROWS * 2 * HIDDEN,), np.dtype[bfloat16]]
    runtime = Runtime(sequence, [panel_ty, positions_panel_ty, panel_ty,
                                 pf.prod(), xf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_vision_pos_add
SPECIALIZE = {"srchash": int(hashlib.sha1((HERE / "add.cc").read_bytes()).hexdigest()[:8], 16)}
