"""Gemma 3 SigLIP 4x4 patch pooling and projector RMSNorm on NPU."""
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

TOKENS, PATCHES, HIDDEN = 256, 16, 1152


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma3_vision_pool_norm(groups: In, weight: In, output: Out,
                            *, srchash: CompileTime[int] = 0):
    group_ty = np.ndarray[(PATCHES * HIDDEN,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    all_groups = np.ndarray[(TOKENS * PATCHES * HIDDEN,), np.dtype[bfloat16]]
    all_output = np.ndarray[(TOKENS * HIDDEN,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma3_vision_pool_norm_group",
                          source_file=str(HERE / "pool_norm.cc"),
                          arg_types=[group_ty, row_ty, row_ty],
                          include_dirs=include_dirs(),
                          compile_flags=["-ffp-contract=off"])
    gf = ObjectFifo(group_ty, name="groups", depth=1)
    wf = ObjectFifo(row_ty, name="weight", depth=1)
    of = ObjectFifo(row_ty, name="output", depth=1)

    def core_body(gi, wi, yo, pool_norm):
        w = wi.acquire(1)
        for _ in range_(TOKENS):
            g = gi.acquire(1)
            y = yo.acquire(1)
            pool_norm(g, w, y)
            gi.release(1)
            yo.release(1)
        wi.release(1)

    worker = Worker(core_body, fn_args=[gf.cons(), wf.cons(), of.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(g, w, o, gp, wp, oc):
        gt = TensorAccessPattern((1, TOKENS * PATCHES * HIDDEN), 0,
                                 [1, 1, TOKENS, PATCHES * HIDDEN],
                                 [0, 0, PATCHES * HIDDEN, 1])
        wt = TensorAccessPattern((1, HIDDEN), 0,
                                 [1, 1, 1, HIDDEN], [0, 0, 0, 1])
        ot = TensorAccessPattern((1, TOKENS * HIDDEN), 0,
                                 [1, 1, TOKENS, HIDDEN], [0, 0, HIDDEN, 1])
        pipe = Pipeline(3)
        pipe.drain(oc, o, ot)
        pipe.fill(gp, g, gt)
        pipe.fill(wp, w, wt)
        pipe.finish()

    runtime = Runtime(sequence, [all_groups, row_ty, all_output,
                                 gf.prod(), wf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma3_vision_pool_norm
_src = (HERE / "pool_norm.cc").read_bytes() + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
