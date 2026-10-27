"""SigLIP BF16 linear bias boundary, optionally followed by tanh GELU."""
from __future__ import annotations

import hashlib
import os
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

ROWS = 256
LOGICAL = int(os.environ.get("PHLEGM_LOGICAL", "1152"))
PADDED = int(os.environ.get("PHLEGM_PADDED", "1280"))
GELU = int(os.environ.get("PHLEGM_GELU", "0"))
if LOGICAL <= 0 or PADDED < LOGICAL or PADDED % 256 or GELU not in (0, 1):
    raise ValueError("invalid SigLIP bias/activation geometry")


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma3_vision_bias_activation(projected: In, bias: In, output: Out,
                                  *, srchash: CompileTime[int] = 0):
    f32_row = np.ndarray[(PADDED,), np.dtype[np.float32]]
    bf16_row = np.ndarray[(PADDED,), np.dtype[bfloat16]]
    fn = ExternalFunction("gemma3_vision_bias_activation_row",
                          source_file=str(HERE / "post.cc"),
                          arg_types=[f32_row, bf16_row, bf16_row],
                          include_dirs=include_dirs(),
                          compile_flags=[f"-DLOGICAL={LOGICAL}", f"-DPADDED={PADDED}",
                                         f"-DGELU={GELU}", "-ffp-contract=off"])
    pf = ObjectFifo(f32_row, name="projected", depth=1)
    bf = ObjectFifo(bf16_row, name="bias", depth=1)
    of = ObjectFifo(bf16_row, name="output", depth=1)

    def body(pi, bi, oo, post):
        b = bi.acquire(1)
        for _ in range_(ROWS):
            p = pi.acquire(1)
            o = oo.acquire(1)
            post(p, b, o)
            pi.release(1)
            oo.release(1)
        bi.release(1)

    worker = Worker(body, fn_args=[pf.cons(), bf.cons(), of.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(p, b, o, pp, bp, oc):
        pt = TensorAccessPattern((1, ROWS * PADDED), 0,
                                 [1, 1, ROWS, PADDED], [0, 0, PADDED, 1])
        bt = TensorAccessPattern((1, PADDED), 0,
                                 [1, 1, 1, PADDED], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(oc, o, pt)
        pipe.fill(pp, p, pt)
        pipe.fill(bp, b, bt)
        pipe.finish()

    runtime = Runtime(sequence, [np.ndarray[(ROWS * PADDED,), np.dtype[np.float32]],
                                 bf16_row,
                                 np.ndarray[(ROWS * PADDED,), np.dtype[bfloat16]],
                                 pf.prod(), bf.prod(), of.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma3_vision_bias_activation
_hash = (HERE / "post.cc").read_bytes() + f"{LOGICAL}:{PADDED}:{GELU}".encode()
SPECIALIZE = {"srchash": int(hashlib.sha1(_hash).hexdigest()[:8], 16)}
