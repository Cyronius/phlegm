"""Bounded, carry-state Gemma 4 single-head decode attention on NPU2."""
from __future__ import annotations

import hashlib
import os
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


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemma4_attention(query_state: In, kv_rows: In, state_out: Out,
                     *, head_dim: CompileTime[int], rows: CompileTime[int],
                     srchash: CompileTime[int] = 0):
    assert (head_dim in (256, 512) and rows in (1, 128, 512, 1024)) or (head_dim == 64 and rows in (472, 1024))
    meta_bytes = head_dim * 2 + (2 * head_dim + 16) * 4
    meta_ty = np.ndarray[(meta_bytes,), np.dtype[np.uint8]]
    row_ty = np.ndarray[(2 * head_dim,), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(rows * 2 * head_dim,), np.dtype[bfloat16]]
    state_ty = np.ndarray[(2 * head_dim + 16,), np.dtype[np.float32]]
    inc = include_dirs()
    flags = [f"-DHEAD_DIM={head_dim}", "-ffp-contract=off"]
    init = ExternalFunction("gemma4_attention_init", source_file=str(HERE / "init.cc"),
                            arg_types=[meta_ty, state_ty], include_dirs=inc, compile_flags=flags)
    step = ExternalFunction("gemma4_attention_step", source_file=str(HERE / "step.cc"),
                            arg_types=[meta_ty, row_ty, state_ty], include_dirs=inc, compile_flags=flags)
    finish = ExternalFunction("gemma4_attention_finish", source_file=str(HERE / "finish.cc"),
                              arg_types=[state_ty], include_dirs=inc, compile_flags=flags)
    mf = ObjectFifo(meta_ty, name="query_state", depth=1)
    kvf = ObjectFifo(row_ty, name="kv_rows", depth=2)
    sof = ObjectFifo(state_ty, name="state_out", depth=1)

    def core_body(mi, ki, so, init_fn, step_fn, finish_fn):
        meta = mi.acquire(1)
        next_state = so.acquire(1)
        init_fn(meta, next_state)
        for _ in range_(rows):
            kv = ki.acquire(1)
            step_fn(meta, kv, next_state)
            ki.release(1)
        finish_fn(next_state)
        so.release(1)
        mi.release(1)

    worker = Worker(core_body,
                    fn_args=[mf.cons(), kvf.cons(), sof.prod(), init, step, finish],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(meta, kv, result, mp, kvp, rc):
        mtap = TensorAccessPattern((1, meta_bytes), 0,
                                   [1, 1, 1, meta_bytes], [0, 0, 0, 1])
        ktap = TensorAccessPattern((1, rows * 2 * head_dim), 0,
                                   [1, 1, rows, 2 * head_dim], [0, 0, 2 * head_dim, 1])
        stap = TensorAccessPattern((1, 2 * head_dim + 16), 0,
                                   [1, 1, 1, 2 * head_dim + 16], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(rc, result, stap)
        pipe.fill(mp, meta, mtap)
        pipe.fill(kvp, kv, ktap)
        pipe.finish()

    runtime = Runtime(sequence, [meta_ty, kv_ty, state_ty,
                                 mf.prod(), kvf.prod(), sof.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


DESIGN = gemma4_attention
_src = b"".join((HERE / name).read_bytes() for name in ("init.cc", "step.cc", "finish.cc")) + \
       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {
    "head_dim": int(os.environ.get("PHLEGM_HEAD_DIM", "256")),
    "rows": int(os.environ.get("PHLEGM_ATTN_ROWS", "128")),
    "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16),
}
