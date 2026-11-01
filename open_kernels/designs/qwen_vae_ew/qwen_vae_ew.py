"""Qwen Image 2.1 VAE pixelwise RMSNorm/SiLU and residual add on NHWC grids."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out,
                      Program, Runtime, TaskGroup, Worker, WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

HERE = Path(__file__).resolve().parent
N_COLS, PER_COL, CHUNK = 8, 2, 256


def check_spec(s: dict) -> str | None:
    if s.get("op") not in ("rms", "rms_silu", "add"):
        return "unsupported Qwen VAE elementwise operation"
    c, h, w, el = (s.get(k, 0) for k in ("C", "H", "W", "el"))
    if c <= 0 or c % 32 or h < 32 or h % 8 or w != h:
        return "invalid Qwen VAE channels or square grid"
    if el < c or el % c or el % CHUNK or el > 4096:
        return "element must hold complete pixels and 256-value DMA chunks"
    if (h * w * c) % (N_COLS * PER_COL * el):
        return "grid does not split evenly over 16 cores"
    for key in ("a", "y") + (("b",) if s["op"] == "add" else ()):
        view = s.get(key, {})
        if view.get("pitch", 0) < w + 2 or view.get("px_stride", c) != c:
            return f"{key} requires a one-pixel border and physical pixel stride C"
    if not all(s.get("sizes", {}).get(k, 0) > 0 for k in ("X", "B", "P", "Y")):
        return "all four buffer sizes must be positive"
    return None


def _base(v: dict, c: int, row: int) -> int:
    border = v.get("border", 1)
    return v.get("off", 0) + ((border + row) * v["pitch"] + border) * c


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def qwen_vae_ew(X: In, B: In, P: In, Y: Out, *, spec: CompileTime[str]):
    s = json.loads(spec)
    why = check_spec(s)
    assert why is None, why
    op, c, h, w, el = s["op"], s["C"], s["H"], s["W"], s["el"]
    l1_ty = np.ndarray[(el,), np.dtype[bfloat16]]
    l2_ty = np.ndarray[(PER_COL * el,), np.dtype[bfloat16]]
    rtp_ty = np.ndarray[(1,), np.dtype[np.int32]]
    source = str(HERE / "qwen_vae_ew.cc")
    flags = [f"-DQVE_C={c}", f"-DQVE_EL={el}",
             f"-DQVE_SILU={int(op == 'rms_silu')}"]
    rms = ExternalFunction("qve_rms", source_file=source,
                           arg_types=[l1_ty, l1_ty, l1_ty], compile_flags=flags)
    add = ExternalFunction("qve_add", source_file=source,
                           arg_types=[l1_ty, l1_ty, l1_ty], compile_flags=flags)
    kernels = [rms, add]
    offsets = [i * el for i in range(PER_COL)]
    fifos, ends = {}, {}
    for name, direction in (("A", "in"), ("B", "in"), ("Y", "out")):
        fifos[name], ends[name] = [], {}
        for col in range(N_COLS):
            fifo = ObjectFifo(l2_ty, name=f"{name}_L2_{col}", depth=2)
            fifos[name].append(fifo)
            if direction == "in":
                parts = fifo.cons().split(offsets, tile=Tile(col, 1),
                                          obj_types=[l1_ty] * PER_COL,
                                          names=[f"{name}_L1_{col}_{r}" for r in range(PER_COL)],
                                          depths=[2] * PER_COL)
                for r in range(PER_COL):
                    ends[name][col, r] = parts[r].cons()
            else:
                parts = fifo.prod().join(offsets, tile=Tile(col, 1),
                                         obj_types=[l1_ty] * PER_COL,
                                         names=[f"{name}_L1_{col}_{r}" for r in range(PER_COL)],
                                         depths=[2] * PER_COL)
                for r in range(PER_COL):
                    ends[name][col, r] = parts[r].prod()

    workers, barriers, rtps = [], {}, {}
    per_core = h * w * c // (N_COLS * PER_COL * el)

    def core_fn(a_in, b_in, y_out, kern, rtp, barrier):
        barrier.wait_for_value(1)
        if op != "add":
            par = a_in.acquire(1)
        for _ in range_(rtp[0]):
            y = y_out.acquire(1)
            if op == "add":
                a = a_in.acquire(1)
                b = b_in.acquire(1)
                kern[1](a, b, y)
                b_in.release(1)
                a_in.release(1)
            else:
                a = b_in.acquire(1)
                kern[0](a, par, y)
                b_in.release(1)
            y_out.release(1)
        if op != "add":
            a_in.release(1)
        barrier.release_with_value(1)

    for col in range(N_COLS):
        for r in range(PER_COL):
            rtps[col, r] = Buffer(rtp_ty, name=f"rtp_{col}_{r}",
                                  initial_value=np.zeros(1, np.int32), use_write_rtp=True)
            barriers[col, r] = WorkerRuntimeBarrier()
            workers.append(Worker(core_fn, [ends[name][col, r] for name in ("A", "B", "Y")]
                                  + [kernels, rtps[col, r], barriers[col, r]],
                                  tile=Tile(col, 2 + r), stack_size=0x800))

    types = [np.ndarray[(s["sizes"][name],), np.dtype[bfloat16]]
             for name in ("X", "B", "P", "Y")]

    def sequence(x, b, p, y, ap, bp, yp):
        for rtp in rtps.values():
            rtp[0] = per_core
        for barrier in barriers.values():
            barrier.set(1)
        tg = TaskGroup()
        for col in range(N_COLS):
            row0, rows = col * h // N_COLS, h // N_COLS
            if op != "add":
                ap[col].fill(p, offset=0, sizes=[PER_COL * el // CHUNK, CHUNK],
                             strides=[CHUNK, 1], group=tg)
            inputs = ([(x, ap[col], s["a"]), (b, bp[col], s["b"])]
                      if op == "add" else [(x, bp[col], s["a"])])
            for data, producer, view in inputs:
                producer.fill(data, offset=_base(view, c, row0),
                              sizes=[rows, w * c // CHUNK, CHUNK],
                              strides=[view["pitch"] * c, CHUNK, 1], group=tg)
            view = s["y"]
            yp[col].drain(y, offset=_base(view, c, row0),
                          sizes=[rows, w * c // CHUNK, CHUNK],
                          strides=[view["pitch"] * c, CHUNK, 1], wait=True, group=tg)
        tg.finish()

    runtime = Runtime(sequence, types + [
        [f.prod(tile=Tile(i, 0)) for i, f in enumerate(fifos[name])]
        if name != "Y" else [f.cons(tile=Tile(i, 0)) for i, f in enumerate(fifos[name])]
        for name in ("A", "B", "Y")])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = qwen_vae_ew
SPECIALIZE = dict(spec=os.environ.get("QVE_SPEC", json.dumps({
    "op": "rms_silu", "C": 1152, "H": 32, "W": 32, "el": 2304,
    "a": {"pitch": 34}, "y": {"pitch": 34},
    "sizes": {"X": 34 * 34 * 1152, "B": 1, "P": 2 * 2304, "Y": 34 * 34 * 1152}})))
