"""Native MOSS Image timestep embedding and six-way/head modulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.moss_dit_block import EL, tensor_entries, vector
from open_kernels.moss_weight_layout import gemm_slots


ROWS = 512  # one real timestep; dit_gemm pads M to 512 rows


@dataclass
class ConditioningPlan:
    buffers: dict[str, int]
    streams: dict[str, dict[str, dict]]
    ops: list[dict]


def plan() -> ConditioningPlan:
    def view(ld=EL, off=0, e=1):
        return {"off": off, "ld": ld, "T": ROWS, "E": e}

    def bias(e, ld, off=0):
        return {"op": "bias", "a": view(ld, off, e), "y": view(ld, off, e),
                "p_off": EL, "n_par": e, "bias_period": e,
                "sizes": {"X": ROWS * ld, "B": EL, "P": (e + 2) * EL,
                          "Y": ROWS * ld, "Z": EL}}

    slots = gemm_slots()
    gemm = {}
    for name in ("time.in", "time.out", "shared.mod", "head.mod"):
        slot = slots[name]
        gemm[name] = {"M": ROWS, "K": slot.k, "N": slot.n}
    gemm["time.in"]["layout"] = {"lda": 256,
                                   "a_size": ROWS * 256 + 512}
    ew = {"bias1": bias(1, EL),
          "bias_mod0": bias(3, 6 * EL),
          "bias_mod1": bias(3, 6 * EL, 3 * EL),
          "bias_head": bias(2, 2 * EL),
          "silu": {"op": "silu", "a": view(), "y": view(),
                   "sizes": {"X": ROWS * EL, "B": EL, "P": EL,
                             "Y": ROWS * EL, "Z": EL}}}
    buffers = {"FEATURE": ROWS * 256 + 512, "TIME1": ROWS * EL,
               "TIME1S": ROWS * EL, "TIME": ROWS * EL,
               "MOD": ROWS * 6 * EL, "HEAD": ROWS * 2 * EL,
               "DUMMY": EL}
    ops = [
        {"set": "gemm", "stream": "time.in", "args": ["FEATURE", "time.in", "TIME1"]},
        {"set": "ew", "stream": "bias1", "args": ["TIME1", "DUMMY", "B_TIME_IN", "TIME1", "DUMMY"]},
        {"set": "ew", "stream": "silu", "args": ["TIME1", "DUMMY", "DUMMY", "TIME1S", "DUMMY"]},
        {"set": "gemm", "stream": "time.out", "args": ["TIME1S", "time.out", "TIME"]},
        {"set": "ew", "stream": "bias1", "args": ["TIME", "DUMMY", "B_TIME_OUT", "TIME", "DUMMY"]},
        {"set": "gemm", "stream": "shared.mod", "args": ["TIME", "shared.mod", "MOD"]},
        {"set": "ew", "stream": "bias_mod0", "args": ["MOD", "DUMMY", "B_MOD0", "MOD", "DUMMY"]},
        {"set": "ew", "stream": "bias_mod1", "args": ["MOD", "DUMMY", "B_MOD1", "MOD", "DUMMY"]},
        {"set": "gemm", "stream": "head.mod", "args": ["TIME", "head.mod", "HEAD"]},
        {"set": "ew", "stream": "bias_head", "args": ["HEAD", "DUMMY", "B_HEAD", "HEAD", "DUMMY"]},
    ]
    return ConditioningPlan(buffers, {"gemm": gemm, "ew": ew}, ops)


def params(model_dir: Path) -> dict[str, np.ndarray]:
    entries = tensor_entries(Path(model_dir))
    zero = np.zeros(EL, bfloat16)

    def block(*rows):
        return np.concatenate((zero, *(np.asarray(row, dtype=bfloat16)
                                       for row in rows), zero))

    def v(name, size):
        return vector(entries, name, size)

    mod = v("shared_block_modulation.bias", 6 * EL).reshape(6, EL)
    head = v("video_head_modulation.bias", 2 * EL).reshape(2, EL)
    return {"B_TIME_IN": block(v("time_embedding.0.bias", EL)),
            "B_TIME_OUT": block(v("time_embedding.2.bias", EL)),
            "B_MOD0": block(*mod[:3]), "B_MOD1": block(*mod[3:]),
            "B_HEAD": block(*head)}
