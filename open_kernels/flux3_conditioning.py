"""FLUX 3 flow-map timestep, reference timestep, and mode modulation on NPU."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.flux3_weight_layout import HIDDEN, gemm_slots

EL = HIDDEN
N = 512
GEMMS = ("time.in", "time.out", "ref_time.in", "ref_time.out",
         "early.mod.image", "early.mod.txt", "single.mod.image",
         "single.mod.txt", "final.mod")


def specs() -> dict[str, dict[str, dict]]:
    slots = gemm_slots()
    gemm = {name: {"M": N, "K": slots[name].k, "N": slots[name].n}
            for name in GEMMS}
    view = {"off": 0, "ld": EL, "T": N, "E": 1}
    ew = {
        "silu": {"op": "silu", "a": view, "y": view,
                 "sizes": {"X": N * EL, "B": EL, "P": EL,
                           "Y": N * EL, "Z": EL}},
        "add": {"op": "euler", "a": view, "b": view, "y": view,
                "p_off": EL, "n_par": 1,
                "sizes": {"X": N * EL, "B": N * EL, "P": 3 * EL,
                          "Y": N * EL, "Z": EL}},
    }
    return {"gemm": gemm, "ew": ew}


def timestep_embedding(t: float) -> np.ndarray:
    """Published 256-wide sinusoidal embedding of a BF16 timestep."""
    t = float(np.asarray(t, dtype=bfloat16).astype(np.float32))
    freq = np.exp(-np.log(10000.0) * np.arange(128, dtype=np.float32) / 128)
    angle = (1000.0 * t) * freq
    return np.concatenate((np.cos(angle), np.sin(angle))).astype(bfloat16)


def input_rows(image_time: float, target_time: float) -> tuple[np.ndarray, np.ndarray]:
    current = np.zeros((N, 512), bfloat16)
    target = np.zeros((N, 512), bfloat16)
    current[0, :256] = timestep_embedding(image_time)
    current[1, :256] = timestep_embedding(0.0)
    target[0, :256] = timestep_embedding(target_time)
    target[1, :256] = target[0, :256]
    return current, target


def add_params() -> np.ndarray:
    zero = np.zeros(EL, bfloat16)
    dt = zero.copy()
    dt[:2] = np.array([1.0], np.float32).view(np.uint16).view(bfloat16)
    return np.concatenate((zero, dt, zero))
