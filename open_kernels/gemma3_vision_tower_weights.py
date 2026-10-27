"""Pad SigLIP transformer BF16 matrices for Phlegm's pre-tiled NPU GEMM."""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

SHAPES = {
    "qkv": (3456, 1152, 3584, 1152),
    "attention_out": (1152, 1152, 1280, 1152),
    "mlp_fc1": (4304, 1152, 4352, 1152),
    "mlp_fc2": (1152, 4304, 1280, 4352),
}


def pad_matrix(weight: np.ndarray, operation: str) -> np.ndarray:
    rows, cols, padded_rows, padded_cols = SHAPES[operation]
    if weight.shape != (rows, cols) or weight.dtype != bfloat16:
        raise ValueError(f"{operation} weight must be BF16[{rows},{cols}]")
    out = np.zeros((padded_rows, padded_cols), bfloat16)
    out[:rows, :cols] = weight
    return out


def qkv_weight(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    if any(x.shape != (1152, 1152) or x.dtype != bfloat16 for x in (q, k, v)):
        raise ValueError("q/k/v must each be BF16[1152,1152]")
    return pad_matrix(np.concatenate((q, k, v), axis=0), "qkv")
