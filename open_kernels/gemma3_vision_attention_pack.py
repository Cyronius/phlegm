"""Pack one SigLIP attention head for NPU QK, softmax, and PV GEMMs."""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

TOKENS, HEADS, HD, PADDED_HD, HIDDEN = 4096, 16, 72, 128, 1152
QKV_PADDED = 3584


def _head(qkv: np.ndarray, part: int, head: int) -> np.ndarray:
    if qkv.shape != (TOKENS, QKV_PADDED) or qkv.dtype != bfloat16:
        raise ValueError("expected BF16[4096,3584] padded QKV")
    if not 0 <= head < HEADS or part not in (0, 1, 2):
        raise ValueError("invalid QKV part or head")
    start = part * HIDDEN + head * HD
    return qkv[:, start:start + HD]


def query_panel(qkv: np.ndarray, head: int, start: int) -> np.ndarray:
    if start % 256 or not 0 <= start < TOKENS:
        raise ValueError("query panel must start on a 256-row boundary")
    out = np.zeros((256, PADDED_HD), bfloat16)
    out[:, :HD] = _head(qkv, 0, head)[start:start + 256]
    return out


def key_weight(qkv: np.ndarray, head: int) -> np.ndarray:
    out = np.zeros((TOKENS, PADDED_HD), bfloat16)
    out[:, :HD] = _head(qkv, 1, head)
    return out


def value_weight(qkv: np.ndarray, head: int) -> np.ndarray:
    out = np.zeros((256, TOKENS), bfloat16)
    out[:HD] = _head(qkv, 2, head).T
    return out
