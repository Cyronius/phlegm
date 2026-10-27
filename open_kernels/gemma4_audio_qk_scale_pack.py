"""Pack Gemma 4 audio attention Q/K projections and learned per-dimension scale."""
from __future__ import annotations

import math

import numpy as np
from ml_dtypes import bfloat16

HEADS, QUERIES, CONTEXT, DIM = 8, 12, 24, 128
Q_SCALE = DIM ** -0.5 / math.log(2)
K_SCALE = math.log(1 + math.e) / math.log(2)


def pack(q: np.ndarray, k: np.ndarray, per_dim_scale: np.ndarray) -> np.ndarray:
    if (q.shape != (HEADS, QUERIES, DIM) or
            k.shape != (HEADS, CONTEXT, DIM) or
            per_dim_scale.shape != (DIM,)):
        raise ValueError("expected Q [8,12,128], K [8,24,128], scale [128]")
    if any(array.dtype != bfloat16 for array in (q, k, per_dim_scale)):
        raise TypeError("audio Q/K and scale must be BF16")
    return np.concatenate((q, k), axis=1)


def reference(q: np.ndarray, k: np.ndarray,
              per_dim_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pack(q, k, per_dim_scale)
    softplus = np.logaddexp(0, per_dim_scale.astype(np.float32)).astype(bfloat16)
    scaled_q = ((q.astype(np.float32) * np.float32(Q_SCALE)) *
                softplus.astype(np.float32)).astype(np.float32)
    scaled_k = (k.astype(np.float32) * np.float32(K_SCALE)).astype(np.float32)
    return scaled_q, scaled_k
