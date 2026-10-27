"""Pack scaled Gemma 4 audio Q/K/V and projected relative keys per NPU head."""
from __future__ import annotations

import numpy as np


def pack(q: np.ndarray, k: np.ndarray, v: np.ndarray,
         relative_key: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if q.shape != (8, 12, 128) or k.shape != (8, 24, 128) or v.shape != k.shape:
        raise ValueError("audio attention requires 8 heads, 12 queries, 24 context rows")
    if relative_key.shape != (8, 13, 128) or mask.shape != (8, 12, 24):
        raise ValueError("relative keys must be [8,13,128] and mask [8,12,24]")
    arrays = (np.asarray(q, np.float32), np.asarray(k, np.float32),
              np.asarray(v, np.float32), np.asarray(relative_key, np.float32))
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("nonfinite audio attention input")
    if np.any(np.all(mask != 0, axis=2)):
        raise ValueError("an audio attention query has no unmasked context")
    blocks = [b"".join((np.ascontiguousarray(a[h]).tobytes() for a in arrays)) +
              np.ascontiguousarray(mask[h], np.uint8).tobytes() for h in range(8)]
    result = np.frombuffer(b"".join(blocks), dtype=np.uint8).copy()
    if result.size != 8 * 37664:
        raise AssertionError("wrong audio attention packet size")
    return result


def reference(q: np.ndarray, k: np.ndarray, v: np.ndarray,
              relative_key: np.ndarray, mask: np.ndarray) -> np.ndarray:
    score = np.einsum("hqd,hkd->hqk", q.astype(np.float64), k.astype(np.float64))
    relative_unshifted = np.einsum("hqd,hrd->hqr", q.astype(np.float64),
                                  relative_key.astype(np.float64))
    shifted = np.zeros((8, 12, 24), np.float64)
    for row in range(12):
        shifted[:, row, row:row + 13] = relative_unshifted[:, row]
    score += shifted
    score = 50 * np.tanh(score / 50)
    score = np.where(mask != 0, -1e9, score)
    score -= score.max(axis=-1, keepdims=True)
    probability = np.exp(score)
    probability /= probability.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,hkd->hqd", probability, v.astype(np.float64)).astype(np.float32)
