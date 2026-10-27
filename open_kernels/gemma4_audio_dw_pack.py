"""Host panels for Gemma 4's [5,1024] BF16 audio depthwise convolution."""
from __future__ import annotations

import numpy as np


def pack(input_frames: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valid convolution input [260,1024] and weights [5,1024] to 128 panels."""
    x, w = np.asarray(input_frames), np.asarray(weight)
    if x.shape != (260, 1024) or w.shape != (5, 1024) or x.dtype != w.dtype:
        raise ValueError("expected matching BF16 [260,1024] and [5,1024]")
    if x.dtype not in (np.uint16, np.dtype("bfloat16")):
        raise TypeError("expected BF16 values or uint16 bits")
    panels = np.empty((8, 16, 36, 64), x.dtype)
    weights = np.empty((8, 16, 5, 64), w.dtype)
    for t in range(8):
        for c in range(16):
            panels[t, c] = x[t * 32:t * 32 + 36, c * 64:(c + 1) * 64]
            weights[t, c] = w[:, c * 64:(c + 1) * 64]
    return panels.reshape(-1), weights.reshape(-1)


def unpack(output: np.ndarray) -> np.ndarray:
    raw = np.asarray(output, np.float32)
    if raw.size != 256 * 1024:
        raise ValueError("expected 128 output panels of 32x64")
    return np.ascontiguousarray(raw.reshape(8, 16, 32, 64)
                                 .transpose(0, 2, 1, 3).reshape(256, 1024))
