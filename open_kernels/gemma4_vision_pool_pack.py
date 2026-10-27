"""Group Gemma 4 E2B/E4B patch vectors by 3x3 spatial pool positions."""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16


def pack(hidden: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hidden.shape != (2520, 768) or positions.shape != (2520, 2):
        raise ValueError("expected 2520 patches, 768 channels, and 2D positions")
    if not np.issubdtype(positions.dtype, np.integer):
        raise TypeError("patch positions must be integers")
    valid = np.all(positions >= 0, axis=1)
    if not valid.any():
        raise ValueError("no valid image patches")
    max_x = int(positions[valid, 0].max()) + 1
    if max_x % 3:
        raise ValueError("patch grid width must be divisible by 3")
    group_width = max_x // 3
    groups = np.zeros((280, 9, 768), dtype=bfloat16)
    counts = np.zeros(280, np.int32)
    source = np.asarray(hidden, dtype=bfloat16)
    for patch in np.flatnonzero(valid):
        x, y = (int(v) for v in positions[patch])
        index = x // 3 + group_width * (y // 3)
        if index < 0 or index >= 280 or counts[index] >= 9:
            raise ValueError("invalid 3x3 patch pool assignment")
        groups[index, counts[index]] = source[patch]
        counts[index] += 1
    return groups, counts > 0
