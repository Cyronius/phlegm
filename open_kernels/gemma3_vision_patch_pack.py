"""Lower Gemma 3 SigLIP 14×14 RGB patches into the NPU GEMM panel layout."""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

IMAGE = 896
PATCH = 14
GRID = IMAGE // PATCH
K = 3 * PATCH * PATCH
PADDED_K = 640
HIDDEN = 1152
PADDED_N = 1280


def pack_pixels(chw: np.ndarray) -> np.ndarray:
    if chw.shape != (3, IMAGE, IMAGE) or chw.dtype != bfloat16:
        raise ValueError("expected BF16 RGB CHW image [3,896,896]")
    patches = chw.reshape(3, GRID, PATCH, GRID, PATCH).transpose(1, 3, 0, 2, 4)
    out = np.zeros((GRID * GRID, PADDED_K), bfloat16)
    out[:, :K] = patches.reshape(GRID * GRID, K)
    return out


def pad_weight(weight: np.ndarray) -> np.ndarray:
    if weight.shape != (HIDDEN, 3, PATCH, PATCH) or weight.dtype != bfloat16:
        raise ValueError("expected BF16 SigLIP Conv2d weight [1152,3,14,14]")
    out = np.zeros((PADDED_N, PADDED_K), bfloat16)
    out[:HIDDEN, :K] = weight.reshape(HIDDEN, K)
    return out


def panel(patches: np.ndarray, start: int) -> np.ndarray:
    if patches.shape != (GRID * GRID, PADDED_K) or start % 256 or not 0 <= start < GRID * GRID:
        raise ValueError("expected a 256-row aligned panel from all 4096 patches")
    return np.ascontiguousarray(patches[start:start + 256])
