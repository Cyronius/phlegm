"""Decode FLM Gemma 4 E2B/E4B vision positions and pack one NPU panel."""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

ROWS, HIDDEN, POSITIONS = 256, 768, 10240


def source_table(source: np.ndarray) -> np.ndarray:
    """FLM BF16 [2,12,40,16384] tiles -> logical [2,10240,768]."""
    raw = np.asarray(source)
    if raw.dtype == bfloat16:
        raw = raw.view(np.uint16)
    if raw.dtype != np.uint16 or raw.shape != (2, 12, 40, 16384):
        raise ValueError("expected the tiled BF16 [2,12,40,16384] vision position table")
    return np.ascontiguousarray(raw.reshape(2, 12, 40, 64, 256)
                                 .transpose(0, 2, 4, 1, 3)
                                 .reshape(2, POSITIONS, HIDDEN)).view(bfloat16)


def pack_positions(table: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gather x/y vectors for a 256-patch panel; (-1,-1) rows are padding."""
    if table.shape != (2, POSITIONS, HIDDEN) or table.dtype != bfloat16:
        raise ValueError("expected decoded BF16 [2,10240,768] table")
    if positions.shape != (ROWS, 2) or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("expected integer [256,2] patch positions")
    padding = np.all(positions == -1, axis=1)
    valid = np.all((positions >= 0) & (positions < POSITIONS), axis=1)
    if not np.all(padding | valid):
        raise ValueError("position must be an in-range (x,y) pair or (-1,-1)")
    x = np.zeros((ROWS, HIDDEN), bfloat16)
    y = np.zeros_like(x)
    x[valid] = table[0, positions[valid, 0]]
    y[valid] = table[1, positions[valid, 1]]
    return x, y


def reference(projected: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two BF16 additions: x+y, then projected+position."""
    if any(array.shape != (ROWS, HIDDEN) for array in (projected, x, y)):
        raise ValueError("expected three [256,768] panels")
    position = (x.astype(np.float32) + y.astype(np.float32)).astype(bfloat16)
    return (projected.astype(np.float32) + position.astype(np.float32)).astype(bfloat16)
