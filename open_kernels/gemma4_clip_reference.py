"""Reference for Gemma 4 clippable vision/audio linear boundaries."""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16


def limits(low: float, high: float) -> np.ndarray:
    if np.isnan(low) or np.isnan(high) or low > high:
        raise ValueError("linear clipping needs ordered non-NaN bounds")
    return np.array((low, high), dtype=np.float32)


def clip(values: np.ndarray, bounds: np.ndarray, *, post_linear: bool) -> np.ndarray:
    if bounds.shape != (2,) or bounds.dtype != np.float32 or bounds[0] > bounds[1]:
        raise ValueError("expected ordered FP32 [min,max] bounds")
    if values.ndim != 2 or values.shape[0] != 256 or values.shape[1] % 256:
        raise ValueError("expected 256 rows and a width divisible by 256")
    if post_linear:
        if values.dtype != np.float32:
            raise TypeError("post-linear input must be FP32 NPU GEMM output")
        rounded = values.astype(bfloat16).astype(np.float32)
    else:
        if values.dtype != bfloat16:
            raise TypeError("pre-linear input must be BF16")
        rounded = values.astype(np.float32)
    return np.clip(rounded, bounds[0], bounds[1]).astype(bfloat16)
