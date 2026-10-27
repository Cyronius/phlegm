"""Pack Gemma 4 audio stride-2, pad-1 3x3 convolutions for BF16 NPU GEMM.

FLM stores these weights as ordinary BF16 [Cout,Cin,3,3]. Each invocation
processes exactly 256 output pixels. The first layer uses Cin=1, Cout=128;
the second uses Cin=128, Cout=32. In both cases the NPU output channel tile
is padded to 256 and the caller keeps only the real channels.
"""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_mm_source import npu_weight_row_major


def geometry(cin: int, cout: int) -> tuple[int, int]:
    if (cin, cout) not in ((1, 128), (128, 32)):
        raise ValueError("unsupported Gemma 4 audio subsampling layer")
    k = 64 if cin == 1 else 1152
    return k, 256


def pack_weights(weight: np.ndarray) -> np.ndarray:
    if weight.ndim != 4 or weight.shape[2:] != (3, 3):
        raise ValueError("expected [Cout,Cin,3,3] weight")
    cout, cin = weight.shape[:2]
    k, n = geometry(cin, cout)
    padded = np.zeros((n, k), dtype=bfloat16)
    padded[:cout, :cin * 9] = np.asarray(weight, dtype=bfloat16).reshape(cout, cin * 9)
    return npu_weight_row_major(padded, n, k)


def pack_window(input_hwc: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    if input_hwc.ndim != 3:
        raise ValueError("expected HWC input")
    height, width, cin = input_hwc.shape
    cout = 128 if cin == 1 else 32
    k, _ = geometry(cin, cout)
    oh, ow = (height + 1) // 2, (width + 1) // 2
    if oh * ow != 256:
        raise ValueError("one NPU call must contain 256 output pixels")
    src = np.asarray(input_hwc, dtype=bfloat16)
    padded = np.pad(src, ((1, 1), (1, 1), (0, 0)))
    rows = np.zeros((256, k), dtype=bfloat16)
    for y in range(oh):
        for x in range(ow):
            patch = padded[2 * y:2 * y + 3, 2 * x:2 * x + 3, :]
            rows[y * ow + x, :cin * 9] = patch.transpose(2, 0, 1).reshape(-1)
    return rows, (oh, ow)


def unpack(output: np.ndarray, shape: tuple[int, int], channels: int) -> np.ndarray:
    oh, ow = shape
    if oh * ow != 256 or channels not in (32, 128):
        raise ValueError("invalid Gemma 4 audio output geometry")
    return np.asarray(output).reshape(256, 256)[:, :channels].reshape(oh, ow, channels)
