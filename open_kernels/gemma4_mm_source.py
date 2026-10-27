"""Convert FLM Gemma 4 vision/audio BF16 64x256 weight tiles for NPU GEMM."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "npu_offload" / "gemm_rtp"))
from npue import tile_b  # noqa: E402


def _bits(source: np.ndarray) -> np.ndarray:
    raw = np.asarray(source)
    if raw.dtype == bfloat16:
        return raw.view(np.uint16)
    if raw.dtype != np.uint16:
        raise TypeError("Gemma 4 BF16 weight must contain BF16 values or uint16 bits")
    return raw


def source_matrix(source: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """FLM [N/64,K/256,16384] BF16 bits -> logical [N,K] BF16 bits."""
    if rows % 256 or cols % 256:
        raise ValueError("FLM multimodal MM pads N and K to 256")
    raw = _bits(source)
    if raw.size != rows * cols:
        raise ValueError("wrong FLM multimodal weight byte count")
    return np.ascontiguousarray(raw.reshape(rows // 64, cols // 256, 64, 256)
                                 .transpose(0, 2, 1, 3).reshape(rows, cols))


def npu_weight(source: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """FLM [N/64,K/256,16384] -> array GEMM's pre-tiled B [K,N] bits."""
    w = source_matrix(source, rows, cols)
    return tile_b(np.ascontiguousarray(w.T), 64, 32, 8, 8, order="k,n")


def npu_weight_row_major(source: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """FLM ordinary BF16 [N,K] matrix -> array GEMM's pre-tiled B bits."""
    if rows % 256 or cols % 64:
        raise ValueError("NPU GEMM needs N/256 and K/64 tiles")
    raw = _bits(source)
    if raw.size != rows * cols:
        raise ValueError("wrong row-major BF16 weight byte count")
    w = raw.reshape(rows, cols)
    return tile_b(np.ascontiguousarray(w.T), 64, 32, 8, 8, order="k,n")
