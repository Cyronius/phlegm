"""BF16 matrix product for Gemma 4 vision and audio projection panels.

A is [M,K] BF16; B is a pre-tiled [K,N] BF16 weight panel; C is [M,N]
FP32. The source package's 64x256 BF16 tiles are converted by
``open_kernels.gemma4_mm_source`` before this kernel consumes them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "npu_offload" / "gemm_rtp"))
from gemm_pretiled import pretiled_array  # noqa: E402

M = int(os.environ.get("PHLEGM_M", "256"))
K = int(os.environ.get("PHLEGM_K", "768"))
N = int(os.environ.get("PHLEGM_N", "768"))
if M % 256 or K % 64 or N % 256:
    raise ValueError("Gemma 4 multimodal GEMM needs M/256, K/64, N/256 tiles")

DESIGN = pretiled_array
SPECIALIZE = dict(M=M, K=K, N=N, m=64, k=64, n=32, n_aie_cols=8,
                  dtype_in_str="bf16", dtype_out_str="f32", rtp=True,
                  tg_depth=2, emulate_bf16_mmul_with_bfp16=False)
