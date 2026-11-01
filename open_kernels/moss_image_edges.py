"""Native MOSS Image latent entry and final velocity projection contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.moss_dit_block import EL, tensor_entries, vector
from open_kernels.moss_weight_layout import gemm_slots


def specs(resolution: int = 512) -> dict[str, dict[str, dict]]:
    if resolution not in (512, 1024):
        raise ValueError("native MOSS image edges require 512 or 1024 pixels")
    tokens = (resolution // 16) ** 2
    slots = gemm_slots()
    gemm = {}
    for name in ("emb.image", "final.image"):
        slot = slots[name]
        gemm[name] = {"M": tokens, "K": slot.k, "N": slot.n}
    gemm["emb.image"]["layout"] = {"lda": 128,
                                     "a_size": tokens * 128 + 512}
    view = {"off": 0, "ld": EL, "T": tokens, "E": 1}
    ew = {"bias_image": {"op": "bias", "a": view, "y": view,
                         "p_off": EL, "n_par": 1, "bias_period": 1,
                         "sizes": {"X": tokens * EL, "B": EL, "P": 3 * EL,
                                   "Y": tokens * EL, "Z": EL}}}
    return {"gemm": gemm, "bias": ew}


def entry_bias(model_dir: Path) -> np.ndarray:
    entries = tensor_entries(Path(model_dir))
    zero = np.zeros(EL, bfloat16)
    return np.concatenate((zero,
                           vector(entries, "video_embedder.bias", EL).astype(bfloat16),
                           zero))


def final_norm_params(model_dir: Path, head: np.ndarray) -> np.ndarray:
    """MOSS final RMSNorm + head shift/scale as affine RMS parameters."""
    h = np.asarray(head, np.float32)
    if h.shape != (2, EL) or not np.isfinite(h).all():
        raise ValueError("expected finite [2,3072] native head modulation")
    entries = tensor_entries(Path(model_dir))
    weight = vector(entries, "final_norm_video.weight", EL)
    zero = np.zeros(EL, bfloat16)
    return np.concatenate((zero, h[0].astype(bfloat16),
                           (weight * (1 + h[1])).astype(bfloat16), zero))
