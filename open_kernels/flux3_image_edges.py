"""NPU contracts for FLUX 3 image/text entry and final image projection."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.flux3_weight_layout import HIDDEN, gemm_slots


def specs(resolution: int = 512, text_length: int = 80) -> dict[str, dict[str, dict]]:
    if resolution not in (512, 1024) or not 1 <= text_length <= 512:
        raise ValueError("FLUX 3 edges support 512/1024 pixels and 1..512 text tokens")
    image = (resolution // 16) ** 2
    text = (text_length + 511) // 512 * 512
    slots = gemm_slots()
    gemm = {}
    for name, rows in (("emb.image", image), ("emb.text", text),
                       ("final.image", image)):
        slot = slots[name]
        gemm[name] = {"M": rows, "K": slot.k, "N": slot.n}
    gemm["emb.image"]["layout"] = {"lda": 128,
                                     "a_size": image * 128 + 512}
    view = {"off": 0, "ld": HIDDEN, "T": image, "E": 1}
    ew = {"final_norm": {"op": "ln_mod", "a": view, "y": view,
                         "p_off": HIDDEN, "n_par": 2,
                         "idx": {"shift": 0, "scale": 1},
                         "sizes": {"X": image * HIDDEN, "B": HIDDEN,
                                   "P": 4 * HIDDEN, "Y": image * HIDDEN,
                                   "Z": HIDDEN}}}
    return {"gemm": gemm, "ew": ew}


def final_norm_params(shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if (np.shape(shift) != (HIDDEN,) or np.shape(scale) != (HIDDEN,) or
            not np.isfinite(shift).all() or not np.isfinite(scale).all()):
        raise ValueError("FLUX 3 final head modulation must be finite [3072] vectors")
    zero = np.zeros(HIDDEN, bfloat16)
    return np.concatenate((zero, np.asarray(shift, bfloat16),
                           np.asarray(scale, bfloat16), zero))
