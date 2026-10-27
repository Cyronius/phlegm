"""Pack a 64×64 SigLIP patch grid into 256 spatial 4×4 projector groups."""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16


def pack(patches: np.ndarray) -> np.ndarray:
    values = np.asarray(patches)
    if values.shape != (4096, 1152) or values.dtype != bfloat16:
        raise ValueError("Gemma 3 patches must be BF16[4096,1152]")
    return np.ascontiguousarray(values.reshape(16, 4, 16, 4, 1152)
                                .transpose(0, 2, 1, 3, 4)
                                .reshape(256, 16, 1152))
