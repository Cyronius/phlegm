"""Select GPT-OSS o_proj chunks from the converter's four-band supertiles."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recipes.pack import supertile_perm  # noqa: E402

BANDS = 96       # 3072 stored rows, 2880 real output channels
K_BLOCKS = 32   # 4096 input values
CHUNK = 2560


def projection(raw: bytes | np.ndarray, start_band: int) -> np.ndarray:
    """Return uint8[16,32,2560] from the padded 3072-row o_proj tensor."""
    if start_band not in range(0, BANDS, 16):
        raise ValueError("o_proj uses six aligned 16-band calls")
    source = np.frombuffer(raw, np.uint8)
    if source.size != BANDS * K_BLOCKS * CHUNK:
        raise ValueError(f"o_proj requires {BANDS * K_BLOCKS * CHUNK} source bytes")
    source = source.reshape(-1, CHUNK)
    order = supertile_perm(BANDS, K_BLOCKS, 4)
    return np.ascontiguousarray(source[order[start_band:start_band + 16]])
