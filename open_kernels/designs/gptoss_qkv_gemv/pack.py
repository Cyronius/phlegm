"""Select GPT-OSS q/k/v half-width chunks into 32-row NPU band order."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recipes.pack import supertile_perm  # noqa: E402

ROLE_ROWS = {"q": 4096, "k": 512, "v": 512}
K_BLOCKS = 23
CHUNK = 2560


def projection(raw: bytes | np.ndarray, role: str, start_band: int, bands: int) -> np.ndarray:
    """Return uint8[bands,23,2560] from one q/k/v container tensor.

    The source has 2944 columns and four-row-block supertiles. A caller may
    choose any contiguous 32-row band range; the 16-band image uses eight Q
    calls and one each for K and V.
    """
    if role not in ROLE_ROWS or bands not in (1, 16):
        raise ValueError("role must be q/k/v and bands must be 1 or 16")
    total = ROLE_ROWS[role] // 32
    if start_band < 0 or start_band + bands > total:
        raise ValueError(f"band range exceeds {role}'s {total} bands")
    source = np.frombuffer(raw, np.uint8)
    if source.size != total * K_BLOCKS * CHUNK:
        raise ValueError(f"{role} requires {total * K_BLOCKS * CHUNK} source bytes")
    source = source.reshape(-1, CHUNK)
    order = supertile_perm(total, K_BLOCKS, 4)
    selected = order[start_band:start_band + bands]
    return np.ascontiguousarray(source[selected])
