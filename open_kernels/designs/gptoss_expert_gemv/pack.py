"""Extract one GPT-OSS expert projection into the NPU kernel's chunk order."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recipes.pack import EXPERT_ROLES, expert_chunks  # noqa: E402

CHUNK = 2560
NSLAB = NCOL128 = 23
RG = 4
ONE_EXPERT_CHUNKS = 3 * NSLAB * NCOL128 * RG


def projection(expert_bytes: bytes | np.ndarray, role: str,
               bands: int = 90) -> np.ndarray:
    """Return uint8[band, 23, 2560], with gate/up/down slabs correctly ordered."""
    if role not in EXPERT_ROLES or bands not in (1, 90):
        raise ValueError("role must be gate/up/down and bands must be 1 or 90")
    raw = np.frombuffer(expert_bytes, np.uint8)
    if raw.size != ONE_EXPERT_CHUNKS * CHUNK:
        raise ValueError(f"one expert requires {ONE_EXPERT_CHUNKS * CHUNK} bytes")
    source = raw.reshape(ONE_EXPERT_CHUNKS, CHUNK)
    pick = expert_chunks(NSLAB, NCOL128, RG)[EXPERT_ROLES.index(role), :bands]
    return np.ascontiguousarray(source[pick])
