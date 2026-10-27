"""Check that role extraction follows the pinned fused expert container order."""
from __future__ import annotations

import numpy as np

from pack import CHUNK, ONE_EXPERT_CHUNKS, projection


def test_role_starts_and_complete_selection():
    raw = np.zeros((ONE_EXPERT_CHUNKS, CHUNK), np.uint8)
    raw[:, :4] = np.arange(ONE_EXPERT_CHUNKS, dtype="<u4").view(np.uint8).reshape(-1, 4)
    for role, first in (("gate", 0), ("up", 92), ("down", 4232)):
        packed = projection(raw, role)
        assert packed.shape == (90, 23, CHUNK)
        assert int(np.frombuffer(packed[0, 0, :4], dtype="<u4")[0]) == first
        assert len(np.unique(packed[:, :, :4].copy().view("<u4"))) == 90 * 23


def test_bad_source_or_role_is_refused():
    import pytest
    with pytest.raises(ValueError, match="requires"):
        projection(b"bad", "gate")
    with pytest.raises(ValueError, match="role"):
        projection(np.zeros(ONE_EXPERT_CHUNKS * CHUNK, np.uint8), "unknown")
