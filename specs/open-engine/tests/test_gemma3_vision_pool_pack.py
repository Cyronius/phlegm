"""Spatial grouping for the Gemma 3 vision projector."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from open_kernels.gemma3_vision_pool_pack import pack  # noqa: E402


def test_4_by_4_spatial_groups_preserve_patch_coordinates():
    patches = np.zeros((4096, 1152), bfloat16)
    patches[:, 0] = np.arange(4096, dtype=np.float32).astype(bfloat16)
    packed = pack(patches)
    assert packed.shape == (256, 16, 1152)
    for tile_y, tile_x in ((0, 0), (0, 1), (7, 9), (15, 15)):
        token = tile_y * 16 + tile_x
        indices = [(tile_y * 4 + y) * 64 + tile_x * 4 + x
                   for y in range(4) for x in range(4)]
        np.testing.assert_array_equal(packed[token, :, 0], patches[indices, 0])
    with pytest.raises(ValueError):
        pack(patches.astype(np.float32))
