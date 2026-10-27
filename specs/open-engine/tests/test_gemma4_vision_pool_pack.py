import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_vision_pool_pack import pack


class VisionPoolPackTest(unittest.TestCase):
    def test_spatial_groups_and_masked_patches(self):
        positions = np.array([(x, y) for y in range(60) for x in range(42)], np.int32)
        hidden = np.zeros((2520, 768), bfloat16)
        hidden[:, 0] = np.arange(2520, dtype=np.float32).astype(bfloat16)
        groups, mask = pack(hidden, positions)
        self.assertEqual(groups.shape, (280, 9, 768))
        self.assertTrue(mask.all())
        expected = [0, 1, 2, 42, 43, 44, 84, 85, 86]
        np.testing.assert_array_equal(groups[0, :, 0].astype(np.float32),
                                      hidden[expected, 0].astype(np.float32))
        last_group = ((positions[:, 0] >= 39) & (positions[:, 1] >= 57))
        self.assertEqual(int(last_group.sum()), 9)
        positions[last_group] = -1
        groups, mask = pack(hidden, positions)
        self.assertFalse(mask[-1])
        self.assertTrue(np.all(groups[-1] == 0))


if __name__ == "__main__":
    unittest.main()
