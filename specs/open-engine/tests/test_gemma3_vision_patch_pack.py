"""Check RGB patch and Conv2d weight ordering at tile boundaries."""
import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_patch_pack import pack_pixels, pad_weight, panel


class PatchPackTest(unittest.TestCase):
    def test_spatial_and_channel_order(self):
        image = np.zeros((3, 896, 896), bfloat16)
        image[0, 0, 0] = 1
        image[1, 0, 0] = 2
        image[2, 0, 0] = 3
        image[0, 0, 14] = 4
        image[0, 14, 0] = 5
        image[2, 895, 895] = 6
        patches = pack_pixels(image)
        self.assertEqual(patches.shape, (4096, 640))
        self.assertEqual(float(patches[0, 0]), 1)
        self.assertEqual(float(patches[0, 196]), 2)
        self.assertEqual(float(patches[0, 392]), 3)
        self.assertEqual(float(patches[1, 0]), 4)
        self.assertEqual(float(patches[64, 0]), 5)
        self.assertEqual(float(patches[-1, 587]), 6)
        self.assertTrue(np.all(patches[:, 588:] == 0))
        self.assertTrue(np.array_equal(panel(patches, 256), patches[256:512]))

    def test_weight_padding(self):
        weight = np.zeros((1152, 3, 14, 14), bfloat16)
        weight[1151, 2, 13, 13] = 7
        padded = pad_weight(weight)
        self.assertEqual(padded.shape, (1280, 640))
        self.assertEqual(float(padded[1151, 587]), 7)
        self.assertTrue(np.all(padded[1152:] == 0))
        self.assertTrue(np.all(padded[:, 588:] == 0))


if __name__ == "__main__":
    unittest.main()
