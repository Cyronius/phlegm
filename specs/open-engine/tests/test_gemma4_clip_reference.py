import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_clip_reference import clip, limits


class Gemma4ClipReferenceTest(unittest.TestCase):
    def test_pre_and_post_linear_boundaries(self):
        source = np.zeros((256, 768), np.float32)
        source[0, :4] = (-2, -0.7, 0.7, 2)
        bounds = limits(-0.5, 0.5)
        before = clip(source.astype(bfloat16), bounds, post_linear=False)
        after = clip(source, bounds, post_linear=True)
        np.testing.assert_array_equal(before[0, :4].astype(np.float32),
                                      [-0.5, -0.5, 0.5, 0.5])
        np.testing.assert_array_equal(after, before)
        with self.assertRaises(ValueError):
            limits(1, -1)


if __name__ == "__main__":
    unittest.main()
