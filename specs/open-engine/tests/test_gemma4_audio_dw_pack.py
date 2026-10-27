import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_audio_dw_pack import pack, unpack


class Gemma4AudioDepthwisePackTest(unittest.TestCase):
    def test_packed_panels_reproduce_valid_convolution(self):
        rng = np.random.default_rng(20261027)
        x = rng.normal(0, 0.2, (260, 1024)).astype(np.float32).astype(bfloat16)
        w = rng.normal(0, 0.06, (5, 1024)).astype(np.float32).astype(bfloat16)
        panels, weights = pack(x, w)
        p = panels.reshape(128, 36, 64).astype(np.float32)
        wt = weights.reshape(128, 5, 64).astype(np.float32)
        y = sum(p[:, t:t + 32] * wt[:, t:t + 1] for t in range(5))
        actual = unpack(y.reshape(-1))
        expected = sum(x[t:t + 256].astype(np.float64) * w[t].astype(np.float64)
                       for t in range(5))
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
