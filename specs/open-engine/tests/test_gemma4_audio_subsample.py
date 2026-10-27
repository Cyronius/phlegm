import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_audio_subsample import pack_window, pack_weights
from open_kernels.gemma4_mm_source import npu_weight_row_major


class AudioSubsampleTest(unittest.TestCase):
    def test_im2col_and_weight_order(self):
        rng = np.random.default_rng(4)
        for cin, cout in ((1, 128), (128, 32)):
            x = rng.normal(size=(32, 32, cin)).astype(np.float32).astype(bfloat16)
            w = rng.normal(size=(cout, cin, 3, 3)).astype(np.float32).astype(bfloat16)
            a, shape = pack_window(x)
            b = pack_weights(w)
            self.assertEqual(shape, (16, 16))
            self.assertEqual(a.shape, (256, 64 if cin == 1 else 1152))
            self.assertEqual(b.size, a.shape[1] * 256)
            # Compare the im2col product with a direct padded convolution.
            dense_w = np.zeros((256, a.shape[1]), bfloat16)
            dense_w[:cout, :cin * 9] = w.reshape(cout, cin * 9)
            self.assertTrue(np.array_equal(b,
                npu_weight_row_major(dense_w, 256, a.shape[1])))
            actual = a.astype(np.float64) @ dense_w[:cout].astype(np.float64).T
            padded = np.pad(x.astype(np.float64), ((1, 1), (1, 1), (0, 0)))
            reference = np.zeros((16, 16, cout), np.float64)
            for y in range(16):
                for col in range(16):
                    p = padded[2*y:2*y+3, 2*col:2*col+3].transpose(2, 0, 1)
                    reference[y, col] = np.sum(w.astype(np.float64) * p[None], axis=(1, 2, 3))
            self.assertTrue(np.allclose(actual.reshape(16,16,cout), reference, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
