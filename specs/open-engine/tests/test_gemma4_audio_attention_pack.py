import unittest

import numpy as np

from open_kernels.gemma4_audio_attention_pack import pack, reference


class AudioAttentionPackTest(unittest.TestCase):
    def test_packet_and_relative_shift(self):
        q = np.zeros((8, 12, 128), np.float32)
        k = np.zeros((8, 24, 128), np.float32)
        v = np.zeros_like(k)
        rel = np.zeros((8, 13, 128), np.float32)
        mask = np.zeros((8, 12, 24), np.uint8)
        q[..., 0] = 1
        rel[:, :, 0] = np.arange(13, dtype=np.float32)
        v[:, :, 0] = np.arange(24, dtype=np.float32)
        data = pack(q, k, v, rel, mask).reshape(8, 37664)
        self.assertTrue(np.array_equal(data[0, 37376:], mask[0].ravel()))
        self.assertTrue(np.array_equal(data[0, 30720:37376].view(np.float32).reshape(13, 128), rel[0]))
        # Mirror the model's pad -> flatten -> truncate -> reshape relative shift.
        unshifted = np.broadcast_to(np.arange(13, dtype=np.float32), (12, 13))
        shifted = np.pad(unshifted, ((0, 0), (0, 12))).reshape(-1)[:12 * 24].reshape(12, 24)
        direct = np.zeros((12, 24), np.float32)
        for row in range(12):
            direct[row, row:row + 13] = np.arange(13, dtype=np.float32)
        self.assertTrue(np.array_equal(shifted, direct))
        result = reference(q, k, v, rel, mask)
        self.assertEqual(result.shape, (8, 12, 128))
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
