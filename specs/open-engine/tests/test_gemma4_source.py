import unittest

import numpy as np

from open_kernels.gemma4_source import pair_q4_0_16row, pair_q8_16row
from open_kernels.model.q4nx import bf16_to_f32, dq_chunks_q4_1, dq_chunks_q8, f32_to_bf16


class Gemma4SourceTilesTest(unittest.TestCase):
    def test_pair_signed_q4_tiles_preserves_every_value(self):
        rng = np.random.default_rng(20261027)
        scales = f32_to_bf16(rng.choice([-1, 1], (2, 8, 16)) *
                             rng.uniform(0.001, 0.1, (2, 8, 16)))
        code = rng.integers(0, 256, (2, 256, 8), dtype=np.uint8)
        source = np.empty((2, 1, 2304), np.uint8)
        source[:, 0, :256] = scales.view(np.uint8).reshape(2, 256)
        source[:, 0, 256:] = code.reshape(2, 2048)
        pool = pair_q4_0_16row(source)
        self.assertEqual(pool.shape, (1, 1, 5120))
        actual = dq_chunks_q4_1(pool.reshape(1, 5120)).reshape(32, 256)
        expected = np.empty((32, 256), np.float32)
        s = bf16_to_f32(scales)
        for half in range(2):
            for row in range(16):
                b = code[half, :, row // 2]
                nib = b & 15 if row % 2 == 0 else b >> 4
                signed = ((nib.astype(np.int16) ^ 8) - 8).astype(np.float32)
                expected[half * 16 + row] = signed * np.repeat(s[half, :, row], 32)
        np.testing.assert_array_equal(actual, expected)

    def test_pair_q8_tiles_preserves_every_value(self):
        rng = np.random.default_rng(20261028)
        scales = f32_to_bf16(rng.uniform(0.001, 0.1, (2, 8, 16)))
        code = rng.integers(-128, 128, (2, 256, 16), dtype=np.int8)
        source = np.empty((2, 1, 4352), np.uint8)
        source[:, 0, :256] = scales.view(np.uint8).reshape(2, 256)
        source[:, 0, 256:] = code.view(np.uint8).reshape(2, 4096)
        pool = pair_q8_16row(source)
        self.assertEqual(pool.shape, (1, 1, 8704))
        actual = dq_chunks_q8(pool.reshape(1, 8704)).reshape(32, 256)
        expected = np.empty((32, 256), np.float32)
        s = bf16_to_f32(scales)
        for half in range(2):
            for row in range(16):
                expected[half * 16 + row] = code[half, :, row].astype(np.float32) * np.repeat(s[half, :, row], 32)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
