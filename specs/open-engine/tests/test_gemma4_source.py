import unittest

import numpy as np

from open_kernels.gemma4_source import (pair_q4_0_16row, pair_q8_16row,
                                        q4_projection_pool, q8_head_pool)
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

    def test_q4_pool_orders_k_tiles_before_row_halves(self):
        source = np.zeros((4, 5120), np.uint8)
        source[:, 1024] = np.arange(4)
        pool = q4_projection_pool(source, 64, 512, source_tile_bytes=5120)
        np.testing.assert_array_equal(pool.reshape(4, 5120)[:, 1024], [0, 2, 1, 3])

        signed = np.zeros((8, 2304), np.uint8)
        signed[:, 256] = np.arange(8)
        pool12 = q4_projection_pool(signed, 64, 512, source_tile_bytes=2304)
        np.testing.assert_array_equal(pool12.reshape(4, 5120)[:, 1024],
                                      np.array([0, 4, 1, 5], np.uint8) ^ 0x88)

    def test_q8_head_pool_orders_k_tiles_before_row_quarters(self):
        source = np.zeros((16, 4352), np.uint8)
        source[:, 256] = np.arange(16)
        pool = q8_head_pool(source, 128, 512).reshape(8, 8704)
        np.testing.assert_array_equal(pool[:, 512], [0, 4, 8, 12, 1, 5, 9, 13])
        np.testing.assert_array_equal(pool[:, 512 + 4096], [2, 6, 10, 14, 3, 7, 11, 15])


if __name__ == "__main__":
    unittest.main()
