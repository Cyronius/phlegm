import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_vision_position_pack import HIDDEN, POSITIONS, ROWS, pack_positions, reference, source_table


class VisionPositionPackTest(unittest.TestCase):
    def test_source_tile_layout_and_padding(self):
        tiled = np.zeros((2, 12, 40, 16384), np.uint16)
        for axis, position, channel, bits in ((0, 257, 65, 0x3f80), (1, 10239, 767, 0x4000)):
            tiled[axis, channel // 64, position // 256,
                  (channel % 64) * 256 + position % 256] = bits
        table = source_table(tiled)
        self.assertEqual(table.shape, (2, POSITIONS, HIDDEN))
        self.assertEqual(int(table[0, 257, 65].view(np.uint16)), 0x3f80)
        self.assertEqual(int(table[1, 10239, 767].view(np.uint16)), 0x4000)
        positions = np.full((ROWS, 2), -1, np.int32)
        positions[0] = (257, 10239)
        x, y = pack_positions(table, positions)
        self.assertEqual(float(x[0, 65]), 1.0)
        self.assertEqual(float(y[0, 767]), 2.0)
        self.assertTrue(np.all(x[1:] == 0))
        self.assertTrue(np.all(y[1:] == 0))

    def test_two_bf16_additions(self):
        projected = np.zeros((ROWS, HIDDEN), bfloat16)
        x = np.zeros_like(projected)
        y = np.zeros_like(projected)
        projected[0, 0] = bfloat16(1)
        x[0, 0] = bfloat16(1 / 256)
        y[0, 0] = bfloat16(1 / 256)
        self.assertEqual(float(reference(projected, x, y)[0, 0]), float(bfloat16(1 + 1 / 128)))


if __name__ == "__main__":
    unittest.main()
