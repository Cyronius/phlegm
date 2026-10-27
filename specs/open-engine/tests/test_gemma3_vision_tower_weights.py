import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_tower_weights import SHAPES, pad_matrix, qkv_weight


class SiglipMatrixPackTest(unittest.TestCase):
    def test_padding(self):
        for operation, (rows, cols, padded_rows, padded_cols) in SHAPES.items():
            with self.subTest(operation=operation):
                weight = np.zeros((rows, cols), bfloat16)
                weight[-1, -1] = 7
                packed = pad_matrix(weight, operation)
                self.assertEqual(packed.shape, (padded_rows, padded_cols))
                self.assertEqual(float(packed[rows - 1, cols - 1]), 7)
                self.assertTrue(np.all(packed[rows:] == 0))
                self.assertTrue(np.all(packed[:, cols:] == 0))

    def test_qkv_order(self):
        q, k, v = [np.full((1152, 1152), value, bfloat16)
                   for value in (1, 2, 3)]
        result = qkv_weight(q, k, v)
        self.assertEqual(float(result[0, 0]), 1)
        self.assertEqual(float(result[1152, 0]), 2)
        self.assertEqual(float(result[2304, 0]), 3)
        self.assertTrue(np.all(result[3456:] == 0))


if __name__ == "__main__":
    unittest.main()
