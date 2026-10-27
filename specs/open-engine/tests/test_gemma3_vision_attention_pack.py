import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_attention_pack import key_weight, query_panel, value_weight


class SiglipAttentionPackTest(unittest.TestCase):
    def test_three_parts_and_padding(self):
        qkv = np.zeros((4096, 3584), bfloat16)
        qkv[256, 5 * 72 + 71] = 1
        qkv[4095, 1152 + 5 * 72 + 71] = 2
        qkv[4095, 2304 + 5 * 72 + 71] = 3
        q = query_panel(qkv, 5, 256)
        k = key_weight(qkv, 5)
        v = value_weight(qkv, 5)
        self.assertEqual(q.shape, (256, 128))
        self.assertEqual(k.shape, (4096, 128))
        self.assertEqual(v.shape, (256, 4096))
        self.assertEqual(float(q[0, 71]), 1)
        self.assertEqual(float(k[-1, 71]), 2)
        self.assertEqual(float(v[71, -1]), 3)
        self.assertTrue(np.all(q[:, 72:] == 0))
        self.assertTrue(np.all(k[:, 72:] == 0))
        self.assertTrue(np.all(v[72:] == 0))


if __name__ == "__main__":
    unittest.main()
