import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_audio_qk_scale_pack import K_SCALE, Q_SCALE, pack, reference


class Gemma4AudioQKScalePackTest(unittest.TestCase):
    def test_head_major_layout_and_learned_scale(self):
        q = np.zeros((8, 12, 128), bfloat16)
        k = np.zeros((8, 24, 128), bfloat16)
        learned = np.zeros(128, bfloat16)
        q[3, 7, 19] = bfloat16(2)
        k[3, 11, 19] = bfloat16(3)
        packet = pack(q, k, learned)
        self.assertEqual(packet.shape, (8, 36, 128))
        self.assertEqual(float(packet[3, 7, 19]), 2)
        self.assertEqual(float(packet[3, 23, 19]), 3)
        sq, sk = reference(q, k, learned)
        self.assertAlmostEqual(float(sq[3, 7, 19]),
                               2 * Q_SCALE * float(bfloat16(np.log(2))), places=6)
        self.assertAlmostEqual(float(sk[3, 11, 19]), 3 * K_SCALE, places=6)


if __name__ == "__main__":
    unittest.main()
