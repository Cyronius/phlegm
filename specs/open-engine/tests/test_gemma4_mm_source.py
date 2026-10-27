import unittest

import numpy as np

from open_kernels.gemma4_mm_source import npu_weight, npu_weight_row_major, source_matrix
from open_kernels.build_gemma4_multimodal_mm import image_plan
from npue import untile_b


class Gemma4MultimodalWeightTest(unittest.TestCase):
    def test_flm_64_by_256_tiles_become_the_npu_b_operand(self):
        rows, cols = 768, 3072
        matrix = (np.arange(rows * cols, dtype=np.uint32) % 65521).astype(np.uint16).reshape(rows, cols)
        flm = matrix.reshape(rows // 64, 64, cols // 256, 256).transpose(0, 2, 1, 3)
        np.testing.assert_array_equal(source_matrix(flm, rows, cols), matrix)
        packed = npu_weight(flm, rows, cols)
        np.testing.assert_array_equal(untile_b(packed, cols, rows, 64, 32, 8, 8), matrix.T)
        self.assertEqual(packed.nbytes, matrix.nbytes)

    def test_bad_weight_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "byte count"):
            source_matrix(np.zeros(10, np.uint16), 256, 256)

    def test_row_major_12b_projection_uses_the_same_image(self):
        rows, cols = 3840, 640
        matrix = (np.arange(rows * cols, dtype=np.uint32) % 65521).astype(np.uint16).reshape(rows, cols)
        packed = npu_weight_row_major(matrix, rows, cols)
        np.testing.assert_array_equal(untile_b(packed, cols, rows, 64, 32, 8, 8), matrix.T)

    def test_matrix_plan_includes_both_audio_subsamplers(self):
        plan = image_plan([{}], ["fixture"])
        self.assertEqual(plan["format"], "phlegm-gemma4-multimodal-mm-v2")
        self.assertEqual(set(plan["images"]), {"m256_n256_k64", "m256_n256_k1152"})


if __name__ == "__main__":
    unittest.main()
