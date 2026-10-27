"""Check FLM 64×256 vision tiles and strict 4B tensor shape validation."""
import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_source import matrix, required_header, validate_header


class Reader:
    def __init__(self, name, tiles):
        self.tensors = {name: {"dtype": "BF16", "shape": list(tiles.shape)}}
        self.payload = tiles.tobytes()

    def raw(self, _name):
        return self.payload


class Gemma3VisionSourceTest(unittest.TestCase):
    def test_full_container_header_is_checked_before_npu_work(self):
        expected = required_header()
        self.assertEqual(len(expected), 439)
        reader = type("HeaderReader", (), {})()
        reader.data_base = 0
        reader.tensors = {}
        offset = 0
        for name, shape in expected.items():
            end = offset + 2 * int(np.prod(shape))
            reader.tensors[name] = {"dtype": "BF16", "shape": list(shape),
                                    "data_offsets": [offset, end]}
            offset = end
        reader.mm = type("LengthOnly", (), {"__len__": lambda _: offset})()
        self.assertEqual(validate_header(reader), 439)
        bad = "vision_tower.vision_model.encoder.layers.26.mlp.fc2.weight"
        reader.tensors[bad]["data_offsets"][1] -= 2
        with self.assertRaisesRegex(ValueError, "invalid BF16 byte extent"):
            validate_header(reader)
        reader.tensors[bad]["data_offsets"] = [0, 2 * int(np.prod(expected[bad]))]
        with self.assertRaisesRegex(ValueError, "overlapping vision tensors"):
            validate_header(reader)

    def test_flm_tile_inverse_and_padding(self):
        logical = np.zeros((4304, 1152), bfloat16)
        logical[0, 0] = 1
        logical[63, 255] = 2
        logical[64, 256] = 3
        logical[-1, -1] = 4
        padded = np.zeros((4352, 1280), bfloat16)
        padded[:4304, :1152] = logical
        tiles = padded.reshape(4352 // 64, 64, 1280 // 256, 256).transpose(0, 2, 1, 3)
        tiles = np.ascontiguousarray(tiles.reshape(4352 // 64, 1280 // 256, 16384))
        decoded = matrix(Reader("fc1", tiles), "fc1", 4304, 1152)
        self.assertTrue(np.array_equal(decoded, logical))

    def test_nonzero_padding_rejected(self):
        padded = np.zeros((1280, 1280), bfloat16)
        padded[1152, 0] = 1
        tiles = padded.reshape(20, 64, 5, 256).transpose(0, 2, 1, 3)
        tiles = np.ascontiguousarray(tiles.reshape(20, 5, 16384))
        with self.assertRaisesRegex(ValueError, "nonzero FLM tile padding"):
            matrix(Reader("q", tiles), "q", 1152, 1152)


if __name__ == "__main__":
    unittest.main()
