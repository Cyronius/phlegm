import unittest

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_source import q4_projection_pool, q8_head_pool, source_shape, tensor_name
from open_kernels.gemma4_text_plan import Projection, TextPlan
from open_kernels.gemma4_text_source import Gemma4TextSource


class Reader:
    def __init__(self):
        self.tensors = {}
        self.data_base = 0
        self._payload = bytearray()
        self.mm = b""

    def add(self, name, dtype, shape, raw):
        start = len(self._payload)
        self._payload.extend(raw)
        self.tensors[name] = {"dtype": dtype, "shape": list(shape),
                              "data_offsets": [start, len(self._payload)]}
        self.mm = bytes(self._payload)


def plan(variant, hidden=512, vocab=128):
    return TextPlan(variant, hidden, vocab, 1024, 30.0, 1e-6, ())


class Gemma4TextSourceTest(unittest.TestCase):
    def test_reads_only_selected_q4_band_and_k_stripe(self):
        projection = Projection(0, "q", 128, 1024)
        source = np.zeros((16, 5120), np.uint8)
        source[:, 1024] = np.arange(16)
        reader = Reader()
        reader.add(tensor_name(projection), *source_shape(plan("E2B"), projection), source.tobytes())
        adapter = Gemma4TextSource(reader, plan("E2B"))
        selected = source.reshape(4, 4, 5120)[2:, 1].reshape(-1)
        expected = q4_projection_pool(selected, 64, 256, source_tile_bytes=5120)
        np.testing.assert_array_equal(adapter.projection_band(projection, 1, 1, 256), expected)
        with self.assertRaisesRegex(ValueError, "band or stripe"):
            adapter.projection_band(projection, 2, 0, 256)

    def test_signed_q4_and_q8_head_use_paired_tiles(self):
        projection = Projection(0, "q", 128, 512)
        signed = np.zeros((16, 2304), np.uint8)
        signed[:, 256] = np.arange(16)
        head = np.zeros((16, 4352), np.uint8)
        head[:, 256] = np.arange(16)
        reader = Reader()
        reader.add(tensor_name(projection), *source_shape(plan("12B"), projection), signed.tobytes())
        reader.add("lm_head.weight", "I8", (8, 2, 4352), head.tobytes())
        adapter = Gemma4TextSource(reader, plan("12B"))
        np.testing.assert_array_equal(
            adapter.projection_band(projection, 1),
            q4_projection_pool(signed[8:], 64, 512, source_tile_bytes=2304))
        np.testing.assert_array_equal(adapter.head_band(0), q8_head_pool(head, 128, 512))

    def test_bf16_projection_is_source_stream_and_embedding_is_scaled(self):
        projection = Projection(0, "per_layer_input_gate", 32, 512, "bf16")
        weight = np.arange(32 * 512, dtype=np.float32).astype(bfloat16).reshape(1, 512, 32)
        reader = Reader()
        reader.add(tensor_name(projection), *source_shape(plan("E2B"), projection), weight.tobytes())
        q = np.arange(128 * 512, dtype=np.int8).reshape(128, 512)
        scale = np.full((128, 16), 0.25, np.float32)
        reader.add("model.embed_tokens.weight", "I8", (128, 512), q.tobytes())
        reader.add("model.embed_tokens.weight.scale", "F32", (128, 16), scale.tobytes())
        adapter = Gemma4TextSource(reader, plan("E2B"))
        np.testing.assert_array_equal(adapter.bf16_projection(projection), weight)
        np.testing.assert_array_equal(adapter.embedding(7), q[7].astype(np.float32) * .25)
        with self.assertRaisesRegex(ValueError, "out of range"):
            adapter.embedding(128)

    def test_rejects_bad_extent_before_reading(self):
        projection = Projection(0, "q", 64, 256)
        reader = Reader()
        reader.add(tensor_name(projection), *source_shape(plan("E2B"), projection), bytes(2 * 5120))
        adapter = Gemma4TextSource(reader, plan("E2B"))
        reader.tensors[tensor_name(projection)]["data_offsets"][1] += 1
        with self.assertRaisesRegex(ValueError, "byte extent"):
            adapter.projection_band(projection, 0)


if __name__ == "__main__":
    unittest.main()
