"""GGUF expert pool order and exact Q4 block values."""

import numpy as np
import pytest

from gguf_pool import (CH_Q4, chunk_geometry, dequant_q4, dequant_q4_chunk,
                       pack_q4_expert_down, pack_q4_expert_stripes, random_q4_blocks)


@pytest.mark.parametrize("quant", ["Q4_0", "Q4_1"])
def test_expert_stripes_and_down_keep_source_values_and_expert_order(quant):
    rng = np.random.default_rng(291)
    experts, ff, hidden = 2, 256, 512
    up = np.stack([random_q4_blocks(ff, hidden, rng, gguf_type=quant) for _ in range(experts)])
    gate = np.stack([random_q4_blocks(ff, hidden, rng, gguf_type=quant) for _ in range(experts)])
    down = np.stack([random_q4_blocks(hidden, ff, rng, gguf_type=quant) for _ in range(experts)])

    packed_stripes = pack_q4_expert_stripes(up, gate, quant)
    packed_down = pack_q4_expert_down(down, quant)
    stripe_chunks = 128 * hidden // (32 * 256)
    up_gate_chunks = packed_stripes.reshape(experts, ff // 128, 2, stripe_chunks, CH_Q4)
    down_chunks = packed_down.reshape(experts, hidden * ff // (32 * 256), CH_Q4)

    for e in range(experts):
        for role, raw in enumerate((up[e], gate[e])):
            source = dequant_q4(raw, quant)
            _, _, rows, cols = chunk_geometry(ff, hidden, 4)
            for c, (r, k) in enumerate(zip(rows, cols)):
                chunk = up_gate_chunks[e, c // stripe_chunks, role, c % stripe_chunks]
                np.testing.assert_array_equal(dequant_q4_chunk(chunk), source[r:r + 32, k:k + 256])
        source = dequant_q4(down[e], quant)
        _, _, rows, cols = chunk_geometry(hidden, ff, 4)
        for c, (r, k) in enumerate(zip(rows, cols)):
            np.testing.assert_array_equal(dequant_q4_chunk(down_chunks[e, c]), source[r:r + 32, k:k + 256])


def test_expert_pack_rejects_incompatible_dimensions():
    raw = np.zeros((2, 128, 8, 20), np.uint8)
    with pytest.raises(ValueError, match="matching"):
        pack_q4_expert_stripes(raw, raw[:, :127])
    with pytest.raises(ValueError, match="128-row"):
        pack_q4_expert_down(raw[:, :127])
