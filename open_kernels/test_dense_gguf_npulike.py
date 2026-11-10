import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'model'))
from dense_gguf_npulike import bf16, kept_q4, q4_refit, q8_requant


def test_bf16_rounds_ties_to_even():
    x = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000], dtype=np.uint32).view(np.float32)
    assert bf16(x).view(np.uint32).tolist() == [0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000]


@pytest.mark.parametrize('typ,size', [('Q4_0', 18), ('Q4_1', 20), ('Q4_K', 144)])
def test_kept_codes_agree_with_gguf_at_exact_bf16_scales(typ, size):
    # The test environment supplies a trusted gguf-py on PYTHONPATH.
    gguf = pytest.importorskip('gguf')
    rng = np.random.default_rng(17)
    a = rng.integers(0, 256, (3, size), dtype=np.uint8)
    a[:, :2] = np.array([0, 60], dtype=np.uint8)  # fp16 1.0
    if typ != 'Q4_0': a[:, 2:4] = np.array([0, 56], dtype=np.uint8)  # fp16 0.5
    expected = gguf.dequantize(a, getattr(gguf.GGMLQuantizationType, typ))
    np.testing.assert_array_equal(kept_q4(a, typ), expected)


def test_requantization_zero_and_constant_blocks():
    x = np.stack([np.zeros(32), np.full(32, -2), np.linspace(-1, 1, 32)]).astype(np.float32)
    for fn in (q4_refit, q8_requant):
        got = fn(x)
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got[:2], x[:2], atol=.001)
        assert np.max(np.abs(got[2] - x[2])) < .08


def test_tied_embedding_remains_input_and_gets_separate_head(tmp_path):
    import subprocess
    gguf = pytest.importorskip('gguf')
    source = tmp_path / 'in.gguf'
    dest = tmp_path / 'out.gguf'
    embedding = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) / 100
    raw = gguf.quantize(np.linspace(-1, 1, 32 * 32, dtype=np.float32).reshape(32, 32),
                        gguf.GGMLQuantizationType.Q4_0)
    w = gguf.GGUFWriter(source, 'llama')
    w.add_tensor('token_embd.weight', embedding)
    w.add_tensor('blk.0.attn_q.weight', raw, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    module = Path(__file__).parent / 'model' / 'dense_gguf_npulike.py'
    subprocess.run([sys.executable, '-I', str(module), str(source), str(dest), '--mode', 'q6-int8',
                    '--gguf-py', str(Path(gguf.__file__).parent.parent)], check=True, capture_output=True)
    t = {x.name: x for x in gguf.GGUFReader(dest).tensors}
    np.testing.assert_array_equal(t['token_embd.weight'].data, embedding)
    np.testing.assert_array_equal(t['output.weight'].data, q8_requant(embedding))
    np.testing.assert_array_equal(t['blk.0.attn_q.weight'].data, kept_q4(raw, 'Q4_0'))


def test_q4k_scale_rounding_uses_products_not_rounded_factors():
    gguf = pytest.importorskip('gguf')
    ml = pytest.importorskip('ml_dtypes')
    rng = np.random.default_rng(284)
    raw = rng.integers(0, 256, (1, 144), dtype=np.uint8)
    raw[0, :2] = np.frombuffer(np.float16(.071).tobytes(), np.uint8)
    raw[0, 2:4] = np.frombuffer(np.float16(.037).tobytes(), np.uint8)
    sc, mn = gguf.quants.Q4_K.get_scale_min(raw[:, 4:16])
    d = raw[:, :2].copy().view(np.float16).astype(np.float32)[0, 0]
    m = raw[:, 2:4].copy().view(np.float16).astype(np.float32)[0, 0]
    expected = []
    for i in range(8):
        ds = np.asarray(d * np.float32(sc[0, i]), dtype=ml.bfloat16).astype(np.float32)
        ms = np.asarray(m * np.float32(mn[0, i]), dtype=ml.bfloat16).astype(np.float32)
        packed = raw[0, 16 + (i // 2) * 32:16 + (i // 2 + 1) * 32]
        q = (packed >> (4 * (i % 2))) & 15
        expected.extend(ds * q.astype(np.float32) - ms)
    np.testing.assert_array_equal(kept_q4(raw, 'Q4_K').ravel(), expected)
