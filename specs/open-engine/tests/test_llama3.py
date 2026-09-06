# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-LLAMA3 (canonical spec: specs/open-engine/spec.md)
"""Llama 3 on the dense recipe: spec derivation (HF and GGUF), the llama3 RoPE
frequency scaling, the 8B's layout (8 KB norm elements, one chunk per weight
element because of the 32 KB table), and the manifest."""
from __future__ import annotations

import math

import numpy as np
import pytest

from recipes import dense as DR
from recipes import pack
from recipes.manifest import manifest
from recipes.spec import DENSE, ModelSpec, SpecError

HF_LLAMA31_8B = {
    "model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0, "vocab_size": 128256, "tie_word_embeddings": False,
    "rope_scaling": {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"},
}
GGUF_LLAMA31_8B = {
    "general.architecture": "llama", "llama.embedding_length": 4096, "llama.block_count": 32,
    "llama.vocab_size": 128256, "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 128, "llama.rope.freq_base": 500000.0, "llama.feed_forward_length": 14336,
    "llama.attention.layer_norm_rms_epsilon": 1e-05, "llama.rope.scaling.type": "llama3",
    "llama.rope.scaling.factor": 8.0, "llama.rope.scaling.low_freq_factor": 1.0,
    "llama.rope.scaling.high_freq_factor": 4.0, "llama.rope.scaling.original_context_length": 8192,
}


def hf_llama3_inv_freq(theta, dim, factor, lo, hi, old):
    """transformers' _compute_llama3_parameters, verbatim in NumPy."""
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    low_wl, high_wl = old / lo, old / hi
    wl = 2 * math.pi / inv
    out = np.where(wl > low_wl, inv / factor, inv)
    smooth = (old / wl - lo) / (hi - lo)
    smoothed = (1 - smooth) * out / factor + smooth * out
    mid = (wl <= low_wl) & (wl >= high_wl)
    return np.where(mid, smoothed, out)


def test_spec_from_hf_and_gguf_agree():
    a = ModelSpec.from_hf_config(HF_LLAMA31_8B, real_vocab=128256)
    b = ModelSpec.from_gguf_metadata(GGUF_LLAMA31_8B)
    assert a.family == "llama3" and a.layer_types == tuple([DENSE] * 32)
    assert a.qk_norm is False and a.attn_gate is False and a.rotary_dim == 128 and a.norm_eps == 1e-5
    assert a.rope_scaling == {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                              "original_max_position_embeddings": 8192}
    da, db = a.to_dict(), b.to_dict()
    for d in (da, db):
        d.pop("extra")
    assert da == db


def test_llama3_rope_scaling_matches_transformers():
    s = ModelSpec.from_hf_config(HF_LLAMA31_8B)
    got = np.array(s.rope_inv_freq())
    want = hf_llama3_inv_freq(500000.0, 128, 8.0, 1.0, 4.0, 8192)
    assert got.shape == (64,) and np.allclose(got, want, rtol=1e-12)
    assert got[0] == 1.0 and got[-1] < want[0] / 8 * 1.0001    # the lowest frequency is divided by the factor
    plain = ModelSpec.from_hf_config(dict(HF_LLAMA31_8B, rope_scaling=None)).rope_inv_freq()
    assert np.allclose(plain, 500000.0 ** (-np.arange(64) / 64))


def test_unsupported_scaling_and_tied_embeddings_are_refused():
    with pytest.raises(SpecError, match="rope_scaling type 'yarn'"):
        ModelSpec.from_hf_config(dict(HF_LLAMA31_8B, rope_scaling={"rope_type": "yarn", "factor": 2}))
    with pytest.raises(SpecError, match="tied embeddings"):
        ModelSpec.from_hf_config(dict(HF_LLAMA31_8B, tie_word_embeddings=True))


def test_8b_layout_and_manifest(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    spec = ModelSpec.from_hf_config(HF_LLAMA31_8B)
    R = DR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (8, 2, 8, 28, 8)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (2, 2, 2, 14)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (8192, 2048, 4096, 2048)
    assert G.PER_CALL == 1 and G.CALL_BYTES == 5120 and G.TAB_BYTES == 32256   # the K = 14336 table
    assert G.QKNORM is False and G.EPS == 1e-5
    assert L.LMHEAD_BANDS == 2004
    m = manifest(spec)
    assert m["family"] == "llama3" and len(m["layout"]["rope_inv_freq"]) == 64
    assert "head_dim" not in m["hf_config_check"] and m["hf_config_check"]["intermediate_size"] == 14336
    consts = m["layer_types"][DENSE]["pack"]["consts"]
    assert [o["tensor"].split(".")[-2] for o in consts] == ["input_layernorm", "post_attention_layernorm"]
    assert m["builds"]["dx"]["build_dir"] == "dense/build_llama3_h4096"


def test_qwen3_4b_keeps_two_chunks_per_element():
    from test_qwen3_dense import HF_QWEN3_4B
    assert DR.per_call(ModelSpec.from_hf_config(HF_QWEN3_4B)) == 2


def test_ptab_uses_the_inverse_frequencies():
    inv = [0.5, 0.25]
    t = pack.ptab(3, 4, 10.0, 1024, inv).reshape(3, 1024)
    cos = t[2, 512:520].view(np.float32)
    sin = t[2, 520:528].view(np.float32)
    assert np.allclose(cos, np.cos([1.0, 0.5])) and np.allclose(sin, np.sin([1.0, 0.5]))
