# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-FLINT, OPEN-ATTN-SINK
# (canonical spec: specs/open-engine/spec.md)
"""Recursive Flint: GQA attention with Qwen3's q / k norms and GPT-OSS's per-head sink, a 513-row
sliding window with RoPE on 35 layers and NO positional encoding on the 14 full ones, two leading
dense SiLU MLPs, then 256 routed SiLU experts (sigmoid router, top 6, renormalised, no bias) beside
an ungated 1024-wide shared expert.

The derivation, the recipe's emission and the reference's arithmetic run here -- pure functions over
a config dict and synthetic blocks, no model and no hardware. The fixture is RecursiveSI/flint's
config.json @ 2a247be0 (fetched 2026-10-31), written out.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from recipes import families
from recipes.spec import DENSE, DENSE_LOCAL, ModelSpec, SpecError

ROOT = Path(__file__).resolve().parents[3]
FULL = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
HF_FLINT = {
    "architectures": ["FlintForCausalLM"], "model_type": "flint", "attention_bias": False,
    "bos_token_id": 150000, "eos_token_id": [150001, 150003, 150005, 150018], "decoder_sparse_step": 1,
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 6144,
    "layer_types": ["full_attention" if i in FULL else "sliding_attention" for i in range(49)],
    "max_position_embeddings": 262144, "mlp_only_layers": [0, 1], "moe_intermediate_size": 512,
    "norm_topk_prob": True, "num_attention_heads": 32, "num_experts": 256, "num_experts_per_tok": 6,
    "num_hidden_layers": 49, "num_key_value_heads": 4, "no_rope_layer_indices": FULL, "pad_token_id": 150001,
    "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "router_aux_loss_coef": 0.0005,
    "router_score_function": "sigmoid", "shared_expert_intermediate_size": 1024, "sliding_window": 513,
    "softmax_type": "learnable", "tie_word_embeddings": False, "use_sliding_window": True, "vocab_size": 150144,
}


def spec(**over):
    cfg = copy.deepcopy(HF_FLINT)
    cfg.update(over)
    return ModelSpec.from_hf_config(cfg)


# ---- the derivation


def test_the_spec_is_its_own_family_with_the_published_geometry():
    s = spec()
    assert s.family == "flint"
    assert (s.hidden, s.num_layers, s.vocab) == (2048, 49, 150144)
    assert (s.num_heads, s.num_kv_heads, s.head_dim, s.rotary_dim) == (32, 4, 128, 128)
    assert (s.num_experts, s.experts_per_tok, s.moe_intermediate) == (256, 6, 512)
    assert (s.shared_expert_intermediate, s.intermediate, s.dense_lead) == (1024, 6144, 2)
    assert s.qk_norm and not s.attn_gate and s.activation == "silu"
    assert s.sliding_window == 513 and s.norm_eps == 1e-6 and s.rope_theta == 1e4


def test_full_layers_are_dense_and_sliding_layers_dense_local():
    s = spec()
    assert [i for i, t in enumerate(s.layer_types) if t == DENSE] == FULL
    assert s.layer_types.count(DENSE_LOCAL) == 35


def test_the_default_layer_pattern_is_hfs_own():
    cfg = {k: v for k, v in HF_FLINT.items() if k not in ("layer_types", "no_rope_layer_indices")}
    s = ModelSpec.from_hf_config(cfg)
    assert [i for i, t in enumerate(s.layer_types) if t == DENSE] == FULL


@pytest.mark.parametrize("key,value", [
    ("softmax_type", "vanilla"), ("router_score_function", "softmax"), ("hidden_act", "gelu"),
    ("attention_bias", True), ("norm_topk_prob", False), ("router_expert_bias", True),
    ("rope_scaling", {"rope_type": "yarn", "factor": 4.0}), ("decoder_sparse_step", 2),
])
def test_a_config_that_is_not_flint_arithmetic_is_refused_by_name(key, value):
    with pytest.raises(SpecError, match=key.split("_")[0]):
        spec(**{key: value})


def test_rope_must_be_exactly_the_sliding_layers():
    with pytest.raises(SpecError, match="no_rope_layer_indices"):
        spec(no_rope_layer_indices=FULL[:-1])


def test_the_dense_layers_must_lead():
    with pytest.raises(SpecError, match="mlp_only_layers"):
        spec(mlp_only_layers=[0, 2])


def test_the_spec_round_trips_and_dense_lead_hides_on_other_families():
    s = spec()
    assert ModelSpec.from_dict(s.to_dict()) == s
    assert s.to_dict()["dense_lead"] == 2
    from test_gptoss import HF_GPTOSS_20B
    assert "dense_lead" not in ModelSpec.from_hf_config(HF_GPTOSS_20B).to_dict()


def test_the_family_routes_to_its_own_recipe():
    s = spec()
    assert families.for_spec(s).__name__ == "recipes.flint"
    assert "flint" in families.FAMILIES
    from recipes import dense, gptoss
    with pytest.raises(Exception):
        gptoss.geometry(s)
    with pytest.raises(Exception):
        dense._check(s)


# ---- the recipe


def test_the_widths_need_no_padding_and_tile_eight_cores():
    from recipes import flint as F
    G = F.geometry(spec())
    assert (G.HID, G.QW, G.KVW, G.FF) == (2048, 4096, 512, 512)
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (8, 1, 4, 1, 4)
    assert (G.SLOTS, G.DENSE_SLOTS, G.POOL_EXPERTS) == (8, 12, 258)
    assert (G.ACORES, G.NHL) == (4, 8)


def test_the_attention_core_has_the_sink_and_the_qk_norm():
    from recipes import flint as F
    fl = F.attn_flags(F.geometry(spec()))
    for f in ("-DATTN_SINK=1", "-DATTN_QKNORM=1", "-DATTN_QKNORM_POST=0", "-DATTN_GATE=0", "-DATTN_HD=128",
              "-DATTN_NH=32", "-DATTN_KVH=4", "-DATTN_NHL=8", "-DATTN_ROT=128"):
        assert f in fl


def test_full_layers_take_the_identity_rotation_and_sliding_layers_the_window():
    from recipes import flint as F
    s = spec()
    P = F.programs(s)
    assert P["globals"]["ptab"]["inv_freq"] == [0.0] * 64 and P["globals"]["ptab"]["window"] == 0
    assert P["globals"]["ptab_local"]["inv_freq"] == s.rope_inv_freq()
    assert P["globals"]["ptab_local"]["window"] == 513
    assert P["kernels"]["fl_att_local"]["window"] == 513 and P["kernels"]["fl_att"]["window"] == 0
    from recipes.pack import ptab
    t = ptab(4, 128, 1e4, 1024, [0.0] * 64)
    cs = t.reshape(4, 1024)[:, 512:1024].copy().view(np.float32).reshape(4, 2, 64)
    assert (cs[:, 0] == 1.0).all() and (cs[:, 1] == 0.0).all(), "NoPE is the exact identity rotation"


def test_a_moe_layer_is_attention_route_moe_and_a_dense_layer_has_no_route():
    from recipes import flint as F
    s = spec()
    P = F.programs(s)
    moe = P["layer_types"]["dense_local"]["program"]
    assert [st["op"] for st in moe] == ["run", "flint_route", "run"]
    assert moe[1]["shared"] == [256, 257] and moe[1]["topk"] == 6 and moe[1]["experts"] == 256
    d = P["layer_types"]["dense_mlp"]["program"]
    assert [(st["op"], st["kernel"]) for st in d] == [("run", "fl_att_d"), ("run", "fl_dense")]
    assert F.layer_type_names(s)[:3] == ["dense_mlp", "dense_mlp", "dense_local"]
    assert F.moe_geometry(s)["experts"] == 258 and F.moe_geometry(s)["topk"] == 8


def test_the_manifest_is_its_own_gguf_view_with_the_flint_layer_names():
    from recipes.manifest import manifest
    m = manifest(spec())
    assert m["layers"][:2] == ["dense_mlp", "dense_mlp"] and m["layers"][5] == "dense"
    assert m["gguf"]["layers"] == m["layers"] and m["gguf"]["pack"] == m["pack"]
    assert "gguf" not in m["gguf"]
    assert m["pack"]["chunk_bytes"] == 5120


def test_the_pool_slots_and_the_shared_halves():
    from recipes import flint as F
    s = spec()
    L, P = F.layout(s), F.pool(s, dense=False)
    assert L.UP_BYTES == 8 * 16 * 5120 and L.EXPERT_UG == 2 * L.UP_BYTES
    ops = F.pack_plan(s)["layer_types"]["dense_local"]["pool"]
    sh = [o for o in ops if "shexp" in o["tensor"]]
    assert [o["dst"] for o in sh] == [256 * L.EXPERT_UG, 256 * L.EXPERT_UG + F.band_bytes(2048),
                                      P.POOL_DOWN + 256 * L.UP_BYTES]
    assert sh[2]["cols"] == 512 and sh[2]["experts"] == 2, "down's halves are its column slices"
    assert P.POOL_BYTES < 1 << 32


def test_the_routing_weights_sit_past_xm_in_its_last_element():
    from recipes import flint as F
    s = spec()
    L, G = F.layout(s), F.geometry(s)
    assert L.AD_W >= L.AD_XM + G.HID * 2 and L.AD_W + 32 <= L.AD_XM + G.XM_ELEMS * 4096


# ---- the packer and the reference's arithmetic


def test_the_numpy_packer_is_q4_1_packs_law():
    sys.path.insert(0, str(ROOT / "open_kernels"))
    import q4_1_pack as Q
    from recipes import flint_pack as FP
    rng = np.random.default_rng(3)
    b = Q.random_q4_1_blocks(128, 1024, rng)
    assert np.array_equal(FP.matrix(b, 128, 1024), Q.pack_q4_1_pool(b, 2))
    assert np.array_equal(FP.matrix(b, 64, 512, row0=64, col0=512),
                          Q.pack_q4_1_pool(np.ascontiguousarray(b[64:, 16:]), 2))
    ug = FP.slot_ug(FP.matrix(b, 64, 1024), FP.matrix(b, 64, 1024, row0=64), 1024)
    bb = FP.band_bytes(1024)
    assert np.array_equal(ug[bb:2 * bb], FP.matrix(b, 64, 1024, row0=64))


def test_the_reference_router_is_sigmoid_topk_renormalised():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    from flint_gguf_reference import route
    rng = np.random.default_rng(0)
    wr = rng.standard_normal((256, 64)).astype(np.float32)
    x = rng.standard_normal((3, 64)).astype(np.float32)
    top, w = route(x, wr, 6, True)
    s = 1 / (1 + np.exp(-(x.astype(np.float64) @ wr.T.astype(np.float64))))
    for t in range(3):
        want = np.argsort(-s[t], kind="stable")[:6]
        assert list(top[t]) == list(want)
        assert np.allclose(w[t], s[t][want] / s[t][want].sum(), rtol=1e-6)


def test_the_reference_sink_takes_weight_and_the_window_counts_itself():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    from flint_gguf_reference import attention
    rng = np.random.default_rng(1)
    T, H, KVH, hd = 6, 4, 2, 8
    q, k, v = (rng.standard_normal((T, n, hd)).astype(np.float32) for n in (H, KVH, KVH))
    no_sink = attention(q, k, v, np.full(H, -1e9, np.float32), 0)
    big = attention(q, k, v, np.full(H, 1e3, np.float32), 0)
    assert np.abs(big).max() < 1e-30, "a dominant sink takes all the weight and has no value"
    w3 = attention(q, k, v, np.full(H, -1e9, np.float32), 3)
    # query 5 with window 3 sees keys 3, 4, 5 only: the same as attending over those three rows
    sub = attention(q[3:], k[3:], v[3:], np.full(H, -1e9, np.float32), 0)
    assert np.allclose(w3[5], sub[2], atol=1e-6) and not np.allclose(w3[5], no_sink[5])
