# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-HYV3
# (canonical spec: specs/open-engine/spec.md)
"""Tencent HY V3 (Hy-Memory-30B-A3B): GQA attention with Qwen3's q / k norms and a full-head NEOX RoPE on
every layer, one leading dense SiLU MLP (6912), then 128 routed SiLU experts (sigmoid router with a
per-expert selection bias, top 8 renormalised and scaled by 2.826) beside an ungated shared expert. The
experts are 768 wide although the published config says moe_intermediate_size 1536.

The derivation, the recipe's emission and the reference's arithmetic run here -- pure functions over a
config dict and synthetic blocks, no model and no hardware. The fixture is tencent/Hy-Memory-30B-A3B's
config.json @ 64a460a9 (fetched 2026-10-31), written out.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from recipes import families
from recipes.spec import DENSE, ModelSpec, SpecError

ROOT = Path(__file__).resolve().parents[3]
HF_HY = {
    "architectures": ["HYV3ForCausalLM"], "model_type": "hy_v3", "attention_bias": False,
    "attention_dropout": 0.0, "bos_token_id": 120000, "enable_lm_head_fp32": True,
    "enable_moe_fp32_combine": True, "eod_token_id": 120026, "eos_token_id": 120025, "expert_hidden_dim": 768,
    "first_k_dense_replace": 1, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048,
    "initializer_range": 0.006, "intermediate_size": 6912, "max_position_embeddings": 262144,
    "mlp_bias": False, "mlp_layer_types": ["dense"] + ["sparse"] * 47, "moe_intermediate_size": 1536,
    "moe_router_enable_expert_bias": True, "moe_router_use_sigmoid": True, "num_attention_heads": 32,
    "num_experts": 128, "num_experts_per_tok": 8, "num_hidden_layers": 48, "num_key_value_heads": 4,
    "num_nextn_predict_layers": 0, "num_shared_experts": 1, "output_router_logits": True,
    "pad_token_id": 120002, "qk_norm": True, "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"}, "route_norm": True,
    "router_scaling_factor": 2.826, "sep_token_id": 120007, "tie_word_embeddings": False,
    "transformers_version": "5.6.0.dev0", "use_cache": True, "use_grouped_mm": False, "vocab_size": 120832,
}


def spec(**over):
    cfg = copy.deepcopy(HF_HY)
    cfg.update(over)
    return ModelSpec.from_hf_config(cfg)


# ---- the derivation


def test_the_spec_is_its_own_family_with_the_stored_expert_width():
    s = spec()
    assert s.family == "hyv3"
    assert (s.hidden, s.num_layers, s.vocab) == (2048, 48, 120832)
    assert (s.num_heads, s.num_kv_heads, s.head_dim, s.rotary_dim) == (32, 4, 128, 128)
    assert (s.num_experts, s.experts_per_tok, s.moe_intermediate) == (128, 8, 768)
    assert (s.shared_expert_intermediate, s.intermediate, s.dense_lead) == (768, 6912, 1)
    assert s.qk_norm and not s.attn_gate and s.activation == "silu" and not s.sliding_window
    assert s.norm_eps == 1e-5 and s.rope_theta == 11158840.0
    assert set(s.layer_types) == {DENSE}
    assert s.extra["router_scale"] == 2.826 and s.extra["published_moe_intermediate"] == 1536


def test_without_expert_hidden_dim_the_published_width_is_used():
    cfg = {k: v for k, v in HF_HY.items() if k != "expert_hidden_dim"}
    cfg["moe_intermediate_size"] = 768
    assert ModelSpec.from_hf_config(cfg).moe_intermediate == 768


def test_an_unknown_width_pairing_is_refused():
    with pytest.raises(SpecError, match="expert_hidden_dim"):
        spec(moe_intermediate_size=1024)


def test_the_default_mlp_pattern_is_hfs_own():
    cfg = {k: v for k, v in HF_HY.items() if k != "mlp_layer_types"}
    assert ModelSpec.from_hf_config(cfg).dense_lead == 1


@pytest.mark.parametrize("key,value", [
    ("moe_router_use_sigmoid", False), ("moe_router_enable_expert_bias", False), ("route_norm", False),
    ("hidden_act", "gelu"), ("attention_bias", True), ("qk_norm", False), ("num_nextn_predict_layers", 1),
    ("rope_parameters", {"rope_theta": 1e6, "rope_type": "yarn", "factor": 4.0}),
])
def test_a_config_that_is_not_hy_v3_arithmetic_is_refused_by_name(key, value):
    with pytest.raises(SpecError, match=key.split("_")[0] if key != "rope_parameters" else "rope"):
        spec(**{key: value})


def test_the_dense_layers_must_lead():
    with pytest.raises(SpecError, match="leading"):
        spec(mlp_layer_types=["sparse", "dense"] + ["sparse"] * 46)


def test_the_spec_round_trips():
    s = spec()
    assert ModelSpec.from_dict(s.to_dict()) == s
    assert s.to_dict()["dense_lead"] == 1


def test_the_family_routes_to_its_own_recipe():
    s = spec()
    assert families.for_spec(s).__name__ == "recipes.hyv3"
    assert "hyv3" in families.FAMILIES
    from recipes import dense, flint, gptoss
    for other in (flint.geometry, gptoss.geometry, dense._check):
        with pytest.raises(Exception):
            other(s)


# ---- the recipe


def test_the_pair_law_tiles_eight_cores():
    from recipes import hyv3 as H
    G = H.geometry(spec())
    assert (G.HID, G.QW, G.KVW, G.FF) == (2048, 4096, 512, 768)
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UG_PC, G.DOWN_PC, G.CPS) == (8, 1, 4, 3, 4, 4)
    assert (G.SLOTS, G.SHARED, G.POOL_EXPERTS) == (9, 1, 129)
    assert [H.core_slots(G, c) for c in (0, 3, 4, 7)] == [[0, 2, 4, 6, 8]] * 2 + [[1, 3, 5, 7]] * 2
    # every slot's 12 bands are covered once: 4 cores x 3 bands
    for s_ in range(G.SLOTS):
        assert len(H.slot_cores(G, s_)) * G.UG_PC * 64 == G.FF


def test_the_attention_core_has_the_qk_norm_and_no_sink():
    from recipes import hyv3 as H
    fl = H.attn_flags(H.geometry(spec()))
    for f in ("-DATTN_QKNORM=1", "-DATTN_QKNORM_POST=0", "-DATTN_GATE=0", "-DATTN_HD=128", "-DATTN_NH=32",
              "-DATTN_KVH=4", "-DATTN_NHL=8", "-DATTN_ROT=128", "-DATTN_EPS=1e-05f"):
        assert f in fl
    assert not any("SINK" in f for f in fl)


def test_every_layer_runs_attention_route_moe_and_the_dense_layer_routes_its_slices():
    from recipes import hyv3 as H
    s = spec()
    P = H.programs(s)
    assert P["globals"]["ptab"]["inv_freq"] == s.rope_inv_freq() and P["globals"]["ptab"]["window"] == 0
    moe = P["layer_types"]["dense"]["program"]
    assert [(st["op"], st["kernel"]) for st in moe] == [("run", "hy_att"), ("hy_route", "hy_moe"), ("run", "hy_moe")]
    r = moe[1]
    assert r["shared"] == [128] and r["topk"] == 8 and r["experts"] == 128 and r["slots"] == 9
    assert r["scale"] == 2.826 and r["bias"] == "blk.{l}.exp_probs_b"
    d = P["layer_types"]["dense_mlp"]["program"]
    assert d[1] == {"op": "hy_route", "kernel": "hy_moe", "a_xm": r["a_xm"], "a_w": r["a_w"], "hidden_real": 2048,
                    "slots": 9, "dense": True}
    assert H.layer_type_names(s)[:2] == ["dense_mlp", "dense"]
    assert H.moe_geometry(s)["experts"] == 129 and H.moe_geometry(s)["topk"] == 9


def test_the_manifest_is_its_own_gguf_view_with_a_q8_head():
    from recipes.manifest import manifest
    m = manifest(spec())
    assert m["layers"][:2] == ["dense_mlp", "dense"] and len(m["layers"]) == 48
    assert m["gguf"]["layers"] == m["layers"] and m["gguf"]["pack"] == m["pack"]
    assert "gguf" not in m["gguf"]
    assert m["pack"]["chunk_bytes"] == 5120
    assert m["pack"]["lm_head"]["ops"][0]["op"] == "gx_lmhead"
    assert m["hf_config_check"]["expert_hidden_dim"] == 768 and "moe_intermediate_size" not in m["hf_config_check"]


def test_the_pool_slots_and_the_dense_slices():
    from recipes import hyv3 as H
    s = spec()
    L = H.layout(s)
    assert L.UP_BYTES == 12 * 16 * 5120 == 32 * 6 * 5120 and L.EXPERT_UG == 2 * L.UP_BYTES
    ops = H.pack_plan(s)["layer_types"]["dense"]["pool"]
    sh = [o for o in ops if "shexp" in o["tensor"]]
    assert [o["dst"] for o in sh] == [128 * L.EXPERT_UG, 128 * L.EXPERT_UG + H.band_bytes(2048),
                                      L.POOL_DOWN + 128 * L.UP_BYTES]
    d = H.pack_plan(s)["layer_types"]["dense_mlp"]["pool"]
    gate, up, down = d[:3]
    assert gate["experts"] == up["experts"] == down["experts"] == 9 and gate["n_total"] == 768
    assert down["cols"] == 768 and down["in_dim"] == 768, "down's slices are its column blocks"
    assert L.POOL_BYTES < 1 << 32
    # the dense layer's attention is the MoE layers' (one pool law)
    assert [o["dst"] for o in d[3:]] == [o["dst"] for o in ops if o["tensor"].startswith("blk.{l}.attn")]


def test_the_routing_record_sits_past_xm_in_its_last_element():
    from recipes import hyv3 as H
    s = spec()
    L, G = H.layout(s), H.geometry(s)
    assert L.AD_W >= L.AD_XM + G.HID * 2 and L.AD_W + 4 * H.WV <= L.AD_XM + G.XM_ELEMS * 4096


def test_the_generated_tus_match_the_recipe():
    sys.path.insert(0, str(ROOT / "open_kernels" / "designs" / "hyv3"))
    import gen_kernels as GK
    from recipes import hyv3 as H
    fs = GK.files(H.recipe(spec()))
    assert "wv[2 * j + half]" in fs["hy_act.cc"] and "load_v<16>" in fs["hy_wgrab.cc"]
    here = ROOT / "open_kernels" / "designs" / "hyv3"
    for name, text in fs.items():
        assert (here / name).read_text() == text, f"{name} is stale: run gen_kernels.py"


# ---- the reference's arithmetic


def test_the_reference_router_selects_on_the_bias_and_weighs_without_it():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    from hyv3_gguf_reference import route
    rng = np.random.default_rng(0)
    wr = rng.standard_normal((128, 64)).astype(np.float32)
    bias = (rng.standard_normal(128) * 0.5).astype(np.float32)
    x = rng.standard_normal((3, 64)).astype(np.float32)
    top, w = route(x, wr, bias, 8, 2.826)
    s = 1 / (1 + np.exp(-(x.astype(np.float64) @ wr.T.astype(np.float64))))
    for t in range(3):
        want = np.argsort(-(s[t] + bias), kind="stable")[:8]
        assert list(top[t]) == list(want)
        assert np.allclose(w[t], s[t][want] / s[t][want].sum() * 2.826, rtol=1e-6)
        assert list(want) != list(np.argsort(-s[t], kind="stable")[:8]), "the bias must change the choice here"


def test_the_reference_attention_is_causal_softmax():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    from hyv3_gguf_reference import attention
    rng = np.random.default_rng(1)
    T, H, KVH, hd = 5, 4, 2, 8
    q, k, v = (rng.standard_normal((T, n, hd)).astype(np.float32) for n in (H, KVH, KVH))
    a = attention(q, k, v)
    assert np.allclose(a[0], np.repeat(v[0], H // KVH, axis=0), atol=1e-6), "position 0 sees only itself"
    sc = np.einsum("hd,lhd->hl", q[4].reshape(KVH, 2, hd)[:, 0], k) / np.sqrt(hd)
    p = np.exp(sc - sc.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    assert np.allclose(a[4, 0], (p[0][:, None] * v[:, 0]).sum(0), atol=1e-5)
