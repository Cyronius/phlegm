# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-XING
# (canonical spec: specs/open-engine/spec.md)
"""Xing 4.0 (Xing4.0-29B-A4B): a DeepSeek-V3 decoder -- MLA attention (q / kv lora bottlenecks, one shared rope key,
YaRN), two dense SiLU MLPs then 64 routed SiLU experts (sigmoid router, selection-only bias, top 4, renormalised x 2)
beside a shared expert -- whose residual is DeepSeek's mHC: four streams mixed by a Sinkhorn matrix around every
sublayer, averaged before the final norm.

The derivation, the recipe's emission and the reference's arithmetic run here -- pure functions over a config dict and
synthetic arrays, no model and no hardware. The fixture is XingChen-AGI/Xing4.0-29B-A4B's config.json @ ac0fc307
(fetched 2026-11-08), written out. The reference's attention (the HF non-absorbed form) is checked against the
absorbed form the runtime computes (xing-runtime.patch xg_mla), and its mHC against a direct transcription of
modeling_xing4_0.py's Xing4_0HyperConnection.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from recipes import families
from recipes.spec import DENSE, MOE_BLOCK, ModelSpec, SpecError

ROOT = Path(__file__).resolve().parents[3]
HF_XING = {
    "architectures": ["Xing4_0ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "bos_token_id": 1, "eos_token_id": 2, "ep_size": 1, "first_k_dense_replace": 2, "hidden_act": "silu",
    "hidden_size": 3584, "initializer_range": 0.02, "intermediate_size": 9216, "kv_lora_rank": 512,
    "max_position_embeddings": 262144, "model_type": "xing4_0", "moe_intermediate_size": 1024, "moe_layer_freq": 1,
    "n_group": 1, "n_routed_experts": 64, "n_shared_experts": 1, "norm_topk_prob": True, "num_attention_heads": 32,
    "num_experts_per_tok": 4, "num_hidden_layers": 40, "num_key_value_heads": 32, "num_nextn_predict_layers": 1,
    "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "mhc_h_res_clamp_min": -30, "mhc_h_res_clamp_max": 30,
    "q_lora_rank": 768, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "rms_norm_eps": 1e-06, "rope_theta": 10000,
    "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 64, "mscale": 1.0, "mscale_all_dim": 1.0,
                     "original_max_position_embeddings": 4096, "type": "yarn"},
    "routed_scaling_factor": 2.0, "scoring_func": "sigmoid", "tie_word_embeddings": False, "topk_group": 1,
    "topk_method": "noaux_tc", "dtype": "bfloat16", "use_cache": True, "v_head_dim": 128, "vocab_size": 131072,
}


def spec(**over):
    cfg = copy.deepcopy(HF_XING)
    cfg.update(over)
    return ModelSpec.from_hf_config(cfg)


def reference():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    import xing_gguf_reference as R
    return R


# ---- the derivation


def test_the_spec_is_its_own_family_with_the_published_geometry():
    s = spec()
    assert s.family == "xing"
    assert (s.hidden, s.num_layers, s.vocab, s.intermediate) == (3584, 40, 131072, 9216)
    assert s.layer_types == (DENSE, DENSE) + (MOE_BLOCK,) * 38
    assert (s.num_heads, s.head_dim, s.rotary_dim, s.rope_theta) == (32, 192, 64, 10000.0)
    assert s.mla == {"q_lora_rank": 768, "kv_lora_rank": 512, "qk_nope_dim": 128, "qk_rope_dim": 64, "v_head_dim": 128}
    assert s.hyper == {"mult": 4, "sinkhorn_iters": 20, "eps": 1e-6, "clamp": 30.0}
    assert (s.num_experts, s.experts_per_tok, s.moe_intermediate, s.shared_expert_intermediate) == (64, 4, 1024, 1024)
    assert s.extra["router_scale"] == 2.0 and s.extra["mtp_layers"] == 1
    assert not s.qk_norm and not s.attn_gate and s.norm_eps == 1e-6 and s.quant == "q4_0"


def test_the_score_scale_is_mscale_squared_over_sqrt_192_and_cos_sin_are_unscaled():
    s = spec()
    mscale = 0.1 * math.log(64) + 1.0
    assert s.attn_scale == pytest.approx(192 ** -0.5 * mscale ** 2, rel=1e-12)
    assert s.rope_scale() == 1.0                      # attention_factor = mscale(1) / mscale_all_dim(1)
    # HF's yarn table (truncate=True) and the reference's own transcription agree pair by pair
    R = reference()
    want = R.yarn_inv_freq(64, 10000.0, 64.0, 4096.0, 32.0, 1.0)
    assert np.allclose(s.rope_inv_freq(), want, rtol=1e-12, atol=0)
    assert s.rope_inv_freq()[0] == 1.0 and s.rope_inv_freq()[-1] == pytest.approx(10000 ** (-62 / 64) / 64)


@pytest.mark.parametrize("over", [
    {"n_group": 8}, {"topk_group": 4}, {"hidden_act": "gelu"}, {"tie_word_embeddings": True},
    {"attention_bias": True}, {"norm_topk_prob": False}, {"rope_interleave": False}, {"q_lora_rank": None},
    {"scoring_func": "softmax"}, {"topk_method": "greedy"}, {"rope_scaling": None}, {"moe_layer_freq": 2},
])
def test_what_the_family_does_not_implement_is_refused(over):
    with pytest.raises(SpecError):
        spec(**over)


def test_the_new_fields_serialise_only_for_xing_and_round_trip():
    s = spec()
    d = s.to_dict()
    assert d["mla"] == s.mla and d["hyper"] == s.hyper
    assert ModelSpec.from_dict(d) == s and ModelSpec.from_dict(d).spec_hash() == s.spec_hash()
    # every other family's dict, and so its hashes, are as before: the keys are absent at None
    from recipes.load import load_spec
    other = load_spec(ROOT / "open_kernels" / "recipes" / "specs" / "qwen3-8b.json")
    assert other.mla is None and other.hyper is None
    assert "mla" not in other.to_dict() and "hyper" not in other.to_dict()


# ---- the recipe


def test_the_layout_pads_nothing_and_keeps_every_offset_32_bit():
    from recipes import xing as X
    s = spec()
    G, L = X.geometry(s), X.layout(s)
    assert (G.QKVA_N, G.QB_N, G.O_K, G.KV_DIM) == (1344, 6144, 4096, 576)
    assert (G.DSLOTS, G.MSLOTS, G.MEXPERTS) == (9, 5, 65)
    for n in (G.QKVA_N, G.QB_N, G.HID, G.FF):
        c = X.split_bands(n // 64)
        assert sum(c) * 64 == n and max(c) - min(c) <= 1
    # the slots' weights sit past xm's hidden values, inside its last element
    assert L.AD_XM + G.HID * 2 <= L.AD_W and L.AD_W + 64 <= L.AD_XM + L.XM_ELEMS * 4096
    assert L.W_OFF == L.AD_W - L.AD_XM - (L.XM_ELEMS - 1) * 4096
    assert L.POOL_QB == 21 * X.band_bytes(3584) and L.POOL_O == L.POOL_QB + 96 * X.band_bytes(768)
    assert L.UPS == 32 * X.band_bytes(3584) and L.DNS == 56 * X.band_bytes(1024)
    assert L.POOL_DOWN_M == L.POOL_EXP + 65 * L.UPS and L.POOL_MOE_BYTES >= L.POOL_DOWN_M + 65 * L.DNS
    assert L.POOL_DOWN_D == L.POOL_EXP + 9 * L.UPS and L.POOL_DENSE_BYTES >= L.POOL_DOWN_D + 9 * L.DNS
    assert L.POOL_BYTES == L.POOL_MOE_BYTES < 1 << 32 and L.KV_ROW == 576 * 4


def test_the_pack_plan_interleaves_gate_and_up_bands_and_slices_the_dense_mlp():
    from recipes import xing as X
    s = spec()
    G, L = X.geometry(s), X.layout(s)
    bb = X.band_bytes(3584)
    plan = X.pack_plan(s)
    lt = plan["layer_types"]
    assert set(lt) == {DENSE, MOE_BLOCK}
    moe = {o["tensor"].split(".", 2)[2]: o for o in lt[MOE_BLOCK]["pool"]}
    assert moe["attn_kv_a_mqa.weight"]["dst"] == 12 * bb and moe["attn_kv_a_mqa.weight"]["rows"] == 576
    assert moe["attn_q_b.weight"]["in_dim"] == 768 and moe["attn_output.weight"]["in_dim"] == 4096
    g, u = moe["ffn_gate_exps.weight"], moe["ffn_up_exps.weight"]
    assert (g["dst"], u["dst"]) == (L.POOL_EXP, L.POOL_EXP + bb) and g["band_stride"] == u["band_stride"] == 2 * bb
    assert g["experts"] == 64 and g["expert_bytes"] == L.UPS and g["n_total"] == 1024
    assert moe["ffn_gate_shexp.weight"]["dst"] == L.POOL_EXP + 64 * L.UPS
    assert moe["ffn_down_shexp.weight"]["dst"] == L.POOL_DOWN_M + 64 * L.DNS
    dense = lt[DENSE]["pool"]
    downs = [o for o in dense if o["tensor"].endswith("ffn_down.weight")]
    assert [o["col0"] for o in downs] == [j * 1024 for j in range(9)]
    assert [o["dst"] for o in downs] == [L.POOL_DOWN_D + j * L.DNS for j in range(9)]
    gd = next(o for o in dense if o["tensor"].endswith("ffn_gate.weight"))
    assert gd["experts"] == 9 and gd["n_total"] == 1024 and gd["band_stride"] == 2 * bb
    for o in dense + lt[MOE_BLOCK]["pool"]:
        assert o["op"] == "nh_q40" and o["nch"] % (o["in_dim"] // 128) == 0 and o["src_dim"] == o["in_dim"]
    assert plan["lm_head"]["ops"][0]["in_dim"] == 3584 and plan["chunk_bytes"] == 4608


def test_the_programs_run_one_image_and_the_manifest_nests_its_gguf_view(monkeypatch):
    from recipes import manifest as M
    # the sequential set (the block route adds its GEMM and xmb contexts: test_xing_block.py)
    monkeypatch.setenv("XING_BLOCK", "0")
    s = spec()
    m = M.manifest(s, 4096, "k")
    assert m["family"] == "xing" and "gguf" in m and m["gguf"]["layer_types"] == m["layer_types"]
    for kind, ffn in ((DENSE, "xg_dense"), (MOE_BLOCK, "xg_moe")):
        prog = m["layer_types"][kind]["program"]
        assert [st["op"] for st in prog] == ["xg_pre", "run", "xg_qnorm", "run", "xg_mla", "run", "xg_post", "xg_pre",
                                             "run", "xg_post"]
        assert [st["kernel"] for st in prog if st["op"] == "run"] == ["xg_qkva", "xg_qb", "xg_o", ffn]
        assert prog[7]["mode"] == ("dense" if kind == DENSE else "route")
        assert m["layer_types"][kind]["buffers"]["state"] == {"kind": "kv", "row": 2304}
    # every layer stream is a stream of ONE image; only the head changes context
    assert {k: v["context"] for k, v in m["kernels"].items()} == {
        "xg_qkva": "xg_one", "xg_qb": "xg_one", "xg_o": "xg_one", "xg_dense": "xg_one", "xg_moe": "xg_one", "lm": "lm"}
    assert m["kernels"]["xg_moe"]["patch"] == "nhroute" and "patch" not in m["kernels"]["xg_dense"]
    assert [st["op"] for st in m["tail"]] == ["xg_final", "run"]
    assert m["globals"]["xres"] == 4 * 3584 * 4
    assert m["layout"]["rotary_dim"] == 64 and len(m["layout"]["rope_inv_freq"]) == 32
    assert set(m["builds"]) == {"xg_qkva", "xg_qb", "xg_o", "xg_dense", "xg_moe", "lm_head_q8"}
    assert m["hf_config_check"]["model_type"] == ["xing4_0"]


def test_only_the_point_that_ran_on_hardware_is_catalogued(monkeypatch):
    from recipes import xing as X
    from recipes.catalogue import OpRangeError
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    X.recipe(spec())
    with pytest.raises(OpRangeError):
        X.recipe(spec(q_lora_rank=1536))


def test_the_family_is_registered():
    assert families.family_module("xing").__name__.endswith("xing")
    assert "xing" in families.FAMILIES


# ---- the reference's arithmetic


class _G:
    def __init__(self, w):
        self._w = w

    def w(self, name):
        return self._w[name]


def test_the_reference_attention_equals_the_absorbed_form_the_runtime_computes():
    R = reference()
    rng = np.random.default_rng(11)
    nh, nope, rot, vd, kvl, ql, hid, T = 4, 16, 8, 12, 24, 20, 32, 9
    p = "blk.0."
    W = {p + "attn_q_a.weight": rng.standard_normal((ql, hid)).astype(np.float32) * 0.3,
         p + "attn_q_a_norm.weight": (1 + 0.1 * rng.standard_normal(ql)).astype(np.float32),
         p + "attn_q_b.weight": rng.standard_normal((nh * (nope + rot), ql)).astype(np.float32) * 0.3,
         p + "attn_kv_a_mqa.weight": rng.standard_normal((kvl + rot, hid)).astype(np.float32) * 0.3,
         p + "attn_kv_a_norm.weight": (1 + 0.1 * rng.standard_normal(kvl)).astype(np.float32),
         p + "attn_k_b.weight": rng.standard_normal((nh, kvl, nope)).astype(np.float32) * 0.3,
         p + "attn_v_b.weight": rng.standard_normal((nh, vd, kvl)).astype(np.float32) * 0.3,
         p + "attn_output.weight": rng.standard_normal((hid, nh * vd)).astype(np.float32) * 0.3}
    inv = R.yarn_inv_freq(rot, 10000.0, 64.0, 4096.0, 32.0, 1.0)
    cfg = {"heads": nh, "nope": nope, "rope": rot, "vdim": vd, "kv_lora": kvl, "eps": 1e-6, "inv": inv, "scale": 0.17}
    h = rng.standard_normal((T, hid))
    pos = np.arange(T)
    got = R.mla(_G(W), p, h, pos, cfg)
    # absorbed: per token t, the KV row [c_t | rope(kr_t)], q_lat = k_b q_nope, scores over rows 0..t, out = v_b ctx
    w = {k: v.astype(np.float64) for k, v in W.items()}
    rms = lambda x, g: x / np.sqrt(np.mean(x * x) + 1e-6) * g  # noqa: E731
    rows = []
    want = np.empty((T, hid))
    for t in range(T):
        kv = w[p + "attn_kv_a_mqa.weight"] @ h[t]
        c = rms(kv[:kvl], w[p + "attn_kv_a_norm.weight"])
        kr = R.rope_pairs(kv[None, None, kvl:], pos[t:t + 1], inv)[0, 0]
        rows.append(np.concatenate([c, kr]))
        q = (w[p + "attn_q_b.weight"] @ rms(w[p + "attn_q_a.weight"] @ h[t], w[p + "attn_q_a_norm.weight"])).reshape(nh, -1)
        out = []
        for hh in range(nh):
            qr = R.rope_pairs(q[None, None, hh, nope:], pos[t:t + 1], inv)[0, 0]
            lat = w[p + "attn_k_b.weight"][hh] @ q[hh, :nope]
            K = np.array(rows)
            s = (K[:, :kvl] @ lat + K[:, kvl:] @ qr) * cfg["scale"]
            e = np.exp(s - s.max())
            ctx = (e / e.sum()) @ K[:, :kvl]
            out.append(w[p + "attn_v_b.weight"][hh] @ ctx)
        want[t] = w[p + "attn_output.weight"] @ np.concatenate(out)
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)


def test_the_reference_mhc_is_the_published_hyper_connection():
    R = reference()
    rng = np.random.default_rng(5)
    T, hc, hid = 3, 4, 16
    S = rng.standard_normal((T, hc, hid))
    fn = rng.standard_normal(((2 + hc) * hc, hc * hid)).astype(np.float32) * 0.5
    base = rng.standard_normal((2 + hc) * hc) * 0.3
    scale = np.array([0.7, 1.3, 5.0])
    collapsed, post, comb = R.hc_pre(S, fn, base, scale, 1e-6, 20, 1e-6)
    for t in range(T):
        # modeling_xing4_0.py Xing4_0HyperConnection.forward, written out per token
        flat = S[t].reshape(-1)
        flat = flat / math.sqrt(float(np.mean(flat * flat)) + 1e-6)
        m = fn.astype(np.float64) @ flat
        pre = 1 / (1 + np.exp(-(m[:hc] * scale[0] + base[:hc])))
        pst = 2 / (1 + np.exp(-(m[hc:2 * hc] * scale[1] + base[hc:2 * hc])))
        cl = np.clip((m[2 * hc:] * scale[2] + base[2 * hc:]).reshape(hc, hc), -30, 30)
        c = np.exp(cl - cl.max(axis=1, keepdims=True))
        for _ in range(20):
            c = c / (c.sum(axis=1, keepdims=True) + 1e-6)
            c = c / (c.sum(axis=0, keepdims=True) + 1e-6)
        assert np.allclose(post[t], pst) and np.allclose(comb[t], c, rtol=1e-12)
        assert np.allclose(collapsed[t], (pre[:, None] * S[t]).sum(0))
        # the new streams: matmul(comb, S) (row i of comb mixes into stream i) plus post_i y
        y = rng.standard_normal(hid)
        new = R.hc_post(y[None], S[t:t + 1], post[t:t + 1], comb[t:t + 1])[0]
        assert np.allclose(new, pst[:, None] * y[None, :] + c @ S[t])
    # Sinkhorn: every column sums to ~1 after the last (column) step (the rows converge more slowly)
    assert np.allclose(comb.sum(axis=1), 1, atol=1e-5)


def test_the_reference_router_picks_with_the_bias_and_weighs_without_it():
    R = reference()
    rng = np.random.default_rng(2)
    T, hid, E, K = 5, 16, 8, 3
    p = "blk.2."
    W = {p + "ffn_gate_inp.weight": rng.standard_normal((E, hid)).astype(np.float32),
         p + "exp_probs_b.bias": np.array([0, 0, 0, 0, 0, 0, 0, 5.0], np.float32)}
    for n, shape in (("ffn_gate_shexp", (4, hid)), ("ffn_up_shexp", (4, hid)), ("ffn_down_shexp", (hid, 4))):
        W[p + n + ".weight"] = np.zeros(shape, np.float32)

    class G(_G):
        def expert(self, name, e):
            return np.full((4, hid) if "down" not in name else (hid, 4), 0.1 * (e + 1), np.float32)
    m = rng.standard_normal((T, hid))
    out = R.moe(G(W), p, m, {"experts": E, "topk": K, "router_scale": 2.0})
    s = 1 / (1 + np.exp(-(m @ W[p + "ffn_gate_inp.weight"].T)))
    for t in range(T):
        idx = np.argsort(-(s[t] + W[p + "exp_probs_b.bias"]), kind="stable")[:K]
        assert 7 in idx                                   # the bias selects expert 7 every time
        wts = s[t, idx] / s[t, idx].sum() * 2.0           # ... but weighs it by its unbiased score
        want = np.zeros(hid)
        for e, wt in zip(idx, wts):
            g = np.full(4, 0.1 * (e + 1) * m[t].sum())
            want += wt * np.full(hid, 0.1 * (e + 1)) * (g / (1 + np.exp(-g)) * g).sum()
        assert np.allclose(out[t], want)


def test_the_npu_like_control_computes_the_same_attention_at_bf16_distance():
    """--npu-like's absorbed form (the runtime's) against the reference's HF form: equal up to the bf16 roundings
    of the GEMV inputs."""
    R = reference()
    rng = np.random.default_rng(13)
    nh, nope, rot, vd, kvl, ql, hid, T = 4, 16, 8, 12, 24, 32, 32, 7
    p = "blk.0."
    W = {p + "attn_q_a.weight": rng.standard_normal((ql, hid)).astype(np.float32) * 0.3,
         p + "attn_q_a_norm.weight": (1 + 0.1 * rng.standard_normal(ql)).astype(np.float32),
         p + "attn_q_b.weight": rng.standard_normal((nh * (nope + rot), ql)).astype(np.float32) * 0.3,
         p + "attn_kv_a_mqa.weight": rng.standard_normal((kvl + rot, hid)).astype(np.float32) * 0.3,
         p + "attn_kv_a_norm.weight": (1 + 0.1 * rng.standard_normal(kvl)).astype(np.float32),
         p + "attn_k_b.weight": rng.standard_normal((nh, kvl, nope)).astype(np.float32) * 0.3,
         p + "attn_v_b.weight": rng.standard_normal((nh, vd, kvl)).astype(np.float32) * 0.3,
         p + "attn_output.weight": rng.standard_normal((hid, nh * vd)).astype(np.float32) * 0.3}
    inv = R.yarn_inv_freq(rot, 10000.0, 64.0, 4096.0, 32.0, 1.0)
    cfg = {"heads": nh, "nope": nope, "rope": rot, "vdim": vd, "kv_lora": kvl, "eps": 1e-6, "inv": inv, "scale": 0.17}
    h = rng.standard_normal((T, hid))
    a = R.mla(_G(W), p, h, np.arange(T), cfg)
    b = R.mla_absorbed(_G(W), p, h, np.arange(T), cfg)
    rel = np.linalg.norm(a - b) / np.linalg.norm(a)
    assert 0 < rel < 2e-2
    assert R.bf(np.array([1.0 + 2 ** -9, 1.0 + 3 * 2 ** -9]))[0] == 1.0      # ties to even
