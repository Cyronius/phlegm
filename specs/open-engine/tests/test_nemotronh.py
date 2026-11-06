# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-NEMOTRONH
# (canonical spec: specs/open-engine/spec.md)
"""NVIDIA Nemotron-H with routed experts (Nemotron 3.5 Lightning 30B-A3B): single-mixer pre-norm blocks -- Mamba-2,
GQA attention without positional encoding, or 128 routed non-gated relu^2 experts (sigmoid router, selection-only
bias, top 6, renormalised x 2.5) with a shared expert -- and an untied head.

The derivation, the recipe's emission and the reference's Mamba arithmetic run here -- pure functions over a config
dict and synthetic arrays, no model and no hardware. The fixture is nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16's
config.json @ a9904d24 (fetched 2026-11-05), written out.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from recipes import families
from recipes.spec import FULL, MAMBA, MOE_BLOCK, ModelSpec, SpecError

ROOT = Path(__file__).resolve().parents[3]
PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
BLOCK = {"M": "mamba", "E": "moe", "*": "attention"}
HF_NEMOTRONH = {
    "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "attention_bias": False,
    "bos_token_id": 1, "chunk_size": 128, "conv_kernel": 4, "dtype": "bfloat16", "eos_token_id": 2, "expand": 2,
    "head_dim": 128, "hidden_size": 2688, "intermediate_size": 1856, "layer_norm_epsilon": 1e-05,
    "mamba_head_dim": 64, "mamba_hidden_act": "silu", "mamba_num_heads": 64, "mamba_proj_bias": False,
    "mamba_ssm_cache_dtype": "float32", "max_position_embeddings": 262144, "mlp_bias": False,
    "mlp_hidden_act": "relu2", "moe_intermediate_size": 1856, "moe_latent_size": None,
    "moe_shared_expert_intermediate_size": 3712, "moe_shared_expert_overlap": True, "n_group": 1, "n_groups": 8,
    "n_routed_experts": 128, "n_shared_experts": 1, "norm_eps": 1e-05, "norm_topk_prob": True,
    "num_attention_heads": 32, "num_experts_per_tok": 6, "num_hidden_layers": 52, "num_key_value_heads": 2,
    "num_logits_to_keep": 1, "num_nextn_predict_layers": 1, "pad_token_id": 0, "partial_rotary_factor": 1.0,
    "rescale_prenorm_residual": True, "residual_in_fp32": False, "rope_theta": 10000, "routed_scaling_factor": 2.5,
    "sliding_window": None, "ssm_state_size": 128, "tie_word_embeddings": False, "time_step_floor": 0.0001,
    "time_step_max": 0.1, "time_step_min": 0.001, "topk_group": 1, "use_bias": False, "use_cache": True,
    "use_conv_bias": True, "use_mamba_kernels": True, "vocab_size": 131072,
    "layers_block_type": [BLOCK[c] for c in PATTERN], "mtp_layers_block_type": ["attention", "moe"],
}


def spec(**over):
    cfg = copy.deepcopy(HF_NEMOTRONH)
    cfg.update(over)
    return ModelSpec.from_hf_config(cfg)


# ---- the derivation


def test_the_spec_is_its_own_family_with_the_published_geometry():
    s = spec()
    assert s.family == "nemotronh"
    assert (s.hidden, s.num_layers, s.vocab) == (2688, 52, 131072)
    assert (s.num_heads, s.num_kv_heads, s.head_dim, s.rotary_dim) == (32, 2, 128, 0)
    assert (s.lin_key_heads, s.lin_value_heads, s.lin_key_dim, s.lin_value_dim, s.conv_kernel) == (8, 64, 128, 64, 4)
    assert (s.num_experts, s.experts_per_tok, s.moe_intermediate, s.shared_expert_intermediate) == (128, 6, 1856, 3712)
    assert s.activation == "relu2" and not s.qk_norm and not s.attn_gate and s.norm_eps == 1e-5
    assert s.extra["router_scale"] == 2.5
    assert s.layer_types.count(MAMBA) == 23 and s.layer_types.count(MOE_BLOCK) == 23 and s.layer_types.count(FULL) == 6
    assert [i for i, t in enumerate(s.layer_types) if t == FULL] == [5, 12, 19, 26, 33, 42]


@pytest.mark.parametrize("over", [
    {"moe_latent_size": 1024}, {"n_group": 8}, {"topk_group": 4}, {"mlp_hidden_act": "silu"},
    {"tie_word_embeddings": True}, {"attention_bias": True}, {"norm_topk_prob": False},
    {"layers_block_type": ["mlp"] + [BLOCK[c] for c in PATTERN[1:]]},
])
def test_what_the_family_does_not_implement_is_refused(over):
    with pytest.raises(SpecError):
        spec(**over)


def test_the_layer_count_must_match_the_block_list():
    with pytest.raises(SpecError):
        spec(num_hidden_layers=51)


# ---- the recipe


def test_the_layout_pads_k_and_never_the_output_rows():
    from recipes import nemotronh as N
    s = spec()
    G, L = N.geometry(s), N.layout(s)
    assert (G.HIDP, G.FFP, G.OUT_K, G.IN_N, G.QKV_N) == (2816, 2048, 4096, 10304, 4608)
    assert (G.SLOTS, G.POOL_EXPERTS) == (8, 130)
    # the uneven band split covers every output row once
    for n in (G.IN_N, G.QKV_N, G.HID, G.FF):
        c = N.split_bands(n // 64)
        assert sum(c) * 64 == n and max(c) - min(c) <= 1
    # the routing weights sit past xm's padded K, inside its last element
    assert L.AD_XM + G.HIDP * 2 <= L.AD_W < L.AD_XM + L.XM_ELEMS * 4096
    assert L.POOL_DOWN == G.POOL_EXPERTS * L.UPS and L.POOL_MOE_BYTES >= L.POOL_DOWN + G.POOL_EXPERTS * L.DNS
    assert L.POOL_MOE_BYTES < 1 << 32 and L.POOL_BYTES == max(L.POOL_MIX_BYTES, L.POOL_MOE_BYTES)
    assert L.STATE_BYTES == 3 * 6144 * 4 + 64 * 64 * 128 * 4 and L.KV_ROW == 2 * 256 * 4


def test_the_pack_plan_reads_every_projection_once_and_the_shared_expert_as_two_slots():
    from recipes import nemotronh as N
    s = spec()
    G, L = N.geometry(s), N.layout(s)
    plan = N.pack_plan(s)
    lt = plan["layer_types"]
    assert set(lt) == {MAMBA, FULL, MOE_BLOCK}
    m = {o["tensor"].split(".", 2)[2]: o for o in lt[MAMBA]["pool"]}
    assert m["ssm_in.weight"]["rows"] == G.IN_N and m["ssm_in.weight"]["in_dim"] == 2816
    assert m["ssm_out.weight"]["dst"] == L.POOL_OUT and m["ssm_out.weight"]["src_dim"] == 4096
    a = lt[FULL]["pool"]
    bb = N.band_bytes(2816)
    assert [o["dst"] for o in a[:3]] == [0, 64 * bb, 68 * bb] and a[3]["dst"] == L.POOL_OUT
    moe = lt[MOE_BLOCK]["pool"]
    sh = [o for o in moe if "shexp" in o["tensor"]]
    assert [(o["row0"], o["col0"]) for o in sh] == [(0, 0), (0, 0), (1856, 0), (0, 1856)]
    assert sh[2]["dst"] == 129 * L.UPS and sh[3]["dst"] == L.POOL_DOWN + 129 * L.DNS
    for o in moe + lt[MAMBA]["pool"] + a:
        assert o["op"] == "nh_q40" and o["nch"] % (o["in_dim"] // 128) == 0
    assert plan["lm_head"]["ops"][0]["in_dim"] == 2816 and plan["chunk_bytes"] == 4608


def test_the_programs_name_their_kernels_and_the_manifest_nests_its_gguf_view():
    from recipes import manifest as M
    s = spec()
    m = M.manifest(s, 4096, "k")
    assert m["family"] == "nemotronh" and "gguf" in m and m["gguf"]["layer_types"] == m["layer_types"]
    ops = {lt: [st["op"] for st in v["program"]] for lt, v in m["layer_types"].items()}
    assert ops[MAMBA] == ["nh_pre", "run", "nh_ssm", "run", "nh_add"]
    assert ops[FULL] == ["nh_pre", "run", "nh_attn", "run", "nh_add"]
    assert ops[MOE_BLOCK] == ["nh_route", "run", "nh_add"]
    for v in m["layer_types"].values():
        for st in v["program"]:
            assert st["kernel"] in m["kernels"]
    assert m["kernels"]["nh_moe"]["patch"] == "nhroute"
    assert [st["op"] for st in m["tail"]] == ["nh_final", "run"]
    assert m["layer_types"][MAMBA]["buffers"]["pool"] < m["layer_types"][MOE_BLOCK]["buffers"]["pool"]
    assert set(m["builds"]) == {"nh_in", "nh_qkv", "nh_out", "nh_moe", "lm_head_q8"}


# ---- the reference's Mamba-2 step against a direct transcription of the published recurrence


def test_the_reference_mamba_matches_the_selective_scan_and_carries_its_state():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    import nemotronh_gguf_reference as R
    rng = np.random.default_rng(3)
    cfg = {"mamba_num_heads": 4, "mamba_head_dim": 8, "n_groups": 2, "ssm_state_size": 16, "conv_kernel": 4}
    nh, hd, ng, ds, kc = 4, 8, 2, 16, 4
    di, cdim, hid, T = nh * hd, nh * hd + 2 * ng * ds, 24, 7
    W = {"blk.0.ssm_in.weight": rng.standard_normal((di + cdim + nh, hid)).astype(np.float32) * 0.2,
         "blk.0.ssm_out.weight": rng.standard_normal((hid, di)).astype(np.float32) * 0.2,
         "blk.0.ssm_conv1d.weight": rng.standard_normal((cdim, kc)).astype(np.float32) * 0.5,
         "blk.0.ssm_conv1d.bias": rng.standard_normal(cdim).astype(np.float32) * 0.1,
         "blk.0.ssm_a": -np.exp(rng.standard_normal((nh, 1))).astype(np.float32),
         "blk.0.ssm_d": rng.standard_normal((nh, 1)).astype(np.float32),
         "blk.0.ssm_dt.bias": rng.standard_normal(nh).astype(np.float32),
         "blk.0.ssm_norm.weight": (1 + 0.1 * rng.standard_normal((ng, di // ng))).astype(np.float32)}

    class G:
        def array(self, n, narrow=False):
            return W[n]
    h = rng.standard_normal((T, hid)).astype(np.float32)
    whole = R.mamba2(h, G(), "blk.0.", cfg, G().array, lambda x: x, 1e-5, 0.0)
    # the same sequence in two pieces through the carried state
    st = {"conv": np.zeros((kc - 1, cdim), np.float32), "ssm": np.zeros((nh, hd, ds))}
    parts = np.concatenate([R.mamba2(h[:3], G(), "blk.0.", cfg, G().array, lambda x: x, 1e-5, 0.0, state=st),
                            R.mamba2(h[3:], G(), "blk.0.", cfg, G().array, lambda x: x, 1e-5, 0.0, state=st)])
    assert np.allclose(whole, parts, rtol=1e-5, atol=1e-6)
    # a direct transcription: conv over zero history, then h_t = exp(dt A) h + dt x B^T, y = h C + D x
    zxbcdt = h.astype(np.float64) @ W["blk.0.ssm_in.weight"].T
    z, xbc, dt = zxbcdt[:, :di], zxbcdt[:, di:di + cdim], zxbcdt[:, di + cdim:]
    pad = np.vstack([np.zeros((kc - 1, cdim)), xbc])
    conv = np.stack([(pad[t:t + kc] * W["blk.0.ssm_conv1d.weight"].T).sum(0) for t in range(T)]) + W["blk.0.ssm_conv1d.bias"]
    xc = conv / (1 + np.exp(-conv))
    S = np.zeros((nh, hd, ds))
    ys = []
    for t in range(T):
        x = xc[t, :di].reshape(nh, hd)
        B = xc[t, di:di + ng * ds].reshape(ng, ds)
        C = xc[t, di + ng * ds:].reshape(ng, ds)
        y = np.empty((nh, hd))
        for hh in range(nh):
            d = np.log1p(np.exp(dt[t, hh] + W["blk.0.ssm_dt.bias"][hh]))
            g = hh // (nh // ng)
            S[hh] = S[hh] * np.exp(d * W["blk.0.ssm_a"][hh, 0]) + d * np.outer(x[hh], B[g])
            y[hh] = S[hh] @ C[g] + W["blk.0.ssm_d"][hh, 0] * x[hh]
        ys.append(y.reshape(-1))
    y = np.array(ys) * (z / (1 + np.exp(-z)))
    yg = y.reshape(T, ng, -1)
    yn = (yg / np.sqrt((yg ** 2).mean(-1, keepdims=True) + 1e-5)).reshape(T, di) * W["blk.0.ssm_norm.weight"].reshape(-1)
    assert np.allclose(whole, yn @ W["blk.0.ssm_out.weight"].T, rtol=1e-4, atol=1e-5)


def test_the_family_is_registered():
    assert families.family_module("nemotronh").__name__.endswith("nemotronh")
    assert "nemotronh" in families.FAMILIES
