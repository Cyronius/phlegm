# Traces: OPEN-PACK-PLAN (canonical spec: specs/open-engine/spec.md)
"""The plan-driven packer (recipes/pack.py over qwen36moe.pack_plan) reproduces
the hand-written packers byte for byte -- on a synthetic container whose
tensors have the 27B's shapes and random bytes (no model file needed)."""
from __future__ import annotations

import numpy as np
import pytest

import legacy_pools as LEG
from recipes import pack
from recipes import qwen36moe as Q
from recipes.load import default_spec
from recipes.spec import FULL, LINEAR

CH = 5120
# tensor name -> bytes, as the .q4nx container stores them (raster-order q4_1 chunks; bf16 / f32 small weights)
LAYER_TENSORS = {
    "mlp.up_exps_proj.weight": 256 * 4 * 163840, "mlp.gate_exps_proj.weight": 256 * 4 * 163840,
    "mlp.down_exps_proj.weight": 256 * 655360,
    "mlp.share_up_exps_proj.weight": 655360, "mlp.share_gate_exps_proj.weight": 655360,
    "mlp.share_down_exps_proj.weight": 655360,
    "input_layernorm.weight": 4096, "post_attention_layernorm.weight": 4096,
    "shared_expert_gate.weight": 4096, "moe_router.weight": 1048576,
}
LINEAR_TENSORS = {
    "linear_attn.qkv_proj.weight": 2048 * CH, "self_attn.gate_proj.weight": 1024 * CH,
    "linear_attn.ssm_conv1d.weight": 65536, "linear_attn.ssm_norm.weight": 256,
    "linear_attn.ssm_a": 128, "linear_attn.ssm_dt.bias": 128,
    "linear_attn.ssm_alpha_proj.weight": 131072, "linear_attn.ssm_beta_proj.weight": 131072,
    "linear_attn.ssm_out_proj.weight": 1024 * CH,
}
FULL_TENSORS = {
    "self_attn.q_proj.weight": 2048 * CH, "self_attn.k_proj.weight": 128 * CH, "self_attn.v_proj.weight": 128 * CH,
    "self_attn.o_proj.weight": 1024 * CH, "self_attn.q_norm.weight": 512, "self_attn.k_norm.weight": 512,
}


class FakeContainer:
    """raw(name) -> random bytes of the tensor's size, deterministic per name."""

    def __init__(self, layers: dict[int, str], lm_head_chunks: int = 64):
        self.sizes = {"lm_head.weight": lm_head_chunks * 8704, "model.norm.weight": 4096}
        for l, kind in layers.items():
            for k, v in {**LAYER_TENSORS, **(FULL_TENSORS if kind == FULL else LINEAR_TENSORS)}.items():
                self.sizes[f"model.layer.{l}.{k}"] = v
        self.cache: dict[str, np.ndarray] = {}

    def raw(self, name: str):
        if name not in self.cache:
            seed = abs(hash(name)) % (2 ** 32)
            self.cache[name] = np.random.default_rng(seed).integers(0, 256, self.sizes[name], dtype=np.uint8)
        return self.cache[name]


@pytest.fixture(scope="module")
def m():
    return FakeContainer({0: LINEAR, 1: FULL})


@pytest.fixture(scope="module")
def plan():
    return Q.pack_plan(default_spec())


@pytest.mark.parametrize("layer,kind", [(0, LINEAR), (1, FULL)])
def test_layer_pool_matches_the_legacy_packer(m, plan, layer, kind):
    ours = pack.build_layer_pool(plan, kind, m, layer)
    ref = LEG.build_layer_pool(m, layer, kind == FULL)
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref)


@pytest.mark.parametrize("layer,kind", [(0, LINEAR), (1, FULL)])
def test_consts_match_the_legacy_layer_consts(m, plan, layer, kind):
    L = Q.layout(default_spec())
    nbytes = L.CA_BYTES if kind == FULL else L.C_BYTES
    ours = pack.build_consts(plan, kind, m, layer, nbytes)
    ref = LEG.layer_consts(m, layer, kind == FULL)
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref)


def test_lm_head_pool_matches(m, plan):
    ours = pack.build_lmhead_pool(plan, m)
    ref = LEG.build_lmhead_pool(m)
    assert ours.shape == ref.shape and np.array_equal(ours, ref)


def test_ptab_matches():
    spec = default_spec()
    assert np.array_equal(pack.ptab(300, spec.rotary_dim, spec.rope_theta), LEG.ptab(300))


def test_a_small_weight_that_overflows_its_slot_is_refused(plan):
    class Big(FakeContainer):
        def raw(self, name):
            if name.endswith("input_layernorm.weight"):
                return np.zeros(4097, np.uint8)
            return super().raw(name)

    with pytest.raises(ValueError, match="does not fit its 4096 B slot"):
        pack.build_consts(plan, LINEAR, Big({0: LINEAR}), 0, Q.layout(default_spec()).C_BYTES)
