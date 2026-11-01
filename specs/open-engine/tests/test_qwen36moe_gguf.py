# Traces: OPEN-FAMILY-QWEN36MOE-GGUF (canonical spec: specs/open-engine/spec.md)
"""The Qwen3.5 / 3.6 MoE GGUF-direct view: the q4_1 manifest, op for op, read from a llama.cpp
qwen35moe GGUF by the runtime's q35_* ops (upstream-patches/qwen35moe-gguf-runtime.patch)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from recipes import qwen36moe as Q
from recipes.load import default_spec
from recipes.manifest import manifest


def _view():
    m = manifest(default_spec())
    return m, m["gguf"]


def test_view_is_the_q4_1_manifest_with_gguf_pack_ops():
    m, g = _view()
    assert g["layout"] == m["layout"] and g["kernels"] == m["kernels"] and g["contexts"] == m["contexts"]
    assert g["spec_hash"] == m["spec_hash"] and g["pack"]["chunk_bytes"] == 5120 == m["pack"]["chunk_bytes"]
    for name, lt in g["layer_types"].items():
        base = m["layer_types"][name]
        assert {k: v for k, v in lt.items() if k != "pack"} == {k: v for k, v in base.items() if k != "pack"}
        for key in ("pool", "consts"):
            ops, bops = lt["pack"][key], base["pack"][key]
            # same order and offsets: the block route's weights name these ops by index
            assert [o["dst"] for o in ops] == [o["dst"] for o in bops]
            assert all(o["op"] in ("put",) or o["op"].startswith("q35_") for o in ops)
            assert all("model.layer" not in o.get("tensor", "") + o.get("up", "") for o in ops)
            for o, b in zip(ops, bops):
                if "nch" in b:
                    assert o["nch"] == b["nch"] and o["in_dim"] == b["in_dim"]
                if "cap" in b and o["op"] in ("put", "q35_vec"):
                    assert o["cap"] == b["cap"]
        gb = lt.get("gemm_block", {})
        for wkey in ("weights", "shared_weights"):
            for w in gb.get(wkey, {}).values():
                src = lt["pack"]["pool" if w["from"] == "pool" else "consts"]
                assert all(src[i]["op"] == "q35_mat" for i in w["ops"])
    assert g["pack"]["embed"]["tensor"] == "token_embd.weight"
    assert g["pack"]["norm"]["tensor"] == "output_norm.weight"
    assert g["pack"]["lm_head"]["ops"][0]["op"] == "q35_lmhead"
    assert "gguf" not in g


def test_q_gate_split_and_value_head_untile_fields():
    _, g = _view()
    s = default_spec()
    full = {o["tensor"].split(".")[-2] + str(o.get("offset", "")): o for o in g["layer_types"]["full_attention"]["pack"]["pool"] if "tensor" in o}
    assert full["attn_q0"]["period"] == 2 * s.head_dim and full["attn_q0"]["take"] == s.head_dim
    assert full["attn_q" + str(s.head_dim)]["offset"] == s.head_dim
    lin = {o["tensor"].split(".")[-2]: o for o in g["layer_types"]["linear_attention"]["pack"]["pool"] + g["layer_types"]["linear_attention"]["pack"]["consts"] if "tensor" in o}
    assert lin["attn_qkv"]["v0"] == 2 * s.lin_key_heads * s.lin_key_dim and lin["attn_qkv"]["vaxis"] == 0
    assert lin["ssm_out"]["vaxis"] == 1 and lin["ssm_out"]["v0"] == 0
    assert lin["ssm_alpha"]["vdim"] == 1 and lin["ssm_alpha"]["rows"] == s.lin_value_heads


def _q35_map(n, kheads, vperk, vdim, v0):
    """The runtime's Q35Map value-head part (pools.cpp)."""
    i = np.arange(n)
    rel = i - v0
    h = rel // vdim
    src = v0 + ((h % vperk) * kheads + h // vperk) * vdim + rel % vdim
    return np.where(i >= v0, src, i)


def test_untile_inverts_llama_cpp_v_head_reorder():
    # llama.cpp conversion/qwen.py _reorder_v_heads: grouped [K][vperk][d] -> tiled [vperk][K][d]
    kh, vperk, d = 16, 2, 128
    hf = np.arange(kh * vperk * d)
    gguf = hf.reshape(kh, vperk, d).transpose(1, 0, 2).reshape(-1)
    assert np.array_equal(gguf[_q35_map(hf.size, kh, vperk, d, 0)], hf)
    # with a q | k prefix ahead of the V rows (attn_qkv, the conv channels)
    pre = 2 * kh * 128
    rows = np.arange(pre + hf.size)
    g2 = np.concatenate([rows[:pre], pre + gguf])
    assert np.array_equal(g2[_q35_map(rows.size, kh, vperk, d, pre)], rows)


def test_no_view_for_formats_the_q35_ops_do_not_write():
    s = default_spec()
    assert Q.gguf_view(replace(s, quant={**s.quant_map, "attn": "q8"}), manifest(s)) is None
    assert Q.gguf_view(replace(s, quant="q4_1_f32"), manifest(s)) is None


def test_equal_head_counts_untile_nothing():
    s = default_spec()
    assert Q.gguf_vtile(replace(s, lin_value_heads=s.lin_key_heads), 0, 128) == {}
    with pytest.raises(Q.OpRangeError):
        Q.gguf_vtile(replace(s, lin_value_heads=24), 0, 128)
