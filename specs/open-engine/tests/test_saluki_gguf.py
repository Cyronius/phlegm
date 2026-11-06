# Traces: OPEN-FAMILY-SALUKI (canonical spec: specs/open-engine/spec.md)
"""The Qwen3.5 dense all-q8 GGUF view (Underdog Saluki 27B, an i-quant GGUF): every projection a
q35_q8 op, the block route's weights q35_mat's exact hi / lo split, and the spec derived from a
GGUF-only model directory by its projections' types."""
from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from recipes import qwen35 as Q
from recipes.manifest import manifest
from recipes.spec import ModelSpec

SPECS = Path(Q.__file__).resolve().parent / "specs"


def _spec(quant):
    d = json.loads((SPECS / "qwen35-27b.json").read_text())
    d["quant"] = quant
    return ModelSpec(**d)


ALL_Q8 = {"attn": "q8", "linear": "q8", "linear_out": "q8", "ffn": "q8"}


@pytest.fixture(scope="module")
def q8_manifest():
    return manifest(_spec(ALL_Q8))      # gemv_q8 K 5120 / 6144 / 8192 / 9216 are catalogued


def test_all_q8_view_translates_every_op(q8_manifest):
    m = q8_manifest
    g = m["gguf"]
    assert g["kernels"] == m["kernels"] and g["layout"] == m["layout"] and g["spec_hash"] == m["spec_hash"]
    for name, lt in g["layer_types"].items():
        base = m["layer_types"][name]
        for key in ("pool", "consts"):
            ops, bops = lt["pack"][key], base["pack"][key]
            assert [o["dst"] for o in ops] == [o["dst"] for o in bops]
            assert all(o["op"] == "put" or o["op"].startswith("q35_") for o in ops)
            assert all(o["tensor"].startswith("blk.{l}.") for o in ops)
            for o, b in zip(ops, bops):
                if b["op"] == "q8_perm":
                    assert o["op"] == "q35_q8" and (o["nch"], o["in_dim"]) == (b["nch"], b["in_dim"])
        weights = {**lt["gemm_block"]["weights"], **lt["gemm_block"].get("ffn_weights", {})}
        assert weights
        for w in weights.values():
            assert w["from"] == "pack"
            assert all(o["op"] == "q35_mat" and o["split"] in ("hi", "lo") for o in w["pack"])
    assert g["pack"]["lm_head"]["ops"][0]["op"] == "q35_lmhead"
    assert g["pack"]["embed"]["tensor"] == "token_embd.weight"


def test_maps_undo_the_conversion(q8_manifest):
    g = q8_manifest["gguf"]
    full = g["layer_types"]["full_attention"]["pack"]["pool"]
    q = [o for o in full if o["tensor"].endswith("attn_q.weight")]
    assert [(o["period"], o["take"], o["offset"]) for o in q] == [(512, 256, 0), (512, 256, 256)]
    lin = g["layer_types"]["linear_attention"]["pack"]
    qkv = next(o for o in lin["pool"] if o["tensor"].endswith("attn_qkv.weight"))
    assert (qkv["kheads"], qkv["vperk"], qkv["vdim"], qkv["v0"]) == (16, 3, 128, 4096)
    out = next(o for o in lin["consts"] if o["tensor"].endswith("ssm_out.weight"))
    assert out["op"] == "q35_q8" and out["vaxis"] == 1
    ab = [o for o in lin["consts"] if o["op"] == "q35_t"]
    assert [o["dst_rows"] for o in ab] == [64, 64] and all(o["rows"] == 48 for o in ab)


def test_q4_1_spec_keeps_its_twin():
    m = manifest(_spec("q4_1"))
    ops = {o["op"] for lt in m["gguf"]["layer_types"].values() for o in lt["pack"]["pool"]}
    assert "q35_q8" not in ops and "q35_mat" not in ops


def _gguf(path: Path, types: dict[str, int]):
    """A header-only GGUF v3: no metadata, the named 2-D tensors at offset 0."""
    b = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(types), 0))
    for name, t in types.items():
        n = name.encode()
        b += struct.pack("<Q", len(n)) + n + struct.pack("<I", 2) + struct.pack("<QQ", 256, 256)
        b += struct.pack("<IQ", t, 0)
    path.write_bytes(bytes(b))


def test_quant_for_gguf(tmp_path):
    spec = _spec("q4_1")
    p = tmp_path / "model.gguf"
    _gguf(p, {"blk.0.ffn_up.weight": 16, "blk.0.attn_qkv.weight": 2, "blk.0.ssm_alpha.weight": 8})
    assert Q.quant_for_gguf(spec, p) == {r: "q8" for r in sorted(Q.Q8_ROLES)}
    _gguf(p, {"blk.0.ffn_up.weight": 2, "blk.0.attn_qkv.weight": 12, "blk.0.ssm_alpha.weight": 8})
    assert Q.quant_for_gguf(spec, p) is None
