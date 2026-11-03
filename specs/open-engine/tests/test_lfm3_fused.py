# Traces: OPEN-FAMILY-LFM3
"""LFM3 decode with one dispatch a layer (LFM3_FUSED=1): the mixer's program then the MLP's over the same fifos."""
from __future__ import annotations

import json

import pytest

from recipes.spec import DENSE_LOCAL, FULL, SHORT_CONV, ModelSpec
from test_lfm3 import HF_LFM3


def spec():
    return ModelSpec.from_hf_config(dict(HF_LFM3))


def test_off_is_the_unfused_set(monkeypatch):
    from recipes import lfm3 as R
    monkeypatch.delenv("LFM3_FUSED", raising=False)
    s = spec()
    assert json.dumps(R.programs(s), sort_keys=True) == json.dumps(R._programs(s), sort_keys=True)
    assert json.dumps(R.builds(s), sort_keys=True) == json.dumps(R._builds(s), sort_keys=True)
    assert "lg_mlp" in R.builds(s) and len(R.programs(s)["layer_types"][SHORT_CONV]["program"]) == 2


def test_on_is_one_dispatch_a_layer_and_the_pack_plan_is_unchanged(monkeypatch):
    from recipes import lfm3 as R
    monkeypatch.delenv("LFM3_FUSED", raising=False)
    s = spec()
    plan = json.dumps(R.pack_plan(s), sort_keys=True)
    monkeypatch.setenv("LFM3_FUSED", "1")
    p = R.programs(s)
    assert json.dumps(R.pack_plan(s), sort_keys=True) == plan
    for lt in (SHORT_CONV, FULL, DENSE_LOCAL):
        assert len(p["layer_types"][lt]["program"]) == 1
    assert p["layer_types"][SHORT_CONV]["program"][0]["kernel"] == "lg_convmlp"
    assert p["layer_types"][DENSE_LOCAL]["program"][0]["kernel"] == "lg_attmlp_local"
    assert p["kernels"]["lg_attmlp_local"]["window"] == 256 and p["kernels"]["lg_attmlp"]["window"] == 0
    assert p["kernels"]["lg_attmlp"]["insts"] == p["kernels"]["lg_attmlp_local"]["insts"] == "lg_attmlp/insts.bin"
    assert "lg_mlp" not in p["kernels"] and "lg_mlp" not in p["contexts"]
    assert set(R.builds(s)) == {"ln", "lm_head_q8", "lg_convmlp", "lg_attmlp"}
    assert R.builds(s)["lg_attmlp"]["env"] == {"LG_PART": "attmlp"}
    for k in p["kernels"].values():          # every kernel's context and build exist
        assert k["context"] in p["contexts"] and k["build"] in R.builds(s)


def test_fused_and_block_together_make_a_consistent_manifest(monkeypatch):
    from recipes import lfm3 as R
    monkeypatch.setenv("LFM3_FUSED", "1")
    monkeypatch.setenv("LFM3_BLOCK", "1")
    s = spec()
    p, b = R.programs(s), R.builds(s)
    assert all(lt["gemm_block"]["kind"] == "lfm" for lt in p["layer_types"].values())
    assert {"gemm_n6144_k2048", "lg_convmlp", "lg_attmlp"} <= set(b)
    for k in p["kernels"].values():
        assert k["build"] in b
