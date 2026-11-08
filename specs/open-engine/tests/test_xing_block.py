# Traces: OPEN-FAMILY-XING
# (canonical spec: specs/open-engine/spec.md)
"""Xing 4.0's block prefill route (OPEN-FAMILY-XING "Block prefill"): the recipe's `gemm_block` (kind xing) -- q4_0
GEMMs that read the layer pool IN PLACE at their steps' pool offsets, and the token-batched xmb streams on the
routed experts -- plus the switch that drops it (XING_BLOCK=0 gives the sequential set's manifest), the
catalogue point, and the host layouts the runtime's block code shares with the kernels (the xmb y scatter, the
GEMM's gate | up band interleave). Pure functions over a config dict; no model, no hardware.
"""
from __future__ import annotations

import copy
import importlib
import os

import numpy as np
import pytest

from recipes import catalogue, xing as XR
from recipes.catalogue import OpRangeError
from recipes.spec import DENSE, MOE_BLOCK, ModelSpec

from test_xing import HF_XING


def spec(**over):
    cfg = copy.deepcopy(HF_XING)
    cfg.update(over)
    return ModelSpec.from_hf_config(cfg)


@pytest.fixture
def block_off(monkeypatch):
    monkeypatch.setenv("XING_BLOCK", "0")


def test_the_route_is_on_by_default_with_one_gemm_block_per_layer_kind():
    s = spec()
    prog = XR.programs(s)
    for kind in (DENSE, MOE_BLOCK):
        gb = prog["layer_types"][kind]["gemm_block"]
        assert gb["kind"] == "xing" and gb["t"] == 256 and gb["eps"] == 1e-6
        assert [st["role"] for st in gb["program"]] == ["qkva", "qb", "o", "ffn_ug", "ffn_dn"]
        assert all(st["op"] == "run" and st["args"][0] == "pool" for st in gb["program"])
    assert "xmb" in prog["layer_types"][MOE_BLOCK]["gemm_block"]
    assert "xmb" not in prog["layer_types"][DENSE]["gemm_block"]
    assert prog["contexts"]["gemm"] == "gemm_n1536_k3584/final.xclbin"
    assert prog["contexts"]["xmb"] == "xmb_s64/final.xclbin"


def test_every_gemm_reads_whole_256_row_blocks_inside_its_own_layer_pool():
    s = spec()
    G, L = XR.geometry(s), XR.layout(s)
    for kind, pool in ((DENSE, L.POOL_DENSE_BYTES), (MOE_BLOCK, L.POOL_MOE_BYTES)):
        for st in XR.block_steps(s, kind):
            N, K = (int(v) for v in st["kernel"][len("gemm_n"):].split("_k"))
            assert N % 256 == 0 and K % 256 == 0 and N >= st["rows"] > N - 256
            assert st["args"][1:] == [f"gemm_x_k{K}", f"gemm_y_n{N}"]
            end = st["pool_off"] + (st.get("count", 1) - 1) * st.get("stride", 0) + N // 64 * XR.band_bytes(K)
            assert end <= pool, (kind, st["role"])


def test_the_steps_point_at_the_sequential_programs_weights():
    s = spec()
    G, L = XR.geometry(s), XR.layout(s)
    moe = {st["role"]: st for st in XR.block_steps(s, MOE_BLOCK)}
    dense = {st["role"]: st for st in XR.block_steps(s, DENSE)}
    for d in (moe, dense):
        assert (d["qkva"]["pool_off"], d["qkva"]["rows"]) == (L.POOL_QKVA, G.QKVA_N)      # 1344 rows, run as 1536
        assert d["qkva"]["kernel"] == "gemm_n1536_k3584"
        assert (d["qb"]["pool_off"], d["qb"]["rows"]) == (L.POOL_QB, G.QB_N)
        assert (d["o"]["pool_off"], d["o"]["rows"]) == (L.POOL_O, G.HID)
    # the shared expert is pool expert EXPERTS of the moe pool law
    assert moe["ffn_ug"]["pool_off"] == L.POOL_EXP + G.EXPERTS * L.UPS and moe["ffn_ug"]["rows"] == 2 * G.FF
    assert moe["ffn_dn"]["pool_off"] == L.POOL_DOWN_M + G.EXPERTS * L.DNS and moe["ffn_dn"]["rows"] == G.HID
    # the dense MLP: all nine slices' gate | up in one GEMM, each slice's down a dispatch DNS apart
    assert (dense["ffn_ug"]["pool_off"], dense["ffn_ug"]["rows"]) == (L.POOL_EXP, 2 * G.FF * G.DSLOTS)
    assert (dense["ffn_dn"]["pool_off"], dense["ffn_dn"]["count"], dense["ffn_dn"]["stride"]) == \
        (L.POOL_DOWN_D, G.DSLOTS, L.DNS)


def test_the_xmb_streams_read_the_moe_pool_law_in_place():
    s = spec()
    G, L = XR.geometry(s), XR.layout(s)
    b = XR.builds(s)
    slots = XR.moe_batch_slots(s)
    assert slots == [64, 32, 16, 8]
    bb_h, bb_f = XR.band_bytes(G.HID), XR.band_bytes(G.FF)
    # mbl's / xmb's per-slot strides (xmb.py EXP_UG and UP_BYTES) are the recipe's UPS and DNS
    assert 2 * (G.FF // 64) * bb_h == L.UPS and (G.HID // 64) * bb_f == L.DNS == (G.FF // 64) * bb_h
    for sl in slots:
        e = b[f"xmb_s{sl}"]["env"]
        assert b[f"xmb_s{sl}"]["design"] == "xing_batch/xmb.py"
        assert (int(e["XMB_SLOTS"]), int(e["XMB_NT"]), int(e["XMB_HID"]), int(e["XMB_FF"])) == (sl, 8, G.HID, G.FF)
        assert (int(e["XMB_POOL_EXP"]), int(e["XMB_POOL_DOWN"]), int(e["XMB_POOL_BYTES"])) == \
            (L.POOL_EXP, L.POOL_DOWN_M, L.POOL_MOE_BYTES)
    gb = XR.programs(s)["layer_types"][MOE_BLOCK]["gemm_block"]
    x = gb["xing"]
    assert (x["pool_exp"], x["ups"], x["pool_down"], x["dns"], x["experts"], x["topk"]) == \
        (L.POOL_EXP, L.UPS, L.POOL_DOWN_M, L.DNS, 64, 4)
    assert gb["xmb"] == {"kernels": {str(sl): f"xmb_s{sl}" for sl in slots}, "args": ["pool", "xmb_x", "xmb_h", "xmb_y"],
                         "nt": 8}
    k = XR.programs(s)["kernels"]
    assert all(k[f"xmb_s{sl}"]["patch"] == "xgbatch" and k[f"xmb_s{sl}"]["context"] == "xmb" for sl in slots)


def test_the_xgbatch_offsets_name_their_slot_and_point_at_the_routed_expert():
    """The runtime's `xgbatch` table (xing-runtime.patch xg_experts): a compiled fill at POOL_EXP + s UPS + intra is
    slot s's gate | up, one at POOL_DOWN + s DNS + intra its down; re-pointed to expert e it reads e's bytes."""
    s = spec()
    L = XR.layout(s)

    def table(off):
        if off >= L.POOL_DOWN_M:
            return (off - L.POOL_DOWN_M) // L.DNS | 0x100, (off - L.POOL_DOWN_M) % L.DNS
        return (off - L.POOL_EXP) // L.UPS, (off - L.POOL_EXP) % L.UPS

    for slot in (0, 7, 63):
        for intra in (0, 9216, L.DNS - 9216):
            sl, it = table(L.POOL_EXP + slot * L.UPS + intra)
            assert (sl, it) == (slot, intra)
            sl, it = table(L.POOL_DOWN_M + slot * L.DNS + intra)
            assert (sl, it) == (slot | 0x100, intra)
    # the last slot's last down byte stays inside the moe pool, and the shared expert's down too
    assert L.POOL_DOWN_M + 65 * L.DNS <= L.POOL_MOE_BYTES


def test_xing_block_0_gives_the_sequential_manifest(block_off):
    s = spec()
    prog = XR.programs(s)
    assert all("gemm_block" not in lt for lt in prog["layer_types"].values())
    assert set(prog["contexts"]) == {"xg_one", "lm"}
    assert not any(k.startswith(("gemm_", "xmb_")) for k in list(prog["kernels"]) + list(XR.builds(s)))
    assert not any(g.startswith(("gemm_", "xmb_")) for g in prog["globals"])
    assert XR.KERNEL_SOURCES_MOE == []


def test_the_block_designs_are_in_the_build_key_only_with_the_route(monkeypatch):
    monkeypatch.delenv("XING_BLOCK", raising=False)
    assert "designs/xing_batch/*.cc" in XR.KERNEL_SOURCES_MOE
    assert "designs/gemm_q4_prefill/*.py" in XR.KERNEL_SOURCES_MOE


def test_the_globals_hold_one_gemm_block_and_the_longest_xmb_stream():
    s = spec()
    g = XR.programs(s)["globals"]
    assert g["gemm_x_k3584"] == 3584 * 256 * 2 and g["gemm_y_n18432"] == 18432 * 256 * 4
    assert g["xmb_x"] == 64 * 3584 * 8 * 2 and g["xmb_h"] == 64 * 1024 * 8 * 2 and g["xmb_y"] == 64 * 3584 * 8 * 4
    builds = XR.builds(s)
    for name, b in builds.items():
        if name.startswith("gemm_"):
            assert b["env"]["GQP_FMT"] == "q4_0" and b["env"]["GQP_T"] == "256"
            assert b["build_dir"].endswith("_q40")


def test_the_catalogue_holds_only_the_validated_xmb_point():
    catalogue.require("xmb", hidden=3584, ff=1024, nt=8, slots_max=64)
    if os.environ.get("OPEN_KERNELS_UNVALIDATED"):
        pytest.skip("OPEN_KERNELS_UNVALIDATED lets unvalidated points through")
    with pytest.raises(OpRangeError):
        catalogue.require("xmb", hidden=3584, ff=1536, nt=8, slots_max=64)
    with pytest.raises(OpRangeError):
        catalogue.require("xmb", hidden=3584, ff=1024, nt=16, slots_max=64)


def test_the_runtime_y_scatter_is_the_kernels_y_layout():
    """xg_experts reads y[slot] at ((band / 4) * 8 + band % 4) * 256 + tt * 8 + half * 1024 + gg * 128 (+ 64 odd),
    row 64 band + 16 (2 half + gg) + 2 jj + parity: lfm2moe_batch/layout.py y_from_dev's law at NT 8."""
    hid, NT = 3584, 8
    rng = np.random.default_rng(3)
    dev = rng.standard_normal(hid * NT).astype(np.float32)
    nb = hid // 64
    d = dev.reshape(1, nb // 4, 2, 4, 2, 2, 1, 8, 8).transpose(0, 1, 3, 2, 4, 5, 6, 7, 8).reshape(1, nb, 4, 2, 1, 8, 8)
    want = d.transpose(0, 4, 5, 1, 2, 6, 3).reshape(NT, hid)
    got = np.zeros((NT, hid), np.float32)
    for tt in range(NT):
        for band in range(nb):
            el = ((band // 4) * 8 + band % 4) * 256 + tt * 8
            for half in range(2):
                for gg in range(2):
                    even = el + half * 4 * 256 + gg * 2 * 64
                    for jj in range(8):
                        got[tt, band * 64 + (2 * half + gg) * 16 + 2 * jj] = dev[even + jj]
                        got[tt, band * 64 + (2 * half + gg) * 16 + 2 * jj + 1] = dev[even + 64 + jj]
    assert np.array_equal(got, want)


def test_the_gate_up_interleave_the_host_reads_is_the_pool_band_law():
    """xg_swiglu reads a slice's gate row r at 128 (r / 64) + r % 64 and its up row at + 64: the pool stores band b of
    the gate at POOL_EXP + 2 b bb and of the up at + bb (recipes/xing.py _layer_plan band_stride 2 bb), and the GEMM
    walks the bands in pool order, 64 output rows each."""
    s = spec()
    G = XR.geometry(s)
    plan = XR._layer_plan(s, MOE_BLOCK)["pool"]
    gate = next(o for o in plan if o["tensor"].endswith("ffn_gate_shexp.weight"))
    up = next(o for o in plan if o["tensor"].endswith("ffn_up_shexp.weight"))
    bb = XR.band_bytes(G.HID)
    assert gate["band_stride"] == up["band_stride"] == 2 * bb and up["dst"] - gate["dst"] == bb
    for r in (0, 63, 64, 1023):
        band = r // 64
        gate_row = 128 * band + r % 64
        assert (gate["dst"] + band * 2 * bb - gate["dst"]) // bb == gate_row // 64
        assert (up["dst"] + band * 2 * bb - gate["dst"]) // bb == (gate_row + 64) // 64


def test_the_manifest_carries_the_route_for_both_kinds():
    from recipes.manifest import manifest
    m = manifest(spec(), 4096, key="X")
    assert m["layer_types"]["dense"]["gemm_block"]["kind"] == "xing"
    assert m["layer_types"]["moe"]["gemm_block"]["kind"] == "xing"
    assert m["gguf"]["layer_types"]["moe"]["gemm_block"] == m["layer_types"]["moe"]["gemm_block"]
    assert {"gemm_n1536_k3584", "gemm_n6144_k768", "gemm_n3584_k4096", "gemm_n2048_k3584", "gemm_n3584_k1024",
            "gemm_n18432_k3584", "xmb_s64", "xmb_s32", "xmb_s16", "xmb_s8"} <= set(m["builds"])
