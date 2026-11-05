# Traces: OPEN-FAMILY-HYV3 (block prefill), OPEN-PREFILL-BATCH
# (canonical spec: specs/open-engine/spec.md)
"""Hy-Memory-30B-A3B block prefill (on by default; HYV3_BLOCK=0 drops it): the recipe's emission over the decode pool as
it is, and that LFM2.5's token-batched expert kernel (designs/lfm2moe_batch/mbl.py) reads Hy's pool law in place at FF
768 / 129 pool experts. Pure functions over the HF config fixture of test_hyv3.py -- no model, no hardware (the route
is validated on hardware: the model gate in .plans/hymemory-native.md)."""
from __future__ import annotations

import pytest

from recipes import hyv3 as H

from test_hyv3 import spec

CH = 5120


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("HYV3_BLOCK", "1")


def test_on_by_default(monkeypatch):
    monkeypatch.delenv("HYV3_BLOCK", raising=False)
    assert H.gemm_route(spec()) is not None


def test_off_leaves_the_sequential_emission_alone(monkeypatch):
    s = spec()
    monkeypatch.setenv("HYV3_BLOCK", "0")
    assert H.gemm_route(s) is None
    progs, blds, plan = H.programs(s), H.builds(s), H.pack_plan(s)
    assert all("gemm_block" not in lt for lt in progs["layer_types"].values())
    assert sorted(blds) == ["hy_att", "hy_moe", "lm_head_q8", "ln"]
    assert sorted(progs["kernels"]) == ["hy_att", "hy_moe", "lm", "ln"]
    assert not any(k.startswith(("gemm_", "mb_")) for k in progs["globals"])
    # the route adds nothing to the pool: the pack plan is the same either way
    monkeypatch.setenv("HYV3_BLOCK", "1")
    assert H.pack_plan(s) == plan
    on = H.programs(s)
    for nm, lt in progs["layer_types"].items():
        assert {k: v for k, v in on["layer_types"][nm].items() if k != "gemm_block"} == lt
    assert {k: on["kernels"][k] for k in progs["kernels"]} == progs["kernels"]


def test_the_route_is_one_lfm_att_block_a_layer_type(on):
    s = spec()
    G = H.geometry(s)
    g = H.programs(s)["layer_types"]
    assert sorted(g) == ["dense", "dense_mlp"]
    moe, dense = g["dense"]["gemm_block"], g["dense_mlp"]["gemm_block"]
    for gb in (moe, dense):
        assert (gb["kind"], gb["mixer"], gb["t"]) == ("lfm", "att", 256)
        assert [p["kernel"] for p in gb["program"]] == ["gemm_n5120_k2048", "gemm_n2048_k4096"]
        assert (gb["qw"], gb["kvw"], gb["nh"], gb["kvh"], gb["hd"], gb["rot"]) == (4096, 512, 32, 4, 128, 128)
        assert "ffn_program" not in gb and gb["moe_batch"]["y_layout"] == "rbg_half_row"
    L = H.layout(s)
    assert (moe["cd_ln"], moe["cd_ffn"], moe["cd_qnorm"], moe["cd_knorm"]) == (
        L.CD_LNW, L.CD_POSTLN, L.CD_META, L.CD_META + 2 * G.HD)
    # the router's 8 then the shared expert (pool expert 128) at weight 1: hy_route's slots; the dense layer's 9 slices
    r = moe["route"]
    assert (r["experts"], r["topk"], r["shared"], r["scale"]) == (128, 8, [128], 2.826)
    assert r["bias"] == "blk.{l}.exp_probs_b" and r["router"] == "blk.{l}.ffn_gate_inp.weight"
    mg = H.moe_geometry(s)
    assert r["topk"] + len(r["shared"]) == mg["topk"] == 9
    assert r["experts"] + len(r["shared"]) == mg["experts"] == G.POOL_EXPERTS == 129
    assert dense["route"] == {"dense": 9} and G.SLOTS == 9


def test_gemm_weights_are_the_decode_pool_runs(on):
    s = spec()
    L = H.layout(s)
    plan = H.pack_plan(s)["layer_types"]
    for nm, lt in H.programs(s)["layer_types"].items():
        gb, pool = lt["gemm_block"], plan[nm]["pool"]
        q, k, v = (pool[i] for i in gb["weights"]["gqkv3_w"]["ops"])
        (o,) = (pool[i] for i in gb["weights"]["go_w"]["ops"])
        assert [x["tensor"].split(".")[-2] for x in (q, k, v, o)] == ["attn_q", "attn_k", "attn_v", "attn_output"]
        assert (q["dst"], k["dst"], v["dst"], o["dst"]) == (L.POOL_Q, L.POOL_K, L.POOL_V, L.POOL_O)
        assert k["dst"] == q["dst"] + 4096 // 64 * H.band_bytes(2048) and v["dst"] == k["dst"] + 512 // 64 * H.band_bytes(2048)
        assert all(x["experts"] == 1 and x["stripe_bytes"] == 0 for x in (q, k, v, o))
        assert o["in_dim"] == 4096 and o["rows"] == 2048


def _mbl_taps(env: dict) -> dict:
    """designs/lfm2moe_batch/mbl.py's byte law, from its env (the design imports aie, so it is restated here)."""
    hid, ff = int(env["MBL_HID"]), int(env["MBL_FF"])
    bb_h, bb_f = 2 * (hid // 256) * CH, 2 * (ff // 256) * CH
    nb_up, nb_dn = ff // 64, hid // 64
    return {"BB_H": bb_h, "BB_F": bb_f, "NB_UP": nb_up, "NB_DN": nb_dn, "EXP_UG": 2 * nb_up * bb_h,
            "UP_BYTES": nb_dn * bb_f, "POOL_DOWN": int(env["MBL_POOL_DOWN"]), "POOL_BYTES": int(env["MBL_POOL_BYTES"]),
            "EXPERTS": int(env["MBL_EXPERTS"]), "SLOTS": int(env["MBL_SLOTS"])}


def _band_at(op: dict, e: int, b: int, band: int) -> int:
    """fl_q4's destination of band b of expert e (flint-runtime.patch: dst + e expert_bytes + b band_stride, the bands
    contiguous when band_stride is 0)."""
    return op["dst"] + e * op["expert_bytes"] + b * (op["stripe_bytes"] or band)


def test_mbl_reads_the_pool_in_place(on):
    """Every FFN the block route runs -- a routed expert, the shared expert (pool expert 128), a dense-layer slice -- sits
    where mbl's taps read pool expert e: gate band b at e EXP_UG + 2b BB_H, up band b one BB_H later, down band b at
    POOL_DOWN + e UP_BYTES + b BB_F; and nothing mbl may map reaches the attention weights."""
    s = spec()
    L, G = H.layout(s), H.geometry(s)
    b = H.moe_batch(s)["builds"]["mb_s64"]
    assert b["design"] == "lfm2moe_batch/mbl.py"
    t = _mbl_taps(b["env"])
    assert (b["env"]["MBL_HID"], b["env"]["MBL_FF"], b["env"]["MBL_EXPERTS"]) == ("2048", "768", "129")
    assert t["NB_UP"] % 4 == 0 and t["NB_DN"] % 4 == 0           # mbl: a core row owns band 4 rbg + r
    assert t["UP_BYTES"] == t["NB_UP"] * t["BB_H"] == L.UP_BYTES   # moeroute2's law: gate = down
    assert t["EXP_UG"] == L.EXPERT_UG and t["POOL_DOWN"] == L.POOL_DOWN
    assert t["POOL_DOWN"] + t["EXPERTS"] * t["UP_BYTES"] == t["POOL_BYTES"] == L.POOL_ATTN <= L.POOL_Q
    plan = H.pack_plan(s)["layer_types"]
    for nm, first in (("dense", 0), ("dense_mlp", 0)):
        ops = {o["tensor"].split(".")[-2]: o for o in plan[nm]["pool"] if "ffn" in o["tensor"]}
        groups = ([("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", 0, 128),
                   ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", 128, 1)] if nm == "dense" else
                  [("ffn_gate", "ffn_up", "ffn_down", 0, 9)])
        for gate, up, down, e0, n in groups:
            assert ops[gate]["experts"] == ops[up]["experts"] == ops[down]["experts"] == n
            for e in (0, n - 1):
                pe = e0 + e                                          # the pool expert mbl is pointed at
                for band in (0, t["NB_UP"] - 1):
                    assert _band_at(ops[gate], e, band, t["BB_H"]) == pe * t["EXP_UG"] + 2 * band * t["BB_H"]
                    assert _band_at(ops[up], e, band, t["BB_H"]) == pe * t["EXP_UG"] + (2 * band + 1) * t["BB_H"]
                for band in (0, t["NB_DN"] - 1):
                    assert _band_at(ops[down], e, band, t["BB_F"]) == t["POOL_DOWN"] + pe * t["UP_BYTES"] + band * t["BB_F"]
    assert G.POOL_EXPERTS * L.EXPERT_UG == L.POOL_DOWN


def test_moebatch_table_reads_each_slot_as_its_pool_expert(on):
    """core.cpp load_kernel(moebatch): moeroute2's table with topk = every pool expert. A gate | up fill at offset off
    < pool_down is slot off / (2 up_bytes), a down fill slot (off - pool_down) / up_bytes: slot s of an mbl stream
    is compiled as pool expert s, so both land on s for every s < 64."""
    s = spec()
    mg = H.moe_geometry(s)
    t = _mbl_taps(H.moe_batch(s)["builds"]["mb_s64"]["env"])
    for slot in range(t["SLOTS"]):
        for proj in (0, 1):
            for rbg in (0, t["NB_UP"] // 4 - 1):
                off = slot * t["EXP_UG"] + (2 * 4 * rbg + proj) * t["BB_H"]
                assert off < mg["pool_down"] and off // (2 * mg["up_bytes"]) == slot
        off = t["POOL_DOWN"] + slot * t["UP_BYTES"] + (t["NB_DN"] // 4 - 1) * 4 * t["BB_F"]
        assert mg["pool_down"] <= off < mg["share_up"] and (off - mg["pool_down"]) // mg["up_bytes"] == slot


def test_expert_ladder_and_globals(on):
    s = spec()
    mb = H.moe_batch(s)
    assert list(mb["moe_batch"]["kernels"]) == ["64", "56", "48", "40", "32", "24", "16", "8"]
    assert mb["moe_batch"]["nt"] == 8 and mb["contexts"] == {"mb": "mb_s64/final.xclbin"}
    assert all(k["patch"] == "moebatch" for k in mb["kernels"].values())
    assert mb["globals"] == {"mb_x": 64 * 2048 * 8 * 2, "mb_h": 64 * 768 * 8 * 2, "mb_y": 64 * 2048 * 8 * 4}
    # distinct from LFM2.5's mbl builds (FF 1536, 64 experts): no shared build directory
    assert {b["build_dir"] for b in mb["builds"].values()} == {
        f"lfm2moe_batch/build_s{n}_h2048_f768_e129" for n in range(8, 65, 8)}


def test_mbl_l1_budget_at_hy_width():
    # mbl.py's L1 formula (NT 8, one u / g band): stack, a double-buffered band-k-group, u / g, y, output elements, the
    # B fifo and the whole h -- smaller than LFM2.5's 62464 B, h being 768 wide
    NT, BAND, FF = 8, 2 * CH, 768
    l1 = 0x1000 + 2 * BAND + 2 * (64 * NT * 4) + 64 * NT * 4 + 2 * (32 * NT * 4) + 5 * (64 * NT * 2) + FF * NT * 2
    assert l1 == 50176 <= 64 << 10


def test_gemm_builds_and_globals(on):
    s = spec()
    p, b = H.programs(s), H.builds(s)
    assert {k for k in b if k.startswith("gemm_")} == {"gemm_n5120_k2048", "gemm_n2048_k4096"}
    assert b["gemm_n2048_k4096"]["env"] == {"GQP_N": "2048", "GQP_K": "4096", "GQP_T": "256"}
    assert p["globals"]["gemm_x_k4096"] == 4096 * 256 * 2 and p["globals"]["gemm_y_n5120"] == 5120 * 256 * 4
    assert p["contexts"]["gemm"].startswith("gemm_n") and p["contexts"]["mb"] == "mb_s64/final.xclbin"
