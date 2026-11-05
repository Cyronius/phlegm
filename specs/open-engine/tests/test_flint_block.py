# Traces: OPEN-FAMILY-FLINT (block prefill), OPEN-PREFILL-BATCH
# (canonical spec: specs/open-engine/spec.md)
"""Recursive Flint block prefill (on by default; FLINT_BLOCK=0 drops it): the recipe's emission, the pool law the
token-batched expert kernel (designs/lfm2moe_batch/mbl.py at Flint's widths) reads in place, and that the block route is
a pure addition to the sequential set. Pure functions over the HF config fixture of test_flint.py -- no model, no
hardware (the route is validated on hardware: the model gate in .plans/flint-native.md)."""
from __future__ import annotations

import copy

import pytest

from recipes import flint as F

from test_flint import spec

BLOCK_KERNELS = ("gemm_", "mb_")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("FLINT_BLOCK", "1")


def test_on_by_default(monkeypatch):
    monkeypatch.delenv("FLINT_BLOCK", raising=False)
    assert F.gemm_route(spec()) is not None
    assert F.KERNEL_SOURCES_MOE, "the block designs are in the build key while the route is emitted"


def _strip_block(progs: dict) -> dict:
    """programs() with the block route's additions taken out."""
    p = copy.deepcopy(progs)
    for lt in p["layer_types"].values():
        lt.pop("gemm_block", None)
    for k in ("contexts", "kernels"):
        p[k] = {n: v for n, v in p[k].items() if not n.startswith(BLOCK_KERNELS) and n not in ("gemm", "mb")}
    p["globals"] = {n: v for n, v in p["globals"].items() if not n.startswith(BLOCK_KERNELS)}
    return p


def test_off_is_the_sequential_set_and_the_route_only_adds(monkeypatch):
    s = spec()
    monkeypatch.setenv("FLINT_BLOCK", "1")
    p_on, b_on, k_on = F.programs(s), F.builds(s), F.pack_plan(s)
    monkeypatch.setenv("FLINT_BLOCK", "0")
    p_off, b_off, k_off = F.programs(s), F.builds(s), F.pack_plan(s)
    assert F.gemm_route(s) is None and F.KERNEL_SOURCES_MOE == []
    assert all("gemm_block" not in lt for lt in p_off["layer_types"].values())
    assert not any(k.startswith(BLOCK_KERNELS) for k in b_off)
    # the route adds and changes nothing of the sequential set: programs, builds and every pool op in place
    assert _strip_block(p_on) == p_off
    assert {k: v for k, v in b_on.items() if not k.startswith(BLOCK_KERNELS)} == b_off
    for nm, lt in k_off["layer_types"].items():
        on_ops = k_on["layer_types"][nm]["pool"]
        assert on_ops[:len(lt["pool"])] == lt["pool"] and k_on["layer_types"][nm]["consts"] == lt["consts"]
        extra = on_ops[len(lt["pool"]):]
        assert [o["tensor"] for o in extra] == (["blk.{l}.ffn_up.weight", "blk.{l}.ffn_gate.weight",
                                                  "blk.{l}.ffn_down.weight"] if nm == "dense_mlp" else [])
    assert {k: v for k, v in k_on.items() if k != "layer_types"} == {k: v for k, v in k_off.items() if k != "layer_types"}


def test_the_route_is_one_flint_gemm_block_a_layer_type(on):
    s = spec()
    L = F.layout(s)
    g = F.programs(s)["layer_types"]
    assert sorted(g) == ["dense", "dense_local", "dense_mlp"]
    full, local, dense = (g[n]["gemm_block"] for n in ("dense", "dense_local", "dense_mlp"))
    for gb in (full, local, dense):
        assert (gb["kind"], gb["mixer"]) == ("flint", "att")
        assert [p["kernel"] for p in gb["program"]] == ["gemm_n5120_k2048", "gemm_n2048_k4096"]
        assert (gb["qw"], gb["kvw"], gb["nh"], gb["kvh"], gb["hd"], gb["rot"]) == (4096, 512, 32, 4, 128, 128)
        assert (gb["cd_qnorm"], gb["cd_knorm"], gb["cd_sinks"]) == (L.CD_META, L.CD_META + 256, L.CD_SINK)
        assert gb["t"] == 256 and gb["eps"] == 1e-6
    # the sliding layers rotate and read 513 rows; the full layers (the dense ones too) rotate nothing, see everything
    assert (local["window"], local["nope"]) == (513, False)
    assert (full["window"], full["nope"]) == (0, True) and (dense["window"], dense["nope"]) == (0, True)
    assert [p["kernel"] for p in dense["ffn_program"]] == ["gemm_n12288_k2048", "gemm_n2048_k6144"]
    assert dense["ff"] == 6144 and "moe_batch" not in dense and "route" not in dense
    for gb in (full, local):
        assert "ffn_program" not in gb
        assert gb["route"] == {"router": "blk.{l}.ffn_gate_inp.weight", "experts": 256, "topk": 6, "scale": 1.0,
                               "shared": [256, 257]}
        # the routed and the shared slots are moe_geometry's: 6 + 2 slots over 256 + 2 pool experts
        assert F.moe_geometry(s)["topk"] == 6 + 2 and F.moe_geometry(s)["experts"] == 256 + 2


def test_gemm_weights_are_contiguous_pool_runs(on):
    s = spec()
    plan = F.pack_plan(s)["layer_types"]
    for nm, dense in (("dense_mlp", True), ("dense_local", False)):
        P = F.pool(s, dense)
        gb = F.programs(s)["layer_types"][nm]["gemm_block"]
        ops = plan[nm]["pool"]
        qkv = [ops[i] for i in gb["weights"]["gqkv3_w"]["ops"]]
        assert [o["dst"] for o in qkv] == [P.POOL_Q, P.POOL_K, P.POOL_V]
        (o,) = (ops[i] for i in gb["weights"]["go_w"]["ops"])
        assert o["dst"] == P.POOL_O and (o["rows"], o["in_dim"]) == (2048, 4096)
    ops = plan["dense_mlp"]["pool"]
    gb = F.programs(s)["layer_types"]["dense_mlp"]["gemm_block"]
    up, gate = (ops[i] for i in gb["ffn_weights"]["gffn_ug_w"]["ops"])
    (down,) = (ops[i] for i in gb["ffn_weights"]["gffn_down_w"]["ops"])
    bb = F.band_bytes(2048)
    assert gate["dst"] == up["dst"] + 6144 // 64 * bb                       # [up | gate]: one GEMM of 2 x 6144 rows
    assert down["dst"] == gate["dst"] + 6144 // 64 * bb and (down["rows"], down["in_dim"]) == (2048, 6144)
    # the copies sit past the dense pool's attention and inside the one pool BO size every layer gets
    P = F.pool(s, dense=True)
    assert P.POOL_O + 32 * F.band_bytes(4096) <= up["dst"]
    assert down["dst"] + 2048 // 64 * F.band_bytes(6144) <= F.pool_bytes(s)


def test_the_expert_kernel_reads_flints_pool_law_in_place(on):
    s = spec()
    mb = F.moe_batch(s)
    assert list(mb["moe_batch"]["kernels"]) == ["256", "128", "64", "32", "16", "8"]
    assert mb["moe_batch"]["y_layout"] == "rbg_half_row" and mb["moe_batch"]["nt"] == 8
    b = mb["builds"]["mb_s256"]
    L, P = F.layout(s), F.pool(s, dense=False)
    assert b["design"] == "lfm2moe_batch/mbl.py"
    env = b["env"]
    assert (env["MBL_HID"], env["MBL_FF"], env["MBL_EXPERTS"]) == ("2048", "512", "258")
    assert int(env["MBL_POOL_DOWN"]) == P.POOL_DOWN == 258 * L.EXPERT_UG
    assert int(env["MBL_POOL_BYTES"]) == P.POOL_ATTN          # the expert region; the attention is never an operand
    # mbl.py's tap law at FF 512: gate / up bands BB_H apart and interleaved, an expert's gate | up twice its down
    HID, FF, CH = 2048, 512, 5120
    bb_h, bb_f = 2 * (HID // 256) * CH, 2 * (FF // 256) * CH
    assert L.EXPERT_UG == 2 * (FF // 64) * bb_h and L.UP_BYTES == (HID // 64) * bb_f == (FF // 64) * bb_h
    # ... and the pack ops write exactly that: gate band b at 2 b BB_H, up at (2 b + 1) BB_H, down after POOL_DOWN
    ops = F.pack_plan(s)["layer_types"]["dense_local"]["pool"]
    gate, up, down = (next(o for o in ops if o["tensor"].endswith(t)) for t in
                      ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
    assert (gate["dst"], gate["stripe_bytes"], gate["expert_bytes"]) == (0, 2 * bb_h, L.EXPERT_UG)
    assert (up["dst"], up["stripe_bytes"]) == (bb_h, 2 * bb_h)
    assert (down["dst"], down["expert_bytes"], down["stripe_bytes"]) == (P.POOL_DOWN, L.UP_BYTES, 0)
    sh = [o for o in ops if "shexp" in o["tensor"]]
    assert [o["dst"] for o in sh] == [256 * L.EXPERT_UG, 256 * L.EXPERT_UG + bb_h, P.POOL_DOWN + 256 * L.UP_BYTES]
    assert mb["globals"]["mb_y"] == 256 * 2048 * 8 * 4 and mb["globals"]["mb_h"] == 256 * 512 * 8 * 2


def test_mbl_fits_l1_at_flints_width():
    # mbl.py's L1 formula (NT 8, one u / g band): FF 512 holds an 8 KB h, under LFM2.5's 62.5 KB point; NT 16 would not fit
    def l1(nt, ff):
        band = 2 * 5120
        return (0x1000 + 2 * band + 2 * (64 * nt * 4) + 64 * nt * 4 + 2 * (32 * nt * 4) + 5 * (64 * nt * 2)
                + ff * nt * 2)
    assert l1(8, 512) == 46080 <= 64 << 10
    assert l1(16, 512) > 64 << 10
    # a core owns 2 gate / up bands (8 bands / 4 rows) and 8 down bands (32 / 4): whole, as mbl asserts
    assert (512 // 64) % 4 == 0 and (2048 // 64) % 4 == 0
