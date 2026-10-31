# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-GEMMA4 (canonical spec: specs/open-engine/spec.md)
"""Gemma 4 12B on two dense geometries: spec derivation, the proportional RoPE table, the
global layers' geometry and pack plan, the per-kernel rows in the manifest, and that the
catalogue does not mistake a Gemma 4 layer for the Gemma 3 point it shares a shape with."""
from __future__ import annotations

import numpy as np
import pytest

from recipes import gemma4 as G4
from recipes.catalogue import OpRangeError
from recipes.manifest import manifest
from recipes.spec import DENSE, DENSE_LOCAL, ModelSpec, SpecError

# google/gemma-4-12B-it config.json, the text tower's fields that matter (transformers 5.15)
TC = {
    "model_type": "gemma4_unified_text", "hidden_size": 3840, "intermediate_size": 15360,
    "num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
    "global_head_dim": 512, "num_global_key_value_heads": 1, "attention_k_eq_v": True,
    "layer_types": ["full_attention" if (l + 1) % 6 == 0 else "sliding_attention" for l in range(48)],
    "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                           "rope_type": "proportional"},
                        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    "sliding_window": 1024, "vocab_size": 262144, "final_logit_softcapping": 30.0,
    "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-06, "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0, "use_double_wide_mlp": False, "enable_moe_block": False,
    "attention_bias": False, "tie_word_embeddings": True,
}
HF_12B = {"model_type": "gemma4_unified", "text_config": TC}


@pytest.fixture
def unvalidated(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


def spec():
    return ModelSpec.from_hf_config(HF_12B)


def test_spec_from_hf():
    s = spec()
    assert s.family == "gemma4" and s.layer_types.count(DENSE) == 8 and s.layer_types[5] == DENSE
    assert s.layer_types[0] == DENSE_LOCAL and (s.head_dim, s.num_kv_heads, s.rotary_dim) == (256, 8, 256)
    assert (s.global_head_dim, s.global_kv_heads, s.global_rotary_dim) == (512, 1, 128)
    assert s.attn_k_eq_v and s.v_norm and s.attn_scale == 1.0 and s.final_softcap == 30.0
    assert (s.rope_theta, s.rope_local_theta, s.sliding_window) == (1e6, 1e4, 1024)


def test_the_new_fields_leave_every_other_spec_alone():
    d = ModelSpec.from_dict({k: v for k, v in spec().to_dict().items()}).to_dict()
    assert d["global_head_dim"] == 512
    g3 = dict(spec().to_dict(), family="gemma3")
    for k in ("global_head_dim", "global_kv_heads", "global_rotary_dim", "attn_k_eq_v", "v_norm",
              "attn_scale", "final_softcap"):
        g3.pop(k)
    assert not any(k in ModelSpec.from_dict(g3).to_dict() for k in ("v_norm", "final_softcap"))


def test_the_e_models_and_unbuilt_features_are_refused():
    with pytest.raises(SpecError, match="E2B / E4B"):
        ModelSpec.from_hf_config({"model_type": "gemma4", "text_config": dict(TC, model_type="gemma4_text")})
    with pytest.raises(SpecError, match="num_kv_shared_layers"):
        ModelSpec.from_hf_config({"model_type": "gemma4_unified", "text_config": dict(TC, num_kv_shared_layers=20)})
    rp = {"full_attention": {"rope_theta": 1e6, "rope_type": "default"}, "sliding_attention": TC["rope_parameters"]["sliding_attention"]}
    with pytest.raises(SpecError, match="proportional"):
        ModelSpec.from_hf_config({"model_type": "gemma4_unified", "text_config": dict(TC, rope_parameters=rp)})


def test_proportional_rope_table():
    """HF _compute_proportional_rope_parameters: 64 turning pairs at theta^(-2i/512); the
    other 192 are zero there, and the kernel's pair distance (256) carries them."""
    s = spec()
    g, l = np.array(s.rope_inv_freq()), np.array(s.rope_inv_freq(local=True))
    assert g.shape == (64,) and np.allclose(g, 1e6 ** (-2 * np.arange(64) / 512))
    assert l.shape == (128,) and np.allclose(l, 1e4 ** (-2 * np.arange(128) / 256))


def test_two_geometries(unvalidated):
    s = spec()
    gl, gg = G4.geometry(s, "local"), G4.geometry(s, "global")
    assert (gl.HD, gl.KVH, gl.ROT, gl.ROPE_PAIR) == (256, 8, 256, 0)
    assert (gg.HD, gg.KVH, gg.ROT, gg.ROPE_PAIR) == (512, 2, 128, 256)
    for g in (gl, gg):
        assert g.VNORM == 1 and g.SCALE == 1.0 and g.QLOCAL == 1 and g.N_CORES == 4 and g.NHL == 4
    assert G4.layout(s, kind="global").PTAB_ROW == 2048 and G4.layout(s, kind="local").PTAB_ROW == 4096


def test_global_pack_plan_duplicates_k_into_both_slots(unvalidated):
    s = spec()
    pool = G4.pack_plan(s)["layer_types"][DENSE]["pool"]
    names = [o["tensor"].rsplit(".", 2)[-2] for o in pool]
    assert names == ["q_proj", "k_proj", "k_proj", "k_proj", "k_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    L = G4.layout(s, kind="global")
    ks = [o["dst"] for o in pool if "k_proj" in o["tensor"]]
    assert ks[0] == L.POOL_K and ks[2] == L.POOL_V and ks[1] - ks[0] == ks[3] - ks[2] == (L.POOL_V - L.POOL_K) // 2
    consts = G4.pack_plan(s)["layer_types"][DENSE_LOCAL]["consts"]
    run = [o["tensor"].rsplit(".", 2)[-2] for o in consts if "div_running" in o]
    assert run == ["post_attention_layernorm", "post_feedforward_layernorm"]


# The div_running case gguf_pack_test.cpp also packs through pools.cpp: the same bf16 norm,
# the same two layer scales (one bf16, one f32), the same expected bits.
DIV_NORM = [0x3F80, 0x3FC0, 0xC030, 0x3DCD, 0x4049, 0x3F81]
DIV_SCALES = [np.array([0x3F33], np.uint16).view(np.uint8), np.array([0.3], np.float32).view(np.uint8)]
DIV_WANT_L2 = [0x4099, 0x40E5, 0xC152, 0x3EF4, 0x4170, 0x409A]


def test_put_div_running_matches_pools_cpp():
    from recipes import pack

    class Container:
        def raw(self, name):
            if name.startswith("norm."):
                return np.array(DIV_NORM, np.uint16).view(np.uint8)
            return DIV_SCALES[int(name.split(".")[1])]

    op = {"op": "put", "tensor": "norm.{l}", "dst": 0, "cap": 2 * len(DIV_NORM), "div_running": "scale.{j}"}
    for layer, want in ((0, DIV_NORM), (2, DIV_WANT_L2)):
        dst = np.zeros(2 * len(DIV_NORM), np.uint8)
        pack.apply_op(op, Container(), layer, dst)
        assert dst.view(np.uint16).tolist() == want, layer


def test_the_load_check_covers_both_rope_tables(unvalidated):
    """check_model compares rope_parameters whole: the 12B's own config passes, and a config
    that moves either theta or the global rotation width no longer matches."""
    import copy
    want = manifest(spec())["hf_config_check"]["rope_parameters"]
    assert want == TC["rope_parameters"]
    for path, value in ((("full_attention", "rope_theta"), 500000.0),
                        (("sliding_attention", "rope_theta"), 20000.0),
                        (("full_attention", "partial_rotary_factor"), 0.5)):
        tc = copy.deepcopy(TC)
        tc["rope_parameters"][path[0]][path[1]] = value
        moved = ModelSpec.from_hf_config({"model_type": "gemma4_unified", "text_config": tc})
        assert manifest(moved)["hf_config_check"]["rope_parameters"] != want, path
        assert tc["rope_parameters"] != want, path


def test_manifest_rows_are_per_kernel(unvalidated):
    m = manifest(spec())["gguf"]
    k = m["kernels"]
    assert (k["dx_local"]["context"], k["dx"]["context"]) == ("dx", "dxg")
    assert (k["dx_local"]["kv_row"], k["dx_local"]["ptab_row"]) == (8192, 4096)
    assert (k["dx"]["kv_row"], k["dx"]["ptab_row"]) == (4096, 2048)
    assert m["globals"]["ptab"]["rotary_dim"] == 128 and len(m["globals"]["ptab"]["inv_freq"]) == 64
    assert m["globals"]["ptab_local"]["rotary_dim"] == 256 and len(m["globals"]["ptab_local"]["inv_freq"]) == 128
    assert m["layout"]["final_logit_softcapping"] == 30.0
    assert m["layer_types"][DENSE]["buffers"]["state"]["row"] == 4096
    # manifest.cpp's own check: the layout's frequencies match the layout's rotary dim, and
    # they are the local table (the layout's rows are the local layers')
    for mm in (m, manifest(spec())):
        lay = mm["layout"]
        assert len(lay["rope_inv_freq"]) == lay["rotary_dim"] // 2
        assert lay["rope_inv_freq"] == mm["globals"]["ptab_local"]["inv_freq"]


def test_the_catalogue_does_not_take_gemma3s_point_for_a_gemma4_layer(monkeypatch):
    """The sliding layers have Gemma 3 12B's shape, which IS validated; with a V norm and a
    score scale of 1 it is another kernel, and it is refused until it has run."""
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    with pytest.raises(OpRangeError):
        G4._require(spec(), "local", G4.geometry(spec(), "local"))


def test_block_prefill_is_on_by_default_and_gguf_only(unvalidated, monkeypatch):
    """Unset (or =1): every GGUF layer type gets the block route, and the q4nx section never does
    (that route has not run for this model). DENSE_GGUF_GEMM=0: the sequential route and the build
    list it had before the route existed."""
    monkeypatch.setenv("DENSE_GGUF_COMPACT", "0")
    monkeypatch.delenv("DENSE_GGUF_GEMM", raising=False)
    m = manifest(spec())
    assert all("gemm_block" in lt for lt in m["gguf"]["layer_types"].values())
    assert not any("gemm_block" in lt for lt in m["layer_types"].values())
    assert any(n.startswith("gemm_") for n in G4.builds(spec())) and "dx_attn_local" in G4.builds(spec())
    monkeypatch.setenv("DENSE_GGUF_GEMM", "1")
    forced = manifest(spec())

    def keyless(mm):     # the probe variable's value joins the build key, nothing else
        return {k: keyless(v) if k == "gguf" else v for k, v in mm.items() if k != "build_key"}

    assert keyless(forced) == keyless(m)
    monkeypatch.setenv("DENSE_GGUF_GEMM", "0")
    off = manifest(spec())
    for sect in (off, off["gguf"]):
        assert not any("gemm_block" in lt for lt in sect["layer_types"].values())
    assert not any(n.startswith(("gemm_", "dx_attn")) for n in G4.builds(spec()))
    assert "DENSE_GGUF_GEMM" in G4.probe_env()     # in the build key


def test_block_prefill_has_each_kinds_geometry(unvalidated, monkeypatch):
    """Per layer type: its own qkv3 / o GEMM shapes, attention kernel, context, window and rows.
    The global qkv3 weight is q plus the four k_proj copies of the K and V slots, consecutive."""
    monkeypatch.setenv("DENSE_GGUF_GEMM", "1")
    monkeypatch.setenv("G4_Q40", "0")                  # not the Q4_0-native view ...
    monkeypatch.setenv("DENSE_GGUF_COMPACT", "0")      # ... nor the compact one: the f32-scale twin's names
    monkeypatch.setenv("G4_BFP16", "0")                # the q4 route as gemm_block
    m = manifest(spec())["gguf"]
    loc, glo = m["layer_types"][DENSE_LOCAL]["gemm_block"], m["layer_types"][DENSE]["gemm_block"]
    assert (loc["qw"], loc["kvw"], glo["qw"], glo["kvw"]) == (4096, 2048, 8192, 1024)
    assert loc["program"][0]["kernel"] == "gemm_n8192_k3840_f32" and glo["program"][0]["kernel"] == "gemm_n10240_k3840_f32"
    assert loc["program"][1]["kernel"] == "gemm_n3840_k4096_f32" and glo["program"][1]["kernel"] == "gemm_n3840_k8192_f32"
    assert loc["program"][2:] == glo["program"][2:]
    plan = G4.pack_plan(G4.replace(spec(), quant="q4_1_f32"))["layer_types"][DENSE]["pool"]
    assert [plan[i]["tensor"].split(".")[-2] for i in glo["weights"]["gqkv3_w"]["ops"]] == ["q_proj"] + ["k_proj"] * 4
    assert (loc["attn_kernel"], glo["attn_kernel"]) == ("dxB_local", "dxB")
    k = m["kernels"]
    assert (k["dxB_local"]["context"], k["dxB"]["context"]) == ("dxa", "dxag")
    assert (k["dxB_local"]["window"], k["dxB"]["window"]) == (1024, 0)
    assert (k["dxB_local"]["kv_row"], k["dxB"]["kv_row"]) == (8192, 4096)
    assert (loc["ad_og"], glo["ad_og"]) == (40960, 49152)
    assert loc["sandwich"] and glo["act"] == "gelu_tanh"
    # the host chain reads the FFN-side norms from the consts, whose post_feedforward has the
    # layer scale folded in (div_running); the file's copy has not
    for gb, kind in ((loc, "local"), (glo, "global")):
        L = G4.layout(spec(), 4096, kind)
        assert (gb["cd_preffn"], gb["cd_postffn"]) == (L.CD_PREFFN, L.CD_POSTFFN)
    consts = G4.pack_plan(spec())["layer_types"][DENSE]["consts"]
    assert {c["dst"]: "div_running" in c for c in consts}[glo["cd_postffn"]]
    assert m["globals"]["gact"] == 256 * G4.layout(spec(), 4096, "global").AD_BYTES
    b = G4.builds(spec())
    assert (b["dx_attn_local"]["env"], b["dx_attn_global"]["env"]) == ({G4.KIND_ENV: "local"}, {G4.KIND_ENV: "global"})
    assert all(b[n]["env"]["GQP_SCALES_F32"] == "1" for n in b if n.startswith("gemm_"))


@pytest.mark.parametrize("gemm", ["0", "1"])
def test_compact_gguf_view_is_the_q4_1_sections_pool_and_kernels(unvalidated, monkeypatch, gemm):
    """Without the Q4_0-native view (G4_Q40=0) and with DENSE_GGUF_COMPACT unset, or 1, the GGUF
    view keeps its reader (std_perm_gguf, the scaled embedding, the folded consts) on the q4_1
    section's chunk law, offsets, buffers, decode kernels and head, and the block route's GEMMs
    are the q4_1-law builds. =0 gives the f32-scale twin. The q4nx section does not change."""
    monkeypatch.setenv("G4_Q40", "0")
    monkeypatch.setenv("DENSE_GGUF_GEMM", gemm)
    monkeypatch.setenv("DENSE_GGUF_COMPACT", "0")
    twin = manifest(spec())
    monkeypatch.delenv("DENSE_GGUF_COMPACT")
    m = manifest(spec())
    monkeypatch.setenv("DENSE_GGUF_COMPACT", "1")
    forced = manifest(spec())["gguf"]
    assert {k: v for k, v in forced.items() if k != "build_key"} == {k: v for k, v in m["gguf"].items() if k != "build_key"}
    g = m["gguf"]
    assert {k: v for k, v in m.items() if k not in ("gguf", "builds", "build_key")} == \
        {k: v for k, v in twin.items() if k not in ("gguf", "builds", "build_key")}
    assert (g["layout"]["chunk_bytes"], g["pack"]["chunk_bytes"]) == (5120, 5120)
    assert (twin["gguf"]["layout"]["chunk_bytes"], g["spec"]["quant"]) == (6144, "q4_1")
    assert g["pack"]["pool_bytes"] == m["pack"]["pool_bytes"] < twin["gguf"]["pack"]["pool_bytes"]
    assert g["globals"]["lmpool"] == m["globals"]["lmpool"] == g["layout"]["lmhead_pool_bytes"]
    assert g["pack"]["embed"] == twin["gguf"]["pack"]["embed"] and "scale" in g["pack"]["embed"]
    assert [o["op"] for o in g["pack"]["lm_head"]["ops"]] == ["std_perm_gguf"]
    for name, lt in g["layer_types"].items():
        base = m["layer_types"][name]
        assert lt["pack"]["consts"] == twin["gguf"]["layer_types"][name]["pack"]["consts"] == base["pack"]["consts"]
        assert [(o["tensor"], o["dst"], o["nch"]) for o in lt["pack"]["pool"]] == \
            [(o["tensor"], o["dst"], o["nch"]) for o in base["pack"]["pool"]]
        assert {o["op"] for o in lt["pack"]["pool"]} == {"std_perm_gguf"}
        assert lt["program"] == base["program"] and lt["buffers"] == base["buffers"]
        gb = lt.get("gemm_block")
        assert (gb is not None) == (gemm == "1")
        if gb:
            assert not any(st["kernel"].endswith("_f32") for st in gb["program"])
            tw = twin["gguf"]["layer_types"][name]["gemm_block"]
            assert gb["weights"] == tw["weights"] and gb["cd_postffn"] == tw["cd_postffn"]
    for kn in ("dx_local", "dx", "lm"):
        assert g["kernels"][kn] == m["kernels"][kn]
        assert g["contexts"][g["kernels"][kn]["context"]] == m["contexts"][g["kernels"][kn]["context"]]
    assert not any(k.endswith("_f32") for k in g["kernels"]) and not g["contexts"]["dx"].startswith("dx_f32")
    monkeypatch.delenv("DENSE_GGUF_COMPACT")
    b = G4.builds(spec())
    gemms = [n for n in b if n.startswith("gemm_n")]
    assert len(gemms) == (6 if gemm == "1" else 0)
    assert sum(n.startswith("gemm_bfp16_") for n in b) == (6 if gemm == "1" else 0)
    assert not any("GQP_SCALES_F32" in b[n]["env"] for n in gemms)
    assert all(n in b for n in g["builds"] if n.startswith("gemm_"))
    monkeypatch.setenv("DENSE_GGUF_COMPACT", "0")
    assert "DENSE_GGUF_COMPACT" in G4.probe_env()     # a set value joins the build key


def test_q40_gguf_view_is_its_own_pool_law_and_kernels(unvalidated, monkeypatch):
    """By default (G4_Q40 unset, or 1) the GGUF view is built at the Q4_0-native law (4608-B
    chunks, no min plane): its own offsets, a tenth smaller pools and head, the _q40 decode, head
    and lean GEMM builds, and the same reader, consts, buffers and bfp16 route as the compact view
    (G4_Q40=0). The q4nx section does not move."""
    m = manifest(spec())
    monkeypatch.setenv("G4_Q40", "1")
    assert {k: v for k, v in manifest(spec())["gguf"].items() if k != "build_key"} == \
        {k: v for k, v in m["gguf"].items() if k != "build_key"}
    monkeypatch.setenv("G4_Q40", "0")
    assert G4.probe_env()["G4_Q40"] == "0"              # a set value joins the build key
    compact = manifest(spec())
    assert not any("q40" in n for n in G4.builds(spec()))
    monkeypatch.delenv("G4_Q40")
    g, c = m["gguf"], compact["gguf"]
    assert {k: v for k, v in m.items() if k not in ("gguf", "builds", "build_key")} == \
        {k: v for k, v in compact.items() if k not in ("gguf", "builds", "build_key")}
    assert (g["spec"]["quant"], g["layout"]["chunk_bytes"], g["pack"]["chunk_bytes"]) == ("q4_0", 4608, 4608)
    assert g["pack"]["pool_bytes"] < c["pack"]["pool_bytes"] and g["globals"]["lmpool"] < c["globals"]["lmpool"]
    assert g["pack"]["embed"] == c["pack"]["embed"] and [o["op"] for o in g["pack"]["lm_head"]["ops"]] == ["std_perm_gguf"]
    for name, lt in g["layer_types"].items():
        cl = c["layer_types"][name]
        assert lt["pack"]["consts"] == cl["pack"]["consts"] and lt["buffers"] == cl["buffers"]
        assert {o["op"] for o in lt["pack"]["pool"]} == {"std_perm_gguf"}
        assert [(o["tensor"], o["nch"]) for o in lt["pack"]["pool"]] == [(o["tensor"], o["nch"]) for o in cl["pack"]["pool"]]
        assert lt["pack"]["pool"][-1]["dst"] < cl["pack"]["pool"][-1]["dst"]
        assert lt["gemm_block"] == cl["gemm_block"]               # bfp16, packed from the GGUF itself
        lean = lt["gemm_block_variants"]["lean"]
        assert all(st["kernel"].endswith("_q40") for st in lean["program"] if st["kernel"].startswith("gemm_n"))
    assert (g["kernels"]["dx_local"]["build"], g["kernels"]["dx"]["build"], g["kernels"]["lm"]["build"]) == \
        ("dx_q40", "dxg_q40", "lm_head_q4_q40")
    b = G4.builds(spec())
    assert b["dx_q40"]["env"]["GEMV_Q4_0"] == "1" and b["lm_head_q4_q40"]["env"]["LMHEAD_Q4_0"] == "1"
    gemms = [n for n in b if n.startswith("gemm_n")]
    assert len(gemms) == 6 and all(n.endswith("_q40") and b[n]["env"]["GQP_FMT"] == "q4_0" for n in gemms)
    assert all(st in b for st in g["builds"])


def test_attn_block_is_on_by_default_and_carries_each_kinds_attention(unvalidated, monkeypatch):
    """By default each GGUF layer type's block route gets the NPU attention products with prep
    qkvnorm_rope and its own rotation, window, norm offsets and window tiers; the dxB kernel stays
    (OFLM_OPEN_ATTN_BLOCK=0 at run time falls back to it). G4_ATTN_BLOCK=0: the route without them."""
    monkeypatch.setenv("G4_ATTN_BLOCK", "0")
    assert G4.probe_env()["G4_ATTN_BLOCK"] == "0"           # a set value joins the build key
    plain = manifest(spec())["gguf"]
    assert not any("attn_block" in lt["gemm_block"] for lt in plain["layer_types"].values())
    assert not any(n.startswith("ag_") for n in G4.builds(spec()))
    monkeypatch.delenv("G4_ATTN_BLOCK")
    g = manifest(spec())["gguf"]
    loc, glo = (g["layer_types"][lt]["gemm_block"]["attn_block"] for lt in (DENSE_LOCAL, DENSE))
    assert (loc["m"], loc["hd"], loc["l_max"], loc["window"]) == (2 * 256, 256, 1280, 1024)
    assert (glo["m"], glo["hd"], glo["l_max"], glo["window"]) == (8 * 256, 512, 4096, 0)
    assert (loc["rot"], loc["rope_pair"], glo["rot"], glo["rope_pair"]) == (256, 128, 128, 256)
    assert loc["inv_freq"] == g["globals"]["ptab_local"]["inv_freq"] and glo["inv_freq"] == g["globals"]["ptab"]["inv_freq"]
    for ab, kind in ((loc, "local"), (glo, "global")):
        L = G4.layout(spec(), 4096, kind)
        assert ab["prep"] == "qkvnorm_rope" and ab["scale"] == 1.0 and ab["qsplit"]
        assert (ab["cd_qnorm"], ab["cd_knorm"]) == (L.CD_META, L.CD_META + ab["hd"] * 2)
        assert sorted(map(int, ab["kernels_s"])) == sorted(map(int, ab["kernels_pv"])) == list(range(256, ab["l_max"] + 1, 256))
        assert sorted(map(int, ab["kernels_shl"])) == sorted(map(int, ab["kernels_s"]))
        # every product of every kind on one image, so a layer's S -> P.V costs no context switch
        for name in list(ab["kernels_s"].values()) + list(ab["kernels_pv"].values()) + list(ab["kernels_shl"].values()):
            assert g["kernels"][name]["context"] == "ag"
    assert g["contexts"]["ag"] == "ag_s256_local/final.xclbin"
    assert not any(c.startswith("ag_") for c in g["contexts"])
    # the consts put q_norm / k_norm exactly where the host half reads them
    consts = G4.pack_plan(spec())["layer_types"][DENSE]["consts"]
    at = {c["dst"]: c["tensor"].split(".")[-2] for c in consts}
    assert (at[glo["cd_qnorm"]], at[glo["cd_knorm"]]) == ("q_norm", "k_norm")
    assert g["globals"]["ag_c"] == 2 * 2048 * 4096 * 4      # the global layers' stacked hi / lo scores
    assert {lt: "dxB" in v["gemm_block"]["attn_kernel"] for lt, v in g["layer_types"].items()} == {DENSE_LOCAL: True, DENSE: True}
    b = G4.builds(spec())
    assert sum(n.startswith("ag_") for n in b) == 3 * (5 + 16)
    assert b["ag_pv1280_local"]["env"] == {"AG_M": "512", "AG_K": "1280", "AG_N": "256", "AG_COLS": "8"}
    assert b["ag_shl1280_local"]["env"] == {"AG_M": "1024", "AG_K": "256", "AG_N": "1280", "AG_COLS": "8"}
    assert b["ag_shl4096_global"]["env"] == {"AG_M": "4096", "AG_K": "512", "AG_N": "4096", "AG_COLS": "8"}


def test_bfp16_route_is_the_default_and_the_q4_route_is_lean(unvalidated, monkeypatch):
    """By default (G4_BFP16 unset, or 1) each GGUF layer type's gemm_block -- the route the default
    `fast` prefill mode runs -- is the q4 route's five steps on gemm_bfp16 (OPEN-PREFILL-BFP16), each
    weight its pool ops' tensors as bfp16_gemm ops stacked in the pool's row order and padded to
    whole 1024-row groups, on one context; the q4 route is the `lean` variant, unchanged. G4_BFP16=0:
    the q4 route alone, as gemm_block, and the manifest version it had."""
    monkeypatch.setenv("G4_BFP16", "0")
    assert G4.probe_env()["G4_BFP16"] == "0"               # a set value joins the build key
    q4 = manifest(spec())
    assert q4["gguf"]["manifest_version"] == 1
    assert not any("gemm_block_variants" in lt for lt in q4["gguf"]["layer_types"].values())
    assert not any(n.startswith("gemm_bfp16") for n in G4.builds(spec()))
    monkeypatch.delenv("G4_BFP16")
    m = manifest(spec())
    g = m["gguf"]
    assert g["manifest_version"] == 4 and m["manifest_version"] == 1     # bfp16_gemm ops: version 4
    plans = G4.pack_plan(G4.replace(spec(), quant="q4_1_f32"))["layer_types"]
    for name, lt in g["layer_types"].items():
        base = q4["gguf"]["layer_types"][name]["gemm_block"]
        assert lt["gemm_block_variants"] == {"lean": base}
        gb = lt["gemm_block"]
        assert {k: v for k, v in gb.items() if k not in ("weights", "program")} == \
            {k: v for k, v in base.items() if k not in ("weights", "program")}
        assert [st["args"][0] for st in gb["program"]] == [st["args"][0] for st in base["program"]]
        for st, b in zip(gb["program"], base["program"]):
            K = int(b["args"][1].removeprefix("gemm_x_k"))
            N = int(b["kernel"].split("_n")[1].split("_k")[0])
            n_pad = -(-N // 1024) * 1024
            assert st == {"op": "run", "kernel": f"gemm_bfp16_n{n_pad}_k{K}", "fmt": "bfp16", "n_pad": n_pad,
                          "args": [b["args"][0], f"gbfp_x_k{K}", f"gbfp_y_n{n_pad}"]}
            assert g["globals"][f"gbfp_x_k{K}"] == 256 * K // 8 * 9 and g["globals"][f"gbfp_y_n{n_pad}"] == n_pad * 256 * 4
            assert g["kernels"][st["kernel"]]["context"] == "gemmbfp"
            w = gb["weights"][st["args"][0]]
            pool = plans[name]["pool"]
            src = [pool[i] for i in base["weights"][st["args"][0]]["ops"]]
            assert w["from"] == "pack" and [o["tensor"] for o in w["pack"]] == [o["tensor"] for o in src]
            rows = [o["nch"] // (o["in_dim"] // 256) * 32 for o in src]
            assert [o["row0"] for o in w["pack"]] == [sum(rows[:i]) for i in range(len(rows))]
            assert {o["n_total"] for o in w["pack"]} == {n_pad} and sum(rows) == N
            assert all(o["op"] == "bfp16_gemm" and o["dst"] == 0 and o["in_dim"] == K for o in w["pack"])
    loc, glo = (g["layer_types"][lt]["gemm_block"]["program"] for lt in (DENSE_LOCAL, DENSE))
    assert [s["kernel"] for s in loc] == ["gemm_bfp16_n8192_k3840", "gemm_bfp16_n4096_k4096", "gemm_bfp16_n15360_k3840",
                                         "gemm_bfp16_n15360_k3840", "gemm_bfp16_n4096_k15360"]
    assert [s["kernel"] for s in glo[:2]] == ["gemm_bfp16_n10240_k3840", "gemm_bfp16_n4096_k8192"]
    assert g["contexts"]["gemmbfp"] == "gemm_bfp16_n8192_k3840/final.xclbin"
    b = G4.builds(spec())
    assert b["gemm_bfp16_n4096_k15360"] == {"design": "gemm_bfp16/gemm_bfp16.py", "build_dir": "gemm_bfp16/build_n4096_k15360_m256",
                                            "env": {"GB_M": "256", "GB_K": "15360", "GB_N": "4096"}}
    assert all(n in b for n in g["builds"] if n.startswith("gemm_"))
