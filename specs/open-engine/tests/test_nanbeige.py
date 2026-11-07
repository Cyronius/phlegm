# Traces: OPEN-FAMILY-NANBEIGE (canonical spec: specs/open-engine/spec.md)
"""Nanbeige4.3-33B on the dense recipe: a Llama (7168 / 52 / 22016, 56 heads over 8 kv heads) whose
attention, norm helper, down GEMV and GGUF view each needed a geometry rule the recipe did not have.
Offline: derivation, layout, the wide attention split, the quarter norm helper's element walk, the
partial down piece, the q4_1-law GGUF section and the head's element size. The NPU acceptance is the
spec's manual procedure."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes import dense as DR
from recipes.attnknobs import AttnKnobs
from recipes.catalogue import OpRangeError
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes.qwen36moe import down_split
from recipes.spec import DENSE, ModelSpec

REPO = Path(__file__).resolve().parents[3]
SPEC = REPO / "open_kernels" / "recipes" / "specs" / "nanbeige43-33b.json"
# Nanbeige/Nanbeige4.3-33B @ df919e1e config.json (the keys the derivation reads)
HF_NANBEIGE43_33B = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 7168,
    "intermediate_size": 22016, "num_hidden_layers": 52, "num_attention_heads": 56, "num_key_value_heads": 8,
    "head_dim": 128, "rms_norm_eps": 1e-05, "rope_theta": 120000000, "rope_scaling": None,
    "vocab_size": 166912, "tie_word_embeddings": False, "hidden_act": "silu", "attention_bias": False,
    "mlp_bias": False,
}


@pytest.fixture
def spec(monkeypatch):
    # The 2026-11-07 hardware pass catalogued ln 7168, gemv_q4 K 7168 / 11776, lm_head_q8 K 7168 and the
    # (128, 56, 8, 128) attention. The base manifest's q4 head (lm_head_q4 K 7168) never ran -- the GGUF view
    # uses the Q8_0 head -- so it stays outside the catalogue (test_only_the_q4_head_is_unvalidated).
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    return ModelSpec.from_hf_config(HF_NANBEIGE43_33B, real_vocab=166107)


def test_derives_on_the_llama_recipe_and_matches_the_checked_in_spec(spec):
    assert spec.family == "llama3" and spec.layer_types == tuple([DENSE] * 52)
    assert (spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.rotary_dim) == (56, 8, 128, 128)
    assert spec.rope_theta == 1.2e8 and spec.rope_scaling is None and spec.norm_eps == 1e-5
    assert spec.vocab == 166912 and spec.real_vocab == 166107       # 166107 tokenizer ids, an 805-row padded head
    assert load_spec(SPEC).spec_hash() == spec.spec_hash()


def test_layout(spec):
    R = DR.recipe(spec)
    L, G = R.layout, R.geo
    assert G.N_CORES == 8 and (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (14, 2, 14, 43, 14)
    assert G.DOWN_SPLIT == (10240, 11776) and G.PER_CALL == 1 and G.KWIDE == 11776 and G.TAB_BYTES == 26496
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (4, 4, 4, 22)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (14336, 2048, 4096, 2048)
    assert L.POOL_BYTES == 370147328 and L.LMHEAD_BANDS == 2608


def test_wide_attention_split(spec):
    """The default split puts 14 heads on each of 4 cores, not whole 8-head og elements: 7 cores of 8
    heads on row 4 (AWIDE), each keeping its own two q elements (QLOCAL), 4 rows a call."""
    G = DR.geometry(spec)
    assert (G.HPE, G.HPO) == (4, 8)
    assert (G.ACORES, G.NHL, G.RB, G.MLS, G.AWIDE, G.QLOCAL) == (7, 8, 4, 32, 1, 1)
    assert "-DATTN_QLOCAL=1" in DR.attn_flags(G) and "-DATTN_NHL=8" in DR.attn_flags(G)


def test_wide_attention_leaves_every_working_split_alone():
    base = AttnKnobs(VEXP=1, MLS=32, ACORES=4, NHL=8, RB=4)
    assert DR.wide_attention(32, 4, 8, base) is None              # Llama 3.1 8B: 4 x 8
    assert DR.wide_attention(20, 2, 4, AttnKnobs(VEXP=1, MLS=32, ACORES=5, NHL=4, RB=4)) is None
    assert DR.wide_attention(56, 4, 8, AttnKnobs(VEXP=0, MLS=56, ACORES=1, NHL=56, RB=1)) is None  # slow path
    with pytest.raises(OpRangeError):
        DR.wide_attention(44, 4, 8, AttnKnobs(VEXP=1, MLS=32, ACORES=4, NHL=11, RB=1))


def test_the_quarter_norm_helper_only_where_the_split_one_overflows(spec):
    assert DR.norm_quarter(spec) and DR.geometry(spec).NORM_SPLIT
    q27 = ModelSpec.from_hf_config({**HF_NANBEIGE43_33B, "hidden_size": 5120, "intermediate_size": 17408,
                                    "num_attention_heads": 40}, real_vocab=166107)
    assert not DR.norm_quarter(q27)                               # 5120: the split helper fits (57 344 B)


def test_partial_down_piece_is_dense_only(spec):
    """22016 is 21.5 f32 elements: the dense recipe's down split ends its second piece inside one, the
    shared default still refuses it (the MoE / Qwen3.5 tails fill whole elements)."""
    assert down_split(spec, partial=True) == (10240, 11776)
    with pytest.raises(OpRangeError):
        down_split(spec)


class FakeFifo:
    """ObjectFifo's sliding window: acquire(n) tops the held window up to n, release(n) frees the oldest."""

    def __init__(self, items):
        self.items, self.held, self.log = list(items), [], []

    def acquire(self, n):
        while len(self.held) < n:
            self.held.append(self.items.pop(0))
        return self.held[0] if n == 1 else list(self.held[:n])

    def release(self, n):
        del self.held[:n]


class Sink:
    def __init__(self):
        self.n = 0

    def acquire(self, n):
        self.n += 1
        return f"o{self.n - 1}"

    def release(self, n):
        pass


def test_quarter_norm_body_walks_the_fills_in_order():
    import ln_split as LS
    calls = []
    ain = FakeFifo(["w0", "x0", "x1", "x2", "x3", "w1"] + [f"{k}{q}" for q in range(4) for k in "xa"]
                   + ["v0", "r0", "r1", "r2", "r3", "v1"] + [f"{k}{q}" for q in range(4) for k in "rab"])
    out = Sink()
    LS.ln_quarter_body3(ain, out, lambda *a: calls.append(("nrq",) + a), lambda *a: calls.append(("add2",) + a),
                        lambda *a: calls.append(("add3",) + a))
    assert calls[0] == ("nrq", "w0", "x0", "x1", "x2", "x3", "o0", 0)
    assert calls[1] == ("nrq", "w1", "x0", "x1", "x2", "x3", "o1", 1)
    assert [c[:3] for c in calls[2:6]] == [("add2", f"x{q}", f"a{q}") for q in range(4)]
    assert calls[6][:6] == ("nrq", "v0", "r0", "r1", "r2", "r3") and calls[7][:6] == ("nrq", "v1", "r0", "r1", "r2", "r3")
    assert [c[:4] for c in calls[8:12]] == [("add3", f"r{q}", f"a{q}", f"b{q}") for q in range(4)]
    assert not ain.items and not ain.held          # every fill consumed, nothing left held


def test_gguf_section_is_the_q4_1_program_packed_from_the_gguf(spec):
    m = manifest(spec)
    g = m["gguf"]
    assert DR.gguf_q41_view(spec) and not DR.f32_twin_fits(spec)
    assert g["layout"]["chunk_bytes"] == m["layout"]["chunk_bytes"] == 5120
    assert g["layout"]["pool_bytes"] == m["layout"]["pool_bytes"]
    pool = g["layer_types"]["dense"]["pack"]["pool"]
    base = m["layer_types"]["dense"]["pack"]["pool"]
    assert [o["op"] for o in pool] == ["std_perm_gguf"] * 7
    assert [{k: v for k, v in o.items() if k != "op"} for o in pool] == [{k: v for k, v in o.items() if k != "op"} for o in base]
    assert g["pack"]["lm_head"]["ops"] == [{"op": "lmhead_q8", "tensor": "lm_head.weight", "dst": 0,
                                            "chunk_bytes": 8704, "in_dim": 7168}]
    assert g["pack"]["lm_head"]["pool_bytes"] == g["layout"]["lmhead_pool_bytes"] == g["globals"]["lmpool"] == 1271922688
    assert g["kernels"]["lm"]["build"] == "lm_head_q8_gguf" and "lm_head_q8_gguf" in m["builds"]
    assert m["builds"]["lm_head_q8_gguf"]["env"]["LMHEAD_SCALES_F32"] == "0"
    assert g["layer_types"]["dense"]["program"] == m["layer_types"]["dense"]["program"]
    gb = g["layer_types"]["dense"]["gemm_block"]
    assert gb["attn_block"]["m"] == 7 * 256 and gb["attn_block"]["prep"] == "rope" and gb["attn_kernel"] == "dxB"
    assert [s["kernel"] for s in gb["program"]] == ["gemm_n9216_k7168", "gemm_n7168_k7168", "gemm_n22016_k7168",
                                                    "gemm_n22016_k7168", "gemm_n7168_k22016"]


def test_q8_head_takes_one_chunk_a_call_at_7168(monkeypatch):
    """lm_head_q8.py: two 8704-B chunks a weight element overflow a core beside a 7168-wide x."""
    import importlib.util
    pytest.importorskip("aie.iron")
    path = REPO / "open_kernels" / "designs" / "lm_head_q8" / "lm_head_q8.py"
    for k, want in (("2048", 2), ("4096", 2), ("5120", 2), ("7168", 1)):
        monkeypatch.setenv("LMHEAD_K", k)
        mod_spec = importlib.util.spec_from_file_location(f"lmq8_probe_{k}", path)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        assert mod.PER_CALL == want, k


def test_only_the_q4_head_is_unvalidated(monkeypatch):
    """Without OPEN_KERNELS_UNVALIDATED the recipe refuses exactly at lm_head_q4 K 7168: every point the
    hardware pass exercised (norm, GEMVs, Q8 head, attention) is catalogued."""
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    s = ModelSpec.from_hf_config(HF_NANBEIGE43_33B, real_vocab=166107)
    with pytest.raises(OpRangeError, match=r"lm_head_q4: K=7168"):
        DR.recipe(s)
