"""The Qwen3 dense recipe: ModelSpec -> everything designs/dense/dx.py, the
packers and the driver need for a GQA + silu-FFN decoder layer.

    ln -> gemv q | k | v -> attention (q/k RMSNorm, full RoPE, no gate) -> gemv o
       -> ln (+residual) -> gemv up | gate -> silu(gate) * up -> gemv down -> +residual

One xclbin, ONE instruction stream per layer (no routing read, so no part
split). Same 8 main cores as the MoE designs (w / x / y streams), the ln
helper core, the attention helper core; the same six buffer arguments
(pool, xres, consts, kv, act, ptab). The lm_head is a q4_1 GEMV
(designs/lm_head_q4), the final norm the ln design at this width.

Element sizes, all derived: x-stream elements are 4 KB (ELEM), the ln core's
elements are HID*2 bytes (ELN: x as two f32 halves, w / xn as one bf16
element), the attention core's elements are one KV row half (E_A = KVH*HD
bf16), so a q element carries KVH/2 heads and an og element KVH heads. For
the 27B these rules give the sizes ax.py uses (4096 / 4096 / 1024).
"""
from __future__ import annotations

from dataclasses import dataclass

from .catalogue import LIMITS, OpRangeError, check_buffer_args, require
from .qwen36moe import BAND_ROWS, CHUNK, ELEM, MB, PER_CALL, CALL_BYTES, band_bytes, q4_bytes, q4_chunks, roundup, tab_bytes
from .spec import DENSE, ModelSpec


@dataclass(frozen=True)
class DenseLayout:
    # consts: [lnw (ELN)][postln (ELN)][meta: qn bf16 HD @0 | kn @HD*2 (E_A)]
    CD_LNW: int; CD_POSTLN: int; CD_META: int; CD_BYTES: int
    # act: the DDR bounce between stages
    AD_XN: int; AD_Q: int; AD_KVN: int; AD_OG: int; AD_OUT: int; AD_RES: int; AD_XM: int
    AD_H: int; AD_OUT2: int; AD_JUNK: int; AD_BYTES: int
    # pool
    POOL_Q: int; POOL_K: int; POOL_V: int; POOL_O: int; POOL_UP: int; POOL_GATE: int; POOL_DOWN: int; POOL_BYTES: int
    # KV / ptab / lm_head
    KV_ROW: int; PTAB_ROW: int; MAX_CTX: int; KV_BYTES: int; PTAB_BYTES: int
    LMHEAD_POOL_BYTES: int; LMHEAD_BANDS: int; LMHEAD_BAND_BYTES: int
    ELN: int; E_A: int

    def constants(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DenseGeometry:
    N_CORES: int; HID: int; FF: int; NH: int; KVH: int; HD: int; ROT: int; GATE: bool
    QW: int; KVW: int
    Q_PC: int; KV_PC: int; O_PC: int; UP_PC: int; DOWN_PC: int      # bands per core
    XN_ELEMS: int; XN_BLOCKS: int; OG_ELEMS: int; OG_BLOCKS: int    # x-stream elements / 32-blocks of the activations
    H_ELEMS: int; H_BLOCKS: int; XM_ELEMS: int
    HPE: int; HPO: int                                              # heads per ain element / per og element
    Q_AIN_ELEMS: int; K_AIN_ELEMS: int; OG_AOUT_ELEMS: int
    TAB_BYTES: int; KWIDE: int
    MS_U: int; MS_G: int; MS_FLOATS: int                            # the up / gate band scratch


@dataclass(frozen=True)
class DenseRecipe:
    spec: ModelSpec
    layout: DenseLayout
    geo: DenseGeometry
    max_ctx: int = 4096


def _check(spec: ModelSpec) -> None:
    n = LIMITS["n_cols"]
    if spec.family != "qwen3":
        raise OpRangeError(f"qwen3 recipe given a {spec.family!r} spec")
    if spec.quant != "q4_1":
        raise OpRangeError(f"qwen3: quant={spec.quant!r}; the gemv_q4 template reads q4_1 chunks only")
    if not spec.has_dense or spec.has_linear or spec.has_full or spec.intermediate == 0:
        raise OpRangeError("qwen3: every layer must be a dense layer with an FFN")
    for what, v in (("hidden", spec.hidden), ("intermediate", spec.intermediate), ("q width", spec.attn_q_width),
                    ("kv width", spec.attn_kv_width)):
        if v % (BAND_ROWS * n):
            raise OpRangeError(f"qwen3: {what} {v} is not a multiple of {BAND_ROWS * n} (64-row bands over {n} cores)")
    if spec.hidden % 256 or spec.intermediate % 256:
        raise OpRangeError("qwen3: hidden and intermediate must be multiples of 256 (one q4 k-tile)")
    if spec.num_kv_heads % 2:
        raise OpRangeError("qwen3: an odd kv-head count does not split into ain elements")
    require("ln", width=spec.hidden)
    require("attn", head_dim=spec.head_dim, num_heads=spec.num_heads, num_kv_heads=spec.num_kv_heads,
            rotary_dim=spec.rotary_dim, rope_theta=spec.rope_theta, qk_norm=spec.qk_norm, attn_gate=spec.attn_gate)
    require("gemv_q4", K=spec.hidden, rs=2, rows_per_core=spec.attn_q_width // n, per_call=PER_CALL)
    require("gemv_q4", K=spec.attn_q_width, rs=2, rows_per_core=spec.hidden // n, per_call=PER_CALL)
    require("gemv_q4", K=spec.hidden, rs=2, rows_per_core=spec.intermediate // n, per_call=PER_CALL)
    require("gemv_q4", K=spec.intermediate, rs=2, rows_per_core=spec.hidden // n, per_call=PER_CALL)
    require("lm_head_q4", K=spec.hidden, vocab=spec.vocab)


def geometry(spec: ModelSpec) -> DenseGeometry:
    n = LIMITS["n_cols"]
    hid, ff, nh, kvh, hd = spec.hidden, spec.intermediate, spec.num_heads, spec.num_kv_heads, spec.head_dim
    qw, kvw = nh * hd, kvh * hd
    e_a = kvw * 2
    hpe = e_a // (hd * 4)                     # q/k/v heads (f32) per ain element = KVH/2
    hpo = e_a // (hd * 2)                     # og heads (bf16) per aout element = KVH
    wide = max(hid, qw, ff)
    return DenseGeometry(
        N_CORES=n, HID=hid, FF=ff, NH=nh, KVH=kvh, HD=hd, ROT=spec.rotary_dim, GATE=spec.attn_gate,
        QW=qw, KVW=kvw,
        Q_PC=qw // BAND_ROWS // n, KV_PC=kvw // BAND_ROWS // n, O_PC=hid // BAND_ROWS // n,
        UP_PC=ff // BAND_ROWS // n, DOWN_PC=hid // BAND_ROWS // n,
        XN_ELEMS=roundup(hid * 2, ELEM) // ELEM, XN_BLOCKS=hid // 32,
        OG_ELEMS=roundup(qw * 2, ELEM) // ELEM, OG_BLOCKS=qw // 32,
        H_ELEMS=roundup(ff * 4, ELEM) // ELEM, H_BLOCKS=ff // 32,
        XM_ELEMS=roundup(hid * 2, ELEM) // ELEM,
        HPE=hpe, HPO=hpo, Q_AIN_ELEMS=nh // hpe, K_AIN_ELEMS=kvh // hpe, OG_AOUT_ELEMS=nh // hpo,
        TAB_BYTES=tab_bytes(wide), KWIDE=wide,
        MS_U=0, MS_G=BAND_ROWS, MS_FLOATS=2 * BAND_ROWS,
    )


def layout(spec: ModelSpec, max_ctx: int = 4096) -> DenseLayout:
    n = LIMITS["n_cols"]
    hid, ff = spec.hidden, spec.intermediate
    G = geometry(spec)
    eln, e_a = hid * 2, G.KVW * 2
    if 2 * G.HD * 2 > e_a or 512 + 4 * spec.rotary_dim > max(1024, e_a):
        raise OpRangeError("qwen3: qn | kn or the RoPE record do not fit the attention element")
    # consts
    c = {"lnw": 0, "postln": eln, "meta": 2 * eln}
    cd_bytes = roundup(2 * eln + e_a, ELEM)
    # act
    a: dict[str, int] = {}
    off = 0
    for name, size in (("xn", G.XN_ELEMS * ELEM), ("q", G.QW * 4), ("kvn", 2 * G.KVW * 4), ("og", G.QW * 2),
                       ("out", hid * 4), ("res", hid * 4), ("xm", G.XM_ELEMS * ELEM), ("h", G.H_ELEMS * ELEM),
                       ("out2", hid * 4), ("junk", eln)):
        a[name] = off
        off += size
    ad_bytes = roundup(off, ELEM)
    # pool
    p: dict[str, int] = {}
    off = 0
    for name, rows, cols in (("q", G.QW, hid), ("k", G.KVW, hid), ("v", G.KVW, hid), ("o", hid, G.QW),
                             ("up", ff, hid), ("gate", ff, hid), ("down", hid, ff)):
        p[name] = off
        off += q4_bytes(rows, cols)
    pool_bytes = roundup(off, MB)
    kv_row = 2 * e_a
    ptab_row = max(1024, e_a)
    band = band_bytes(hid)
    bands = spec.vocab // BAND_ROWS
    return DenseLayout(
        CD_LNW=c["lnw"], CD_POSTLN=c["postln"], CD_META=c["meta"], CD_BYTES=cd_bytes,
        AD_XN=a["xn"], AD_Q=a["q"], AD_KVN=a["kvn"], AD_OG=a["og"], AD_OUT=a["out"], AD_RES=a["res"], AD_XM=a["xm"],
        AD_H=a["h"], AD_OUT2=a["out2"], AD_JUNK=a["junk"], AD_BYTES=ad_bytes,
        POOL_Q=p["q"], POOL_K=p["k"], POOL_V=p["v"], POOL_O=p["o"], POOL_UP=p["up"], POOL_GATE=p["gate"],
        POOL_DOWN=p["down"], POOL_BYTES=pool_bytes,
        KV_ROW=kv_row, PTAB_ROW=ptab_row, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row, PTAB_BYTES=max_ctx * ptab_row,
        LMHEAD_POOL_BYTES=roundup(roundup(bands, n) * band, MB), LMHEAD_BANDS=bands, LMHEAD_BAND_BYTES=band,
        ELN=eln, E_A=e_a,
    )


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> DenseRecipe:
    _check(spec)
    return DenseRecipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def pack_plan(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    hid, ff = spec.hidden, spec.intermediate
    pre = "model.layers.{l}."
    return {
        "pool_bytes": L.POOL_BYTES, "chunk_bytes": CHUNK,
        "layer_types": {DENSE: {
            "pool": [
                {"op": "std_perm", "tensor": pre + "self_attn.q_proj.weight", "dst": L.POOL_Q, "nch": q4_chunks(G.QW, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.k_proj.weight", "dst": L.POOL_K, "nch": q4_chunks(G.KVW, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.v_proj.weight", "dst": L.POOL_V, "nch": q4_chunks(G.KVW, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.o_proj.weight", "dst": L.POOL_O, "nch": q4_chunks(hid, G.QW), "in_dim": G.QW},
                {"op": "std_perm", "tensor": pre + "mlp.up_proj.weight", "dst": L.POOL_UP, "nch": q4_chunks(ff, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "mlp.gate_proj.weight", "dst": L.POOL_GATE, "nch": q4_chunks(ff, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "mlp.down_proj.weight", "dst": L.POOL_DOWN, "nch": q4_chunks(hid, ff), "in_dim": ff},
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CD_LNW, "cap": L.ELN},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CD_POSTLN, "cap": L.ELN},
                {"op": "put", "tensor": pre + "self_attn.q_norm.weight", "dst": L.CD_META, "cap": G.HD * 2},
                {"op": "put", "tensor": pre + "self_attn.k_norm.weight", "dst": L.CD_META + G.HD * 2, "cap": G.HD * 2},
            ],
        }},
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "std_perm", "tensor": "lm_head.weight", "dst": 0, "nch": q4_chunks(spec.vocab, hid), "in_dim": hid}]},
        "embed": {"tensor": "model.embed_tokens.weight", "dim": hid},
        "norm": {"tensor": "model.norm.weight", "bytes": hid * 2},
    }


def programs(spec: ModelSpec) -> dict:
    L = layout(spec)
    args = ["pool", "xres", "consts", "state", "act", "ptab"]
    check_buffer_args("dx", args)
    return {
        "contexts": {"dx": "dx/final.xclbin", "ln": "ln/final.xclbin", "lm": "lm_head_q4/final.xclbin"},
        "kernels": {"dx": {"context": "dx", "insts": "dx/insts.bin", "patch": "attnpos", "build": "dx"},
                    "ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
                    "lm": {"context": "lm", "insts": "lm_head_q4/insts.bin", "build": "lm_head_q4"}},
        "layer_types": {DENSE: {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": "dx", "args": args}],
        }},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": spec.hidden * 4, "zero": spec.hidden * 4, "normw": spec.hidden * 2,
                    "xresf": spec.hidden * 4, "hn": spec.hidden * 2, "logits": spec.vocab * 4,
                    "lmpool": L.LMHEAD_POOL_BYTES, "ptab": {"per_row": L.PTAB_ROW}},
    }


def builds(spec: ModelSpec) -> dict[str, dict]:
    n = LIMITS["n_cols"]
    return {
        "dx": {"design": "dense/dx.py", "build_dir": f"dense/build_h{spec.hidden}", "env": {}},
        "ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{spec.hidden}", "env": {"LN_N": str(spec.hidden)}},
        "lm_head_q4": {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{spec.vocab}",
                       "env": {"LMHEAD_N": str(spec.vocab), "LMHEAD_K": str(spec.hidden), "LMHEAD_CORES": str(n)}},
    }


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L = layout(spec, max_ctx)
    return {"hidden": spec.hidden, "vocab": spec.vocab, "real_vocab": spec.real_vocab,
            "chunk_bytes": CHUNK, "pool_bytes": L.POOL_BYTES, "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta}


def hf_config_check(spec: ModelSpec) -> dict:
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads, "head_dim": spec.head_dim,
            "intermediate_size": spec.intermediate}


KERNEL_SOURCES = [
    "designs/dense/*.py", "designs/dense/*.cc", "designs/dense/*.h",
    "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h", "designs/gemv_q4/gemv_q4_prep_rt.cc",
    "designs/gemv_q4/gemv_q4_prep_f32_rt.cc",
    "designs/attn/*.cc", "designs/attn/*.h",
    "designs/ln/ln.cc", "designs/ln/ln.py", "designs/lin_layer/ln_nr.cc",
    "designs/lm_head_q4/*.py", "designs/lm_head_q4/*.cc",
    "include/vecmath.h", "ironutil.py", "build_design.py",
]
