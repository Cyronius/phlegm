"""Recursive Flint (OPEN-FAMILY-FLINT) on the open engine, from the GGUF.

A layer is two dispatches with the router on the host between them:

  fl_att  (designs/flint/fl.py, FL_PART=att; attnpos):
          ln -> q | k | v -> per-head RMSNorm of q and k -> RoPE (the sliding layers; the full
          layers' position table is the identity rotation) -> attention with the learned sink
          (a 513-row window on the sliding layers) -> o -> res = x + o; xm = ffn_norm(res)
  flint_route (host): sigmoid scores of the F32 router [256, 2048] over xm, top six,
          renormalised; slots 6 and 7 are the shared expert's two halves (pool experts 256 and
          257) at weight 1; the weights go into act beside xm, the eight experts into fl_moe's
          weight fills (moeroute2's patch)
  fl_moe  (FL_PART=moe; moeroute2): per slot, gate | up -> w_e * silu(gate) * up -> h_e; down
          sums the eight slots in each core's accumulator; xres = res + down

The two leading dense layers run fl_att_d (the same core program over their own pool layout)
and fl_dense: the 6144-wide MLP as twelve 512-row slots of the same expert program at weight 1,
with no route. Then the final norm (`ln`) and the q4 head (`lm_head_q4`).

Weights. The GGUF is RecursiveSI's flint-int4-q4_1.gguf: every matrix Q4_1 -- "the MLX export's
trained INT4 codes", whose fp16 scales and mins are bf16 values (checked 2026-10-31) -- so the
q4_1 pool law (5120-B chunks, d and m as bf16; pools.cpp std_perm_gguf, q4_1_pack.py) holds them
EXACTLY, codes kept. The GEMVs are gemv_q4.h's (the dense dx's). The embedding (F32) stays on the
host; norms, q / k norms and sinks go to bf16 in the consts.

Widths need no padding: hidden 2048 and the expert width 512 are whole 64-row bands over 8 cores
and whole 256-wide k-tiles. The shared expert (1024) is two 512-row experts: SwiGLU is per row and
down sums over rows, so the split is exact.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from .catalogue import OpRangeError, check_buffer_args
from .qwen36moe import BAND_ROWS, ELEM, MB, roundup, tab_bytes
from .spec import DENSE, DENSE_LOCAL, ModelSpec

CH = 5120                 # a q4_1 chunk: 32 rows x 256 K, bf16 d | m planes (gemv_q4.h)
PER_CALL = 2              # chunks per w element (10240 B)
N_CORES = 8
TOPK_MAX = 6              # routed slots the route record and the slot table are built for
SHARED_SLOTS = 2          # the shared expert as two expert-width slots
SLOTS = TOPK_MAX + SHARED_SLOTS
ATTN_CORES = 4            # attnknobs.attn_cores(32): the largest divisor of 32 heads <= 6
STACK = 0x1800
LM_CORES = 8
# layer kinds: the attention form x the FFN form
KIND_DENSE, KIND_FULL, KIND_LOCAL = "dense", "full", "local"


def band_bytes(K: int) -> int:
    """64 rows x K: K/256 k-tiles x 2 thirty-two-row chunks of CH bytes."""
    return K // 128 * CH


@dataclass(frozen=True)
class Geometry:
    HID: int; FF: int; DFF: int; SFF: int
    NH: int; KVH: int; HD: int; ROT: int; QW: int; KVW: int
    N_CORES: int; PER_CALL: int; CALL_BYTES: int
    Q_PC: int; KV_PC: int; O_PC: int; UP_PC: int; DOWN_PC: int
    E_A: int; HPE: int; HPO: int; Q_AIN_ELEMS: int; K_AIN_ELEMS: int
    XN_ELEMS: int; XM_ELEMS: int; OG_ELEMS: int; H_ELEMS: int; H_STRIDE: int
    TAB_BYTES: int; MS_U: int; MS_G: int; MS_FLOATS: int
    EPS: float; WINDOW: int; TOPK: int; EXPERTS: int; SLOTS: int; DENSE_SLOTS: int; POOL_EXPERTS: int
    ACORES: int; NHL: int; VEXP: int; MLS: int; RB: int
    PTAB_ELEMS: int; W_OFF: int


@dataclass(frozen=True)
class Layout:
    CD_LNW: int; CD_POSTLN: int; CD_META: int; CD_SINK: int; CD_BYTES: int
    AD_XN: int; AD_Q: int; AD_KVN: int; AD_OG: int; AD_OUT: int; AD_RES: int; AD_XM: int; AD_W: int
    AD_H: int; AD_OUT2: int; AD_JUNK: int; AD_BYTES: int
    UP_BYTES: int; EXPERT_UG: int
    KV_ROW: int; PTAB_ROW: int; MAX_CTX: int; KV_BYTES: int; PTAB_BYTES: int
    LM_ROWS: int; LMHEAD_BANDS: int; LMHEAD_BAND_BYTES: int; LMHEAD_POOL_BYTES: int
    ELN: int


@dataclass(frozen=True)
class Pool:
    """One layer kind's pool: `slots` expert-shaped slots from byte 0 (gate | up per band, then the
    down slices), the attention after them."""
    SLOTS: int; POOL_DOWN: int; POOL_ATTN: int; POOL_Q: int; POOL_K: int; POOL_V: int; POOL_O: int
    POOL_BYTES: int


@dataclass(frozen=True)
class FlintRecipe:
    spec: ModelSpec
    layout: Layout
    geo: Geometry
    max_ctx: int
    kind: str = "flint"


def _check(spec: ModelSpec) -> None:
    if spec.family != "flint":
        raise OpRangeError(f"flint recipe given a {spec.family!r} spec")
    if set(spec.layer_types) - {DENSE, DENSE_LOCAL}:
        raise OpRangeError("flint: layers must be sliding (dense_local) or full (dense)")
    if spec.activation != "silu" or not spec.num_experts or not spec.moe_intermediate:
        raise OpRangeError("flint: expected routed SiLU experts")
    if spec.experts_per_tok != TOPK_MAX:
        raise OpRangeError(f"flint: top-{spec.experts_per_tok} routing (the slot table is built for {TOPK_MAX})")
    if spec.shared_expert_intermediate != SHARED_SLOTS * spec.moe_intermediate:
        raise OpRangeError(f"flint: a shared expert of {spec.shared_expert_intermediate} is not {SHARED_SLOTS} "
                           f"expert widths ({spec.moe_intermediate})")
    if spec.dense_lead and (not spec.intermediate or spec.intermediate % spec.moe_intermediate):
        raise OpRangeError(f"flint: the dense MLP {spec.intermediate} is not whole expert widths")
    if any(spec.layer_types[i] != DENSE for i in range(spec.dense_lead)):
        raise OpRangeError("flint: the dense-MLP layers must be full-attention layers (one dense pool layout)")
    if not spec.qk_norm or spec.attn_gate:
        raise OpRangeError("flint: q/k norms and no attention gate")
    if spec.rotary_dim != spec.head_dim or spec.rope_scaling:
        raise OpRangeError("flint: a full-head default RoPE")
    if spec.num_kv_heads % 2 or spec.num_heads % spec.num_kv_heads or spec.num_heads % ATTN_CORES:
        raise OpRangeError(f"flint: {spec.num_heads} q / {spec.num_kv_heads} kv heads do not split "
                           f"over the attention elements and {ATTN_CORES} cores")
    qw, kvw = spec.num_heads * spec.head_dim, spec.num_kv_heads * spec.head_dim
    for what, w in (("hidden", spec.hidden), ("q width", qw), ("kv width", kvw), ("expert width", spec.moe_intermediate)):
        if w % (BAND_ROWS * N_CORES) or w % 256:
            raise OpRangeError(f"flint: {what} {w} is not whole 64-row bands over {N_CORES} cores and 256-wide k-tiles")
    if 4 * spec.head_dim + 2 * spec.num_heads > 2 * kvw:
        raise OpRangeError("flint: qn | kn | sinks do not fit one meta element (attn.h ATTN_SINK)")


def geometry(spec: ModelSpec) -> Geometry:
    _check(spec)
    hid, ff = spec.hidden, spec.moe_intermediate
    nh, kvh, hd = spec.num_heads, spec.num_kv_heads, spec.head_dim
    qw, kvw = nh * hd, kvh * hd
    e_a = kvw * 2
    ptab_row = max(1024, e_a)
    nhl = nh // ATTN_CORES
    xm_bytes = hid * 2
    xm_elems = roundup(xm_bytes + 4 * SLOTS, ELEM) // ELEM
    dense_slots = spec.intermediate // ff if spec.dense_lead else 0
    return Geometry(
        HID=hid, FF=ff, DFF=spec.intermediate, SFF=spec.shared_expert_intermediate,
        NH=nh, KVH=kvh, HD=hd, ROT=spec.rotary_dim, QW=qw, KVW=kvw,
        N_CORES=N_CORES, PER_CALL=PER_CALL, CALL_BYTES=PER_CALL * CH,
        Q_PC=qw // BAND_ROWS // N_CORES, KV_PC=kvw // BAND_ROWS // N_CORES, O_PC=hid // BAND_ROWS // N_CORES,
        UP_PC=ff // BAND_ROWS // N_CORES, DOWN_PC=hid // BAND_ROWS // N_CORES,
        E_A=e_a, HPE=e_a // (hd * 4), HPO=e_a // (hd * 2), Q_AIN_ELEMS=nh // (e_a // (hd * 4)),
        K_AIN_ELEMS=kvh // (e_a // (hd * 4)),
        XN_ELEMS=roundup(hid * 2, ELEM) // ELEM, XM_ELEMS=xm_elems,
        OG_ELEMS=roundup(qw * 2, ELEM) // ELEM, H_ELEMS=roundup(ff * 4, ELEM) // ELEM, H_STRIDE=roundup(ff * 4, ELEM),
        TAB_BYTES=tab_bytes(max(hid, qw, ff)), MS_U=0, MS_G=BAND_ROWS, MS_FLOATS=3 * BAND_ROWS,   # u | g | a down band

        EPS=spec.norm_eps, WINDOW=spec.sliding_window, TOPK=spec.experts_per_tok, EXPERTS=spec.num_experts,
        SLOTS=SLOTS, DENSE_SLOTS=dense_slots, POOL_EXPERTS=spec.num_experts + SHARED_SLOTS,
        # RB 4 and the vector softmax: GPT-OSS's point (Peano 2026-10 compiles attn_row_impl's blocked walk;
        # granite and GPT-OSS run it). 8 heads a core at head dim 128.
        ACORES=ATTN_CORES, NHL=nhl, VEXP=1, MLS=roundup(nhl, 32), RB=4,
        PTAB_ELEMS=ptab_row // e_a,
        W_OFF=xm_bytes % ELEM,        # the routing weights' byte offset in xm's last x element
    )


def lm_rows(spec: ModelSpec) -> int:
    return roundup(spec.vocab, BAND_ROWS)


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Layout:
    G = geometry(spec)
    eln = G.HID * 2
    cd = {"lnw": 0, "postln": eln, "meta": 2 * eln}
    cd_bytes = roundup(2 * eln + G.E_A, ELEM)
    a: dict[str, int] = {}
    off = 0
    slots_h = max(G.SLOTS, G.DENSE_SLOTS)
    for name, size in (("xn", G.XN_ELEMS * ELEM), ("q", G.QW * 4), ("kvn", 2 * G.KVW * 4), ("og", G.OG_ELEMS * ELEM),
                       ("out", G.HID * 4), ("res", G.HID * 4), ("xm", G.XM_ELEMS * ELEM),
                       ("h", slots_h * G.H_STRIDE), ("out2", G.HID * 4), ("junk", eln)):
        a[name] = off
        off += size
    ad_bytes = roundup(off, ELEM)
    up = (G.FF // BAND_ROWS) * band_bytes(G.HID)       # one slot's gate (= up) projection
    down = (G.HID // BAND_ROWS) * band_bytes(G.FF)
    if up != down:
        raise OpRangeError("flint: a slot's gate and down must be the same size for the routed-expert patch (moe2)")
    rows = lm_rows(spec)
    bands = rows // BAND_ROWS
    lm_band = band_bytes(G.HID)
    kv_row = 2 * G.E_A
    ptab_row = G.PTAB_ELEMS * G.E_A
    return Layout(
        CD_LNW=cd["lnw"], CD_POSTLN=cd["postln"], CD_META=cd["meta"], CD_SINK=cd["meta"] + 4 * G.HD,
        CD_BYTES=cd_bytes,
        AD_XN=a["xn"], AD_Q=a["q"], AD_KVN=a["kvn"], AD_OG=a["og"], AD_OUT=a["out"], AD_RES=a["res"],
        AD_XM=a["xm"], AD_W=a["xm"] + (G.XM_ELEMS - 1) * ELEM + G.W_OFF, AD_H=a["h"], AD_OUT2=a["out2"],
        AD_JUNK=a["junk"], AD_BYTES=ad_bytes,
        UP_BYTES=up, EXPERT_UG=2 * up,
        KV_ROW=kv_row, PTAB_ROW=ptab_row, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row, PTAB_BYTES=max_ctx * ptab_row,
        LM_ROWS=rows, LMHEAD_BANDS=bands, LMHEAD_BAND_BYTES=lm_band,
        LMHEAD_POOL_BYTES=roundup(roundup(bands, LM_CORES) * lm_band, MB),
        ELN=eln,
    )


def pool(spec: ModelSpec, dense: bool) -> Pool:
    """The MoE layers' pool (the 256 routed experts and the shared expert's two halves) or the
    dense layers' (the MLP's slots)."""
    G, L = geometry(spec), layout(spec)
    slots = G.DENSE_SLOTS if dense else G.POOL_EXPERTS
    pool_down = slots * L.EXPERT_UG
    p_attn = pool_down + slots * L.UP_BYTES
    p = {"q": p_attn}
    p["k"] = p["q"] + (G.QW // BAND_ROWS) * band_bytes(G.HID)
    p["v"] = p["k"] + (G.KVW // BAND_ROWS) * band_bytes(G.HID)
    p["o"] = p["v"] + (G.KVW // BAND_ROWS) * band_bytes(G.HID)
    pool_bytes = roundup(p["o"] + (G.HID // BAND_ROWS) * band_bytes(G.QW), MB)
    if pool_bytes >= 1 << 32:
        raise OpRangeError("flint: a layer pool past 4 GiB does not fit a 32-bit fill offset")
    return Pool(SLOTS=slots, POOL_DOWN=pool_down, POOL_ATTN=p_attn, POOL_Q=p["q"], POOL_K=p["k"], POOL_V=p["v"],
                POOL_O=p["o"], POOL_BYTES=pool_bytes)


def pool_bytes(spec: ModelSpec) -> int:
    """The one pool BO size every layer gets (the MoE layers'; the dense layers use its head)."""
    return pool(spec, dense=False).POOL_BYTES


def kinds(spec: ModelSpec) -> list[str]:
    """Each layer's kind: dense (a leading dense-MLP layer, full attention), full or local."""
    return [KIND_DENSE if i < spec.dense_lead else KIND_LOCAL if t == DENSE_LOCAL else KIND_FULL
            for i, t in enumerate(spec.layer_types)]


def layer_type_names(spec: ModelSpec) -> list[str]:
    """The manifest's per-layer type names: `dense` for the dense-MLP layers, else the spec's own."""
    return ["dense_mlp" if k == KIND_DENSE else t for k, t in zip(kinds(spec), spec.layer_types)]


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> FlintRecipe:
    pool(spec, dense=False)
    if spec.dense_lead:
        pool(spec, dense=True)
    return FlintRecipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def attn_flags(G: Geometry, probe_env=()) -> list[str]:
    """attn.h for Flint: Qwen3's q / k norm before the rotation, no gate, the sink, the vector softmax
    over 4 cores of 8 heads."""
    flags = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
             "-DATTN_QKNORM=1", "-DATTN_QKNORM_POST=0", f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}",
             f"-DATTN_NHL={G.NHL}", "-DATTN_SINK=1"]
    if G.RB > 1:
        flags.append(f"-DATTN_RB={G.RB}")
    if G.PTAB_ELEMS > 1:
        flags.append("-DATTN_PTAB_SPLIT=1")
    for k, v in probe_env:
        flags.append(f"-D{k}={v}")
    return flags


def ln_flags(G: Geometry) -> list[str]:
    return [f"-DLN_N={G.HID}", f"-DLN_EPS={G.EPS:g}f"]


PROBE_VARS = ("ATTN_NULL", "FL_NULL")


def probe_env() -> dict[str, str]:
    return {k: os.environ[k] for k in PROBE_VARS if os.environ.get(k)}


def gguf_quant(spec: ModelSpec) -> str:
    """The law the GGUF view's pools hold: q4_1 (the spec's own; the GGUF's Q4_1 exactly)."""
    return "q4_1"


def finalize_gguf_manifest(spec: ModelSpec, manifest: dict) -> None:
    """The manifest IS the GGUF view (there is no q4nx container for Flint): core.cpp reads a
    model.gguf through the nested `gguf` section, so the section is the same manifest, nested once.
    And the per-layer type names: the dense-MLP layers are their own type (`dense_mlp`), which the
    spec's attention-only layer_types cannot say."""
    manifest["layers"] = layer_type_names(spec)
    if "gguf" not in manifest:
        manifest["gguf"] = copy.deepcopy(manifest)


def _q4_op(tensor: str, dst: int, rows: int, in_dim: int, row0: int = 0, col0: int = 0, experts: int = 1,
           row_step: int = 0, col_step: int = 0, expert_bytes: int = 0, band_stride: int = 0) -> dict:
    """runtime pack op fl_q4 (flint-runtime.patch): for e < experts, the rows x in_dim block at
    (row0 + e row_step, col0 + e col_step) of a Q4_0 / Q4_1 GGUF tensor (the trailing dims flattened
    into rows) as 64-row bands of q4_1 chunks (std_perm_gguf's byte law, scale and min bf16 RNE),
    band b at dst + e expert_bytes + b band_stride (band_stride 0: the bands contiguous)."""
    if rows % BAND_ROWS or in_dim % 256 or col0 % 32 or col_step % 32:
        raise OpRangeError(f"flint: {tensor}: {rows} x {in_dim} at column {col0} is not whole bands and k-tiles")
    return {"op": "fl_q4", "tensor": tensor, "dst": dst, "rows": rows, "in_dim": in_dim, "row0": row0,
            "chunk0": col0, "experts": experts, "n_total": row_step, "cols": col_step,
            "expert_bytes": expert_bytes, "stripe_bytes": band_stride}


def _layer_plan(spec: ModelSpec, dense: bool) -> dict:
    G, L, P = geometry(spec), layout(spec), pool(spec, dense)
    b = "blk.{l}."
    ff, hid, bbh = G.FF, G.HID, band_bytes(G.HID)
    ug = dict(expert_bytes=L.EXPERT_UG, band_stride=2 * bbh)
    if dense:
        n = G.DENSE_SLOTS
        ffn = [  # the MLP's rows (gate, up) and columns (down) in 512-wide slots
            _q4_op(b + "ffn_gate.weight", 0, ff, hid, experts=n, row_step=ff, **ug),
            _q4_op(b + "ffn_up.weight", bbh, ff, hid, experts=n, row_step=ff, **ug),
            _q4_op(b + "ffn_down.weight", P.POOL_DOWN, hid, ff, experts=n, col_step=ff, expert_bytes=L.UP_BYTES),
        ]
    else:
        E, s0 = spec.num_experts, spec.num_experts     # the shared halves are pool experts E, E + 1
        ffn = [
            _q4_op(b + "ffn_gate_exps.weight", 0, ff, hid, experts=E, row_step=ff, **ug),
            _q4_op(b + "ffn_up_exps.weight", bbh, ff, hid, experts=E, row_step=ff, **ug),
            _q4_op(b + "ffn_down_exps.weight", P.POOL_DOWN, hid, ff, experts=E, row_step=hid,
                   expert_bytes=L.UP_BYTES),
            _q4_op(b + "ffn_gate_shexp.weight", s0 * L.EXPERT_UG, ff, hid, experts=SHARED_SLOTS, row_step=ff, **ug),
            _q4_op(b + "ffn_up_shexp.weight", s0 * L.EXPERT_UG + bbh, ff, hid, experts=SHARED_SLOTS, row_step=ff, **ug),
            _q4_op(b + "ffn_down_shexp.weight", P.POOL_DOWN + s0 * L.UP_BYTES, hid, ff, experts=SHARED_SLOTS,
                   col_step=ff, expert_bytes=L.UP_BYTES),
        ]
    attn = [_q4_op(b + "attn_q.weight", P.POOL_Q, G.QW, hid), _q4_op(b + "attn_k.weight", P.POOL_K, G.KVW, hid),
            _q4_op(b + "attn_v.weight", P.POOL_V, G.KVW, hid), _q4_op(b + "attn_output.weight", P.POOL_O, hid, G.QW)]
    consts = [
        {"op": "put", "tensor": b + "attn_norm.weight", "dst": L.CD_LNW, "cap": L.ELN},
        {"op": "put", "tensor": b + "ffn_norm.weight", "dst": L.CD_POSTLN, "cap": L.ELN},
        {"op": "put", "tensor": b + "attn_q_norm.weight", "dst": L.CD_META, "cap": 2 * G.HD},
        {"op": "put", "tensor": b + "attn_k_norm.weight", "dst": L.CD_META + 2 * G.HD, "cap": 2 * G.HD},
        {"op": "put", "tensor": b + "attn_sinks.weight", "dst": L.CD_SINK, "cap": 2 * G.NH},
    ]
    return {"pool": ffn + attn, "consts": consts}


def pack_plan(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    names = layer_type_names(spec)
    out = {}
    for nm, k in zip(names, kinds(spec)):
        out.setdefault(nm, _layer_plan(spec, k == KIND_DENSE))
    return {
        "pool_bytes": pool_bytes(spec), "chunk_bytes": CH,
        "layer_types": out,
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "std_perm_gguf", "tensor": "output.weight", "dst": 0,
                             "nch": L.LM_ROWS // 32 * (G.HID // 256), "in_dim": G.HID}]},
        "embed": {"tensor": "token_embd.weight", "dim": G.HID},
        "norm": {"tensor": "output_norm.weight", "bytes": L.ELN},
    }


def route_step(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    return {"op": "flint_route", "kernel": "fl_moe", "a_xm": L.AD_XM, "a_w": L.AD_W, "hidden_real": G.HID,
            "router": "blk.{l}.ffn_gate_inp.weight", "experts": spec.num_experts, "topk": spec.experts_per_tok,
            "shared": [spec.num_experts + i for i in range(SHARED_SLOTS)]}


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    out = {
        "contexts": {"fl_att": "fl_att/final.xclbin", "fl_moe": "fl_moe/final.xclbin",
                     "ln": "ln/final.xclbin", "lm": "lm_head_q4/final.xclbin"},
        "kernels": {
            "fl_moe": {"context": "fl_moe", "insts": "fl_moe/insts.bin", "patch": "moeroute2", "build": "fl_moe"},
            "ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
            "lm": {"context": "lm", "insts": "lm_head_q4/insts.bin", "build": "lm_head_q4"},
        },
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": G.HID * 4, "zero": G.HID * 4, "normw": G.HID * 2, "xresf": G.HID * 4, "hn": G.HID * 2,
                    "logits": L.LM_ROWS * 4, "lmpool": L.LMHEAD_POOL_BYTES},
    }
    if spec.dense_lead:
        out["contexts"].update({"fl_att_d": "fl_att_d/final.xclbin", "fl_dense": "fl_dense/final.xclbin"})
        out["kernels"]["fl_dense"] = {"context": "fl_dense", "insts": "fl_dense/insts.bin", "build": "fl_dense"}
    for nm, k in zip(layer_type_names(spec), kinds(spec)):
        if nm in out["layer_types"]:
            continue
        local = k == KIND_LOCAL
        build = "fl_att_d" if k == KIND_DENSE else "fl_att"
        kn = {KIND_DENSE: "fl_att_d", KIND_FULL: "fl_att", KIND_LOCAL: "fl_att_local"}[k]
        tab = "ptab_local" if local else "ptab"
        window = spec.sliding_window if local else 0
        args = ["pool", "xres", "consts", "state", "act", tab]
        check_buffer_args(kn, args)
        out["kernels"][kn] = {"context": build, "insts": f"{build}/insts.bin", "patch": "attnpos", "build": build,
                              "window": window}
        # The sliding layers rotate at theta^(-2i/d); the full layers do not rotate at all (NoPE), which
        # is the same kernel over a table of zero frequencies: cos 1, sin 0 is the identity exactly.
        inv = spec.rope_inv_freq() if local else [0.0] * (G.ROT // 2)
        out["globals"][tab] = {"per_row": L.PTAB_ROW, "inv_freq": inv, "window": window}
        if k == KIND_DENSE:
            prog = [{"op": "run", "kernel": kn, "args": args},
                    {"op": "run", "kernel": "fl_dense", "args": ["pool", "xres", "act"]}]
        else:
            prog = [{"op": "run", "kernel": kn, "args": args}, route_step(spec),
                    {"op": "run", "kernel": "fl_moe", "args": ["pool", "xres", "act"]}]
        out["layer_types"][nm] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": prog,
        }
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    G, L = geometry(spec), layout(spec)
    out = {
        "fl_att": {"design": "flint/fl.py", "build_dir": f"flint/build_att_h{G.HID}",
                   "env": {"FL_PART": "att", "FL_POOL": "moe"}},
        "fl_moe": {"design": "flint/fl.py", "build_dir": f"flint/build_moe_h{G.HID}_s{G.SLOTS}",
                   "env": {"FL_PART": "moe", "FL_POOL": "moe"}},
        "ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{G.HID}_{G.EPS:g}",
               "env": {"LN_N": str(G.HID), "LN_EPS": f"{G.EPS:g}"}},
        "lm_head_q4": {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{L.LM_ROWS}_k{G.HID}",
                       "env": {"LMHEAD_N": str(L.LM_ROWS), "LMHEAD_K": str(G.HID), "LMHEAD_CORES": str(LM_CORES)}},
    }
    if spec.dense_lead:
        out["fl_att_d"] = {"design": "flint/fl.py", "build_dir": f"flint/build_att_h{G.HID}_dense",
                           "env": {"FL_PART": "att", "FL_POOL": "dense"}}
        out["fl_dense"] = {"design": "flint/fl.py", "build_dir": f"flint/build_moe_h{G.HID}_s{G.DENSE_SLOTS}_dense",
                           "env": {"FL_PART": "moe", "FL_POOL": "dense"}}
    return out


def moe_geometry(spec: ModelSpec) -> dict:
    """stream_patch::MoeGeometry for moeroute2 over fl_moe's fills: slot j is compiled as pool expert j,
    and the pool holds the routed experts and the shared halves (num_experts + 2) from byte 0."""
    L, P = layout(spec), pool(spec, dense=False)
    return {"stripe": L.UP_BYTES, "up_bytes": L.UP_BYTES, "down_core": L.UP_BYTES, "pool_down": P.POOL_DOWN,
            "share_up": P.POOL_ATTN, "share_gate": P.POOL_ATTN, "share_down": P.POOL_ATTN,
            "experts": P.SLOTS, "topk": SLOTS}


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    return {"hidden": G.HID, "vocab": L.LM_ROWS, "real_vocab": spec.real_vocab, "chunk_bytes": CH,
            "pool_bytes": pool_bytes(spec), "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": CH,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
            "rope_inv_freq": spec.rope_inv_freq(), "moe": moe_geometry(spec)}


def hf_config_check(spec: ModelSpec) -> dict:
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim, "intermediate_size": spec.intermediate,
            "moe_intermediate_size": spec.moe_intermediate, "num_experts": spec.num_experts,
            "num_experts_per_tok": spec.experts_per_tok,
            "shared_expert_intermediate_size": spec.shared_expert_intermediate,
            "sliding_window": spec.sliding_window, "rope_theta": spec.rope_theta,
            "softmax_type": "learnable", "router_score_function": "sigmoid"}


GEN_KERNELS = "designs/flint/gen_kernels.py"
KERNEL_SOURCES = [
    "designs/flint/*.py", "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h",
    "designs/attn/*.cc", "designs/attn/*.h", "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py",
    "designs/lin_layer/ln_nr.cc", "designs/lm_head_q4/*.py", "designs/lm_head_q4/*.cc", "designs/gemv_q4/gemv_q4_prep_rt.cc",
    "include/vecmath.h", "include/scalar_fp.h", "ironutil.py", "build_design.py",
]
