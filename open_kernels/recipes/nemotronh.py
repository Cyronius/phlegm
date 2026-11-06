"""NVIDIA Nemotron-H with routed experts (OPEN-FAMILY-NEMOTRONH; Nemotron 3.5 Lightning 30B-A3B) on the open engine,
from the GGUF.

Every block is `x = x + mixer(rms(x) * norm)` with one of three mixers. The NPU streams the weights -- every
projection is a q4 GEMV, the experts one expert program -- and the host does everything between them, so a block
is a short program of host steps around one or two dispatches (nemotronh-runtime.patch):

  mamba      nh_pre (rms -> bf16 x) | nh_in: in_proj GEMV [10304 = z | xBC | dt] | nh_ssm (host: causal conv +
             bias + silu, softplus dt, the 64-head selective-state update S = S exp(dt A) + dt x B^T, y = S C +
             D x, y * silu(z), the 8-group RMSNorm -> bf16) | nh_out: out_proj GEMV | nh_add (x += o)
  attention  nh_pre | nh_qkv: q | k | v GEMV [4608] | nh_attn (host: causal GQA attention over the layer's f32
             K / V rows, NO positional encoding) | nh_out: o GEMV | nh_add
  moe        nh_route (host: rms -> bf16 xm; F32 router, sigmoid, the top 6 of score + e_score_correction_bias,
             the unbiased scores renormalised x 2.5; slots 6 and 7 = the shared expert's two halves at weight 1;
             the slots' expert offsets into nh_moe's fills, `nhroute`) | nh_moe: per slot, up -> w relu(up)^2
             -> h, then down summed over the 8 slots in each core's accumulator | nh_add
  tail       nh_final (host: rms * output_norm -> bf16 hn) | lm: the Q8_0 head (lm_head_q8, f32 scales)

The mamba out_proj and the attention o projection are the same shape ([hidden, 4096]) at the same pool offset, so
one `nh_out` build serves both. Recurrent state: the layer's `linear` state BO holds the conv's 3 previous xBC rows
and the f32 SSM state [64, 64, 128]; attention keeps f32 [K | V] rows in its `kv` state. Both live on the host
side of their BOs (no kernel reads them), so checkpoint / restore / reset handle them as every other family's.

Widths. hidden 2688 is 42 bands of 64 rows but 10.5 k-tiles of 256, so every K = hidden operand is padded to
2816 with zero weights (and the activation with zeros: exact); the expert width 1856 (29 bands) is padded to 2048
as the down projection's K. Output bands are split unevenly over the 8 cores (lm_head_q8's split_bands), so no
output rows are padded. The shared expert (3712 = 2 x 1856) runs as two more slots whose weights are its row
halves (up) and column halves (down) -- exact, relu^2 is elementwise.

Weights. The GGUF is our own pure Q4_0 from NVIDIA's BF16 safetensors (llama quantize --pure, the head Q8_0;
.plans/nemotronh.md). From a BF16 source every Q4_0 scale is a block max / -8, a bf16 value, so the Q4_0-native
pool law (4608-B chunks, the bf16 d plane, m = -8 d on the core; `nh_q40`) holds every matrix exactly (checked
over the file). The router, its bias, the norms and the Mamba constants are F32 tensors the host reads from the
GGUF; the embedding (Q4_0) stays on the host (embed_row).
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from .catalogue import OpRangeError
from .qwen36moe import BAND_ROWS, ELEM, MB, roundup, tab_bytes
from .spec import FULL, MAMBA, MOE_BLOCK, ModelSpec

CH = 4608                 # a Q4_0-native chunk: 32 rows x 256 K, the bf16 d plane then the codes (gemv_q4.h)
PER_CALL = 2              # chunks per w element (9216 B)
N_CORES = 8
SLOTS_MAX = 8             # the routing-weight record and the patch table hold 8 slots
STACK = 0x1800
LM_CORES = 8
LM_BAND_ROWS = 128        # lm_head_q8: 128-row bands
LM_TILE = 9216            # lm_head_q8.h's f32-scale chunk: 32 rows x 256 K
STATE_MOE = 64            # a moe block's (unused) state BO: the engine wants one per layer
CONSTS = 4096             # no consts are packed: the host reads the small F32 tensors from the GGUF
KINDS = (MAMBA, FULL, MOE_BLOCK)


def shared_gemv() -> bool:
    """NEMOTRONH_SHARED=1: the three projection GEMVs as streams of ONE image (runtime-parameter shapes, no context
    change between them; R-0003). Off: one image a projection."""
    return os.environ.get("NEMOTRONH_SHARED", "0") == "1"


def kpad(k: int) -> int:
    """A GEMV operand's K: whole 256-wide k-tiles (zero weights past the real width)."""
    return roundup(k, 256)


def band_bytes(K: int) -> int:
    """64 rows x K (K whole k-tiles): K/256 k-tiles x 2 thirty-two-row chunks."""
    return K // 128 * CH


def split_bands(bands: int, n_cores: int = N_CORES) -> list[int]:
    """lm_head_q8's uneven split: the first `bands % n_cores` cores take one band more."""
    q, r = divmod(bands, n_cores)
    return [q + (1 if c < r else 0) for c in range(n_cores)]


@dataclass(frozen=True)
class Geometry:
    HID: int; HIDP: int; EPS: float
    NH: int; KVH: int; HD: int; QW: int; KVW: int; QKV_N: int
    MH: int; MHD: int; MG: int; MDS: int; MDI: int; MCONV: int; IN_N: int; TAPS: int
    FF: int; FFP: int; SH_SLOTS: int; TOPK: int; SLOTS: int; EXPERTS: int; POOL_EXPERTS: int
    OUT_K: int


@dataclass(frozen=True)
class Layout:
    AD_X: int; AD_Y: int; AD_O: int; AD_XM: int; AD_W: int; AD_H: int; AD_MOE: int; AD_BYTES: int
    XE: int; XM_ELEMS: int; H_STRIDE: int
    POOL_IN: int; POOL_OUT: int; POOL_MIX_BYTES: int
    UPS: int; DNS: int; POOL_DOWN: int; POOL_MOE_BYTES: int; POOL_BYTES: int
    STATE_CONV: int; STATE_BYTES: int; KV_ROW: int
    LM_ROWS: int; LMHEAD_POOL_BYTES: int


@dataclass(frozen=True)
class NemotronHRecipe:
    spec: ModelSpec
    layout: Layout
    geo: Geometry
    max_ctx: int
    kind: str = "nemotronh"


def router_scale(spec: ModelSpec) -> float:
    return float(spec.extra.get("router_scale", 1.0))


def _check(spec: ModelSpec) -> None:
    if spec.family != "nemotronh":
        raise OpRangeError(f"nemotronh recipe given a {spec.family!r} spec")
    bad = sorted(set(spec.layer_types) - set(KINDS))
    if bad:
        raise OpRangeError(f"nemotronh: block types {bad} are not mamba / full_attention / moe")
    if spec.activation != "relu2" or not spec.num_experts or not spec.moe_intermediate:
        raise OpRangeError("nemotronh: expected routed relu^2 experts")
    ff, sh = spec.moe_intermediate, spec.shared_expert_intermediate
    if not sh or sh % ff:
        raise OpRangeError(f"nemotronh: the shared expert {sh} is not whole {ff}-wide slots")
    if spec.experts_per_tok + sh // ff > SLOTS_MAX:
        raise OpRangeError(f"nemotronh: top-{spec.experts_per_tok} + {sh // ff} shared slots exceed {SLOTS_MAX}")
    qw = spec.num_heads * spec.head_dim
    di = spec.lin_value_heads * spec.lin_value_dim
    if qw != di:
        raise OpRangeError(f"nemotronh: the attention width {qw} and the Mamba width {di} differ (one out GEMV)")
    if spec.num_heads % spec.num_kv_heads or spec.lin_value_heads % spec.lin_key_heads:
        raise OpRangeError("nemotronh: heads do not group evenly")
    for what, w in (("hidden", spec.hidden), ("q width", qw), ("kv width", spec.num_kv_heads * spec.head_dim),
                    ("expert width", ff), ("Mamba width", di)):
        if w % BAND_ROWS:
            raise OpRangeError(f"nemotronh: {what} {w} is not whole 64-row bands")
    if spec.rotary_dim or spec.qk_norm or spec.attn_gate or spec.sliding_window:
        raise OpRangeError("nemotronh: the attention has no RoPE, no q/k norm, no gate, no window")


def geometry(spec: ModelSpec) -> Geometry:
    _check(spec)
    hid = spec.hidden
    nh, kvh, hd = spec.num_heads, spec.num_kv_heads, spec.head_dim
    mh, mhd, mg, mds = spec.lin_value_heads, spec.lin_value_dim, spec.lin_key_heads, spec.lin_key_dim
    mdi = mh * mhd
    mconv = mdi + 2 * mg * mds
    ff, sh = spec.moe_intermediate, spec.shared_expert_intermediate
    return Geometry(
        HID=hid, HIDP=kpad(hid), EPS=spec.norm_eps,
        NH=nh, KVH=kvh, HD=hd, QW=nh * hd, KVW=kvh * hd, QKV_N=nh * hd + 2 * kvh * hd,
        MH=mh, MHD=mhd, MG=mg, MDS=mds, MDI=mdi, MCONV=mconv, IN_N=mdi + mconv + mh, TAPS=spec.conv_kernel,
        FF=ff, FFP=kpad(ff), SH_SLOTS=sh // ff, TOPK=spec.experts_per_tok, SLOTS=spec.experts_per_tok + sh // ff,
        EXPERTS=spec.num_experts, POOL_EXPERTS=spec.num_experts + sh // ff,
        OUT_K=kpad(mdi),
    )


def lm_rows(spec: ModelSpec) -> int:
    return roundup(spec.vocab, LM_BAND_ROWS)


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Layout:
    G = geometry(spec)
    a: dict[str, int] = {}
    off = 0
    xe = roundup(max(G.HIDP, G.OUT_K) * 2, ELEM) // ELEM
    xm_elems = roundup(G.HIDP * 2 + 4 * SLOTS_MAX, ELEM) // ELEM
    h_stride = roundup(G.FFP * 4, ELEM)
    for name, size in (("x", xe * ELEM), ("y", max(G.IN_N, G.QKV_N) * 4), ("o", G.HID * 4),
                       ("xm", xm_elems * ELEM), ("h", G.SLOTS * h_stride), ("moe", G.HID * 4)):
        a[name] = off
        off += roundup(size, ELEM)
    ad_w = a["xm"] + (xm_elems - 1) * ELEM + 2048      # the 8 routing weights: byte 2048 of xm's last element
    if ad_w < a["xm"] + G.HIDP * 2:
        raise OpRangeError("nemotronh: the routing weights would overlap xm")
    # the mixer pools: the input GEMV (in_proj or q | k | v) at 0, the out GEMV at one offset for both kinds
    bb_h, bb_o, bb_f = band_bytes(G.HIDP), band_bytes(G.OUT_K), band_bytes(G.FFP)
    pool_out = roundup(max(G.IN_N, G.QKV_N) // BAND_ROWS * bb_h, MB)
    pool_mix = roundup(pool_out + G.HID // BAND_ROWS * bb_o, MB)
    # the moe pool: every expert's up at e * UPS (the routed ones, then the shared expert's halves), then the down
    # projections at POOL_DOWN + e * DNS
    ups = G.FF // BAND_ROWS * bb_h
    dns = G.HID // BAND_ROWS * bb_f
    pool_down = G.POOL_EXPERTS * ups
    pool_moe = roundup(pool_down + G.POOL_EXPERTS * dns, MB)
    if pool_moe >= 1 << 32:
        raise OpRangeError("nemotronh: a layer pool past 4 GiB does not fit a 32-bit fill offset")
    conv = (G.TAPS - 1) * G.MCONV * 4
    rows = lm_rows(spec)
    bands = rows // LM_BAND_ROWS
    lm_band = 4 * (G.HIDP // 256) * LM_TILE
    return Layout(
        AD_X=a["x"], AD_Y=a["y"], AD_O=a["o"], AD_XM=a["xm"], AD_W=ad_w, AD_H=a["h"], AD_MOE=a["moe"],
        AD_BYTES=roundup(off, ELEM), XE=xe, XM_ELEMS=xm_elems, H_STRIDE=h_stride,
        POOL_IN=0, POOL_OUT=pool_out, POOL_MIX_BYTES=pool_mix,
        UPS=ups, DNS=dns, POOL_DOWN=pool_down, POOL_MOE_BYTES=pool_moe, POOL_BYTES=max(pool_mix, pool_moe),
        STATE_CONV=conv, STATE_BYTES=conv + G.MH * G.MHD * G.MDS * 4, KV_ROW=2 * G.KVW * 4,
        LM_ROWS=rows, LMHEAD_POOL_BYTES=roundup(roundup(bands, LM_CORES) * lm_band, MB),
    )


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> NemotronHRecipe:
    return NemotronHRecipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def gemv_shapes(spec: ModelSpec) -> dict[str, dict]:
    """The three projection GEMVs: rows, K (padded), the pool offset of the weight, the act regions in / out and
    the pool size of the layer kind it reads (designs/nemotronh/nh.py NH_PART)."""
    G, L = geometry(spec), layout(spec)
    return {
        "in": {"n": G.IN_N, "k": G.HIDP, "pool": L.POOL_IN, "ax": L.AD_X, "ay": L.AD_Y, "pool_bytes": L.POOL_MIX_BYTES},
        "qkv": {"n": G.QKV_N, "k": G.HIDP, "pool": L.POOL_IN, "ax": L.AD_X, "ay": L.AD_Y,
                "pool_bytes": L.POOL_MIX_BYTES},
        "out": {"n": G.HID, "k": G.OUT_K, "pool": L.POOL_OUT, "ax": L.AD_X, "ay": L.AD_O,
                "pool_bytes": L.POOL_MIX_BYTES},
    }


def _q40(tensor: str, dst: int, rows: int, in_dim: int, src_dim: int, row0: int = 0, col0: int = 0,
         experts: int = 1, n_total: int = 0, expert_bytes: int = 0) -> dict:
    """runtime pack op nh_q40 (nemotronh-runtime.patch): for e < experts, `rows` file rows from row0 + e n_total
    and `src_dim` columns from col0 of a Q4_0 GGUF tensor, as Q4_0-native chunks of a [roundup(rows, 64), in_dim]
    matrix in std_perm's band law (zero past the rows and columns read), expert e at dst + e expert_bytes."""
    bands = roundup(rows, BAND_ROWS) // BAND_ROWS
    if in_dim % 256 or src_dim > in_dim or src_dim % 32 or col0 % 32:
        raise OpRangeError(f"nemotronh: {tensor}: {rows} x {src_dim} at column {col0} into K {in_dim}")
    return {"op": "nh_q40", "tensor": tensor, "dst": dst, "nch": bands * (in_dim // 128), "in_dim": in_dim,
            "src_dim": src_dim, "col0": col0, "rows": rows, "row0": row0, "experts": experts,
            "n_total": n_total or rows, "expert_bytes": expert_bytes}


def _layer_plan(spec: ModelSpec, kind: str) -> dict:
    G, L = geometry(spec), layout(spec)
    b = "blk.{l}."
    hid, hp = G.HID, G.HIDP
    if kind == MAMBA:
        pool = [_q40(b + "ssm_in.weight", L.POOL_IN, G.IN_N, hp, hid),
                _q40(b + "ssm_out.weight", L.POOL_OUT, hid, G.OUT_K, G.MDI)]
    elif kind == FULL:
        bb = band_bytes(hp)
        pool = [_q40(b + "attn_q.weight", L.POOL_IN, G.QW, hp, hid),
                _q40(b + "attn_k.weight", L.POOL_IN + G.QW // BAND_ROWS * bb, G.KVW, hp, hid),
                _q40(b + "attn_v.weight", L.POOL_IN + (G.QW + G.KVW) // BAND_ROWS * bb, G.KVW, hp, hid),
                _q40(b + "attn_output.weight", L.POOL_OUT, hid, G.OUT_K, G.QW)]
    else:
        E, ff = G.EXPERTS, G.FF
        pool = [_q40(b + "ffn_up_exps.weight", 0, ff, hp, hid, experts=E, n_total=ff, expert_bytes=L.UPS),
                _q40(b + "ffn_down_exps.weight", L.POOL_DOWN, hid, G.FFP, ff, experts=E, n_total=hid,
                     expert_bytes=L.DNS)]
        for j in range(G.SH_SLOTS):        # the shared expert: rows j ff.. of up, columns j ff.. of down
            pool += [_q40(b + "ffn_up_shexp.weight", (E + j) * L.UPS, ff, hp, hid, row0=j * ff),
                     _q40(b + "ffn_down_shexp.weight", L.POOL_DOWN + (E + j) * L.DNS, hid, G.FFP, ff, col0=j * ff)]
    return {"pool": pool, "consts": []}


def pack_plan(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    return {
        "pool_bytes": L.POOL_BYTES, "chunk_bytes": CH,
        "layer_types": {k: _layer_plan(spec, k) for k in KINDS if k in spec.layer_types},
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "gx_lmhead", "tensor": "output.weight", "dst": 0, "rows": L.LM_ROWS,
                             "in_dim": G.HIDP}]},
        "embed": {"tensor": "token_embd.weight", "dim": G.HID},
        "norm": {"tensor": "output_norm.weight", "bytes": G.HID * 2},
    }


def _pre(spec: ModelSpec, kernel: str) -> dict:
    G, L = geometry(spec), layout(spec)
    return {"op": "nh_pre", "kernel": kernel, "a_x": L.AD_X, "norm": "blk.{l}.attn_norm.weight", "eps": G.EPS,
            "hidden": G.HID, "k": G.HIDP}


def _add(spec: ModelSpec, kernel: str, a_o: int) -> dict:
    return {"op": "nh_add", "kernel": kernel, "a_o": a_o, "hidden": geometry(spec).HID}


def route_step(spec: ModelSpec) -> dict:
    """nh_route: the router and the slot law nh_moe's fills are patched with (slot j compiled as expert j: up at
    j UPS, down at POOL_DOWN + j DNS; slots TOPK.. are the shared expert's halves, experts EXPERTS..)."""
    G, L = geometry(spec), layout(spec)
    return {"op": "nh_route", "kernel": "nh_moe", "a_xm": L.AD_XM, "a_w": L.AD_W, "norm": "blk.{l}.attn_norm.weight",
            "router": "blk.{l}.ffn_gate_inp.weight", "bias": "blk.{l}.exp_probs_b.bias", "eps": G.EPS,
            "hidden": G.HID, "k": G.HIDP, "experts": G.EXPERTS, "topk": G.TOPK, "slots": G.SLOTS,
            "scale": router_scale(spec), "ups": L.UPS, "dns": L.DNS, "pool_down": L.POOL_DOWN}


def ssm_step(spec: ModelSpec) -> dict:
    G, L = geometry(spec), layout(spec)
    b = "blk.{l}."
    return {"op": "nh_ssm", "kernel": "nh_out", "a_y": L.AD_Y, "a_x": L.AD_X, "conv_w": b + "ssm_conv1d.weight",
            "conv_b": b + "ssm_conv1d.bias", "dt_b": b + "ssm_dt.bias", "ssm_a": b + "ssm_a", "ssm_d": b + "ssm_d",
            "ssm_norm": b + "ssm_norm.weight", "heads": G.MH, "head_dim": G.MHD, "groups": G.MG, "dstate": G.MDS,
            "taps": G.TAPS, "eps": G.EPS, "state_conv": L.STATE_CONV, "k": G.OUT_K}


def attn_step(spec: ModelSpec) -> dict:
    G, L = geometry(spec), layout(spec)
    return {"op": "nh_attn", "kernel": "nh_out", "a_y": L.AD_Y, "a_x": L.AD_X, "heads": G.NH, "kv_heads": G.KVH,
            "head_dim": G.HD, "k": G.OUT_K}


GEMV_ARGS = ["pool", "act"]


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    ks = set(spec.layer_types)
    out = {"contexts": {"nh_out": "nh_out/final.xclbin", "lm": "lm_head_q8/final.xclbin"},
           "kernels": {"nh_out": {"context": "nh_out", "insts": "nh_out/insts.bin", "build": "nh_out"},
                       "lm": {"context": "lm", "insts": "lm_head_q8/insts.bin", "build": "lm_head_q8"}},
           "layer_types": {},
           "tail": [{"op": "nh_final", "kernel": "lm", "norm": "output_norm.weight", "eps": G.EPS, "hidden": G.HID,
                     "k": G.HIDP, "dst": "hn"},
                    {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
           "globals": {"xres": G.HID * 4, "hn": G.HIDP * 2, "logits": L.LM_ROWS * 4, "lmpool": L.LMHEAD_POOL_BYTES}}

    shared = shared_gemv()
    if shared:     # one image, three streams: the context is nh_out's xclbin (every build of it is the same image)
        out["contexts"] = {"nh_gemv": "nh_out/final.xclbin", "lm": out["contexts"]["lm"]}
        out["kernels"]["nh_out"]["context"] = "nh_gemv"

    def kern(name: str, patch: str = "") -> None:
        ctx = "nh_gemv" if shared and name in ("nh_in", "nh_qkv") else name
        if ctx == name:
            out["contexts"][name] = f"{name}/final.xclbin"
        out["kernels"][name] = {"context": ctx, "insts": f"{name}/insts.bin", "build": name}
        if patch:
            out["kernels"][name]["patch"] = patch
    mix = {"consts": CONSTS, "act": L.AD_BYTES, "pool": L.POOL_MIX_BYTES}
    if MAMBA in ks:
        kern("nh_in")
        out["layer_types"][MAMBA] = {
            "buffers": {**mix, "state": {"kind": "linear", "bytes": L.STATE_BYTES}},
            "program": [_pre(spec, "nh_in"), {"op": "run", "kernel": "nh_in", "args": GEMV_ARGS}, ssm_step(spec),
                        {"op": "run", "kernel": "nh_out", "args": GEMV_ARGS}, _add(spec, "nh_out", L.AD_O)]}
    if FULL in ks:
        kern("nh_qkv")
        out["layer_types"][FULL] = {
            "buffers": {**mix, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [_pre(spec, "nh_qkv"), {"op": "run", "kernel": "nh_qkv", "args": GEMV_ARGS}, attn_step(spec),
                        {"op": "run", "kernel": "nh_out", "args": GEMV_ARGS}, _add(spec, "nh_out", L.AD_O)]}
    if MOE_BLOCK in ks:
        kern("nh_moe", "nhroute")
        out["layer_types"][MOE_BLOCK] = {
            "buffers": {"consts": CONSTS, "act": L.AD_BYTES, "pool": L.POOL_MOE_BYTES,
                        "state": {"kind": "linear", "bytes": STATE_MOE}},
            "program": [route_step(spec), {"op": "run", "kernel": "nh_moe", "args": GEMV_ARGS},
                        _add(spec, "nh_moe", L.AD_MOE)]}
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    G, L = geometry(spec), layout(spec)
    out = {
        "lm_head_q8": {"design": "lm_head_q8/lm_head_q8.py",
                       "build_dir": f"lm_head_q8/build_gguf_n{L.LM_ROWS}_k{G.HIDP}_c{LM_CORES}",
                       "env": {"LMHEAD_N": str(L.LM_ROWS), "LMHEAD_K": str(G.HIDP), "LMHEAD_CORES": str(LM_CORES),
                               "LMHEAD_SCALES_F32": "1"}},
        "nh_out": {"design": "nemotronh/nh.py", "build_dir": f"nemotronh/build_out_n{G.HID}_k{G.OUT_K}",
                   "env": {"NH_PART": "out"}},
    }
    ks = set(spec.layer_types)
    if MAMBA in ks:
        out["nh_in"] = {"design": "nemotronh/nh.py", "build_dir": f"nemotronh/build_in_n{G.IN_N}_k{G.HIDP}",
                        "env": {"NH_PART": "in"}}
    if FULL in ks:
        out["nh_qkv"] = {"design": "nemotronh/nh.py", "build_dir": f"nemotronh/build_qkv_n{G.QKV_N}_k{G.HIDP}",
                         "env": {"NH_PART": "qkv"}}
    if shared_gemv():
        for name in ("nh_out", "nh_in", "nh_qkv"):
            if name in out:
                out[name]["build_dir"] += "_rtp"
                out[name]["env"]["NEMOTRONH_SHARED"] = "1"
    if MOE_BLOCK in ks:
        out["nh_moe"] = {"design": "nemotronh/nh.py",
                         "build_dir": f"nemotronh/build_moe_h{G.HID}_f{G.FF}_s{G.SLOTS}",
                         "env": {"NH_PART": "moe"}}
    return out


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    return {"hidden": G.HID, "vocab": L.LM_ROWS, "real_vocab": spec.real_vocab, "chunk_bytes": CH,
            "pool_bytes": L.POOL_BYTES, "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": LM_TILE,
            "kv_row": L.KV_ROW, "ptab_row": 1024, "rotary_dim": 0, "rope_theta": 0.0, "rope_inv_freq": []}


def gguf_quant(spec: ModelSpec) -> str:
    """The law the pools hold: Q4_0-native (our pure Q4_0 exactly)."""
    return "q4_0"


def finalize_gguf_manifest(spec: ModelSpec, manifest: dict) -> None:
    """The manifest IS the GGUF view (there is no q4nx container): core.cpp reads a model.gguf through the nested
    `gguf` section, so the section is the same manifest, nested once."""
    if "gguf" not in manifest:
        manifest["gguf"] = copy.deepcopy(manifest)


def hf_config_check(spec: ModelSpec) -> dict:
    G = geometry(spec)
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads, "head_dim": spec.head_dim,
            "mamba_num_heads": G.MH, "mamba_head_dim": G.MHD, "n_groups": G.MG, "ssm_state_size": G.MDS,
            "conv_kernel": G.TAPS, "moe_intermediate_size": spec.moe_intermediate,
            "moe_shared_expert_intermediate_size": spec.shared_expert_intermediate,
            "n_routed_experts": spec.num_experts, "num_experts_per_tok": spec.experts_per_tok,
            "routed_scaling_factor": router_scale(spec), "norm_topk_prob": True, "tie_word_embeddings": False}


GEN_KERNELS = "designs/nemotronh/gen_kernels.py"
KERNEL_SOURCES = [
    "designs/nemotronh/*.py", "designs/nemotronh/*.cc",
    "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h", "designs/gemv_q4/gemv_q4.py",
    "designs/lm_head_q8/*.py", "designs/lm_head_q8/*.h", "designs/lm_head_q8/*.cc",
    "include/vecmath.h", "ironutil.py", "build_design.py",
]
