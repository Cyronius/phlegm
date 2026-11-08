"""Xing 4.0 (OPEN-FAMILY-XING; Xing4.0-29B-A4B): a DeepSeek-V3 decoder (MLA attention, leading dense MLPs, routed
SiLU experts with a shared expert) whose residual is DeepSeek's mHC -- four streams mixed around every sublayer --
on the open engine, from the GGUF.

The NPU streams the weights -- every projection is a q4 GEMV, the FFN one gated expert program -- and the host does
everything between them (xing-runtime.patch, over Nemotron-H's host-step chain), so a layer is a program of host
steps around four dispatches of ONE image (R-0003: the image changes only for the head):

  xg_pre      (host) mHC pre of the attention: f = the 4 x hidden streams / rms; m = f hc_fn^T [24]; pre, post,
              the Sinkhorn comb; h = rms(sum_i pre_i S_i) * attn_norm -> the GEMV's f32 x
  run xg_qkva q_a | kv_a_mqa GEMV [768 + 576, 3584]
  xg_qnorm    (host) q_a_norm -> the next GEMV's x; kv_a_norm on the 512 latent, RoPE on the 64-dim key -> the
              layer's KV row [latent | rope key] (f32, 576 values) at this position
  run xg_qb   q_b GEMV [32 x 192, 768]
  xg_mla      (host) RoPE on the q rope halves, q_nope absorbed through W_UK into the latent (attn_k_b), the
              MQA-like attention over the KV rows (score dim 576, value dim 512, scale 192^-0.5 mscale^2), the
              output through W_UV (attn_v_b) -> [32 x 128]
  run xg_o    o GEMV [3584, 4096]
  xg_post     (host) S_i <- post_i o + sum_j C_ij S_j
  xg_pre      (host) mHC pre of the FFN, ffn_norm -> xm (bf16); `route`: the sigmoid router (top 4 of s +
              exp_probs_b, the unbiased s renormalised x 2.0) and the shared expert at weight 1, the slots'
              fills re-pointed (`nhroute`); `dense`: the 9 slices of the dense MLP at weight 1
  run xg_moe / xg_dense   per slot: gate | up bands -> h = w silu(g) u; then down summed over the slots
  xg_post     (host) S_i <- post_i m + sum_j C_ij S_j
  tail        xg_final (host: mean of the streams, output_norm -> bf16 hn) | lm: the Q8_0 head (lm_head_q8)

Why the host does mHC and the attention. mHC per sublayer is ~344 K MACs (the 24 x 14336 mixes), a 4 x 4 Sinkhorn
and 4 x 3584 element updates: tiny arithmetic on a 57 KB f32 residual that the host already holds (the residual
never goes to the device in this chain). On the NPU it would be another dispatch per sublayer (or a resident
57 KB state in a core's 64 KB L1, beside the GEMV's x) for ~0.1 ms of host work. The absorbed MLA attention is a
new shape for the attention family (32 q heads sharing one 576-wide K and one 512-wide V, 2.3 KB a row): its
products at context 4096 are ~0.15 GMAC a layer, host f32 for now (as Nemotron-H's attention); an NPU kernel is
an open option. The absorption's W_UK / W_UV (4.2 M MACs a layer) are per-head 512 x 128 products, host f32 from
the GGUF's attn_k_b / attn_v_b (their Q4_0 values widened exactly).

Widths. hidden 3584 = 56 bands = 14 k-tiles, q_lora 768 = 3 k-tiles, the attention output 4096, the expert width
1024: every K is whole k-tiles and every N whole bands, so nothing is padded. The dense MLP (9216) runs as nine
1024-wide slots (exact: SiLU-gated products are per row); the shared expert is one more slot. The pool's expert
law interleaves each 64-row gate band with its up band, so a core's slot fill is one contiguous run.

Weights. The GGUF is our own pure Q4_0 from the publisher's BF16 safetensors (llama.cpp convert --no-mtp, then
llama quantize --pure --output-tensor-type q8_0; .plans/xing.md). From a BF16 source every Q4_0 scale is a bf16
value, so the Q4_0-native pool law (4608-B chunks, the bf16 d plane, m = -8 d on the core; `nh_q40`) holds every
matrix exactly. The router, its bias, hc_fn / hc_base / hc_scale and the norms are host tensors (the quantizer
keeps them out of Q4_0); attn_k_b / attn_v_b (Q4_0) are widened to f32 on the host; the embedding (Q4_0) stays on
the host (embed_row).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from .catalogue import OpRangeError
from .qwen36moe import BAND_ROWS, ELEM, MB, roundup
from .spec import DENSE, MOE_BLOCK, ModelSpec

CH = 4608                 # a Q4_0-native chunk: 32 rows x 256 K, the bf16 d plane then the codes (gemv_q4.h)
PER_CALL = 2              # chunks per w element (9216 B)
N_CORES = 8
WV_SLOTS = 16             # the routing-weight record (f32[16] in xm's last element) and the patch table
STACK = 0x1800
LM_CORES = 8
LM_BAND_ROWS = 128        # lm_head_q8: 128-row bands
LM_TILE = 9216            # lm_head_q8.h's f32-scale chunk: 32 rows x 256 K
CONSTS = 4096             # no consts are packed: the host reads the small tensors from the GGUF
KINDS = (DENSE, MOE_BLOCK)
RTP_N = 12
GEMVS = ("qkva", "qb", "o")


PROBE_VARS = ("XG_NULL",)


def probe_env() -> dict[str, str]:
    """The DMA-only probe (XG_NULL: no arithmetic) folds into the build key (cache.py)."""
    import os
    return {k: os.environ[k] for k in PROBE_VARS if os.environ.get(k)}


def kpad(k: int) -> int:
    return roundup(k, 256)


def band_bytes(K: int) -> int:
    """64 rows x K (K whole k-tiles): K/256 k-tiles x 2 thirty-two-row chunks."""
    return K // 128 * CH


def split_bands(bands: int, n_cores: int = N_CORES) -> list[int]:
    """lm_head_q8's uneven split: the first `bands % n_cores` cores take one band more."""
    q, r = divmod(bands, n_cores)
    return [q + (1 if c < r else 0) for c in range(n_cores)]


def offsets(counts: list[int]) -> list[int]:
    out, o = [], 0
    for n in counts:
        out.append(o)
        o += n
    return out


@dataclass(frozen=True)
class Geometry:
    HID: int; EPS: float
    NH: int; NOPE: int; ROPE: int; VD: int; QL: int; KVL: int
    QKVA_N: int; QB_N: int; O_K: int
    FF: int; DFF: int; DSLOTS: int; TOPK: int; MSLOTS: int; EXPERTS: int; MEXPERTS: int
    HC: int; HC_MIX: int; HC_ITERS: int; HC_EPS: float; HC_CLAMP: float
    KV_DIM: int; SCALE: float; ROUTER_SCALE: float


@dataclass(frozen=True)
class Layout:
    AD_Y: int; AD_O: int; AD_XM: int; AD_W: int; AD_H: int; AD_MOE: int; AD_XF: int; AD_BYTES: int
    XM_ELEMS: int; H_STRIDE: int; W_OFF: int
    POOL_QKVA: int; POOL_QB: int; POOL_O: int; POOL_EXP: int
    UPS: int; DNS: int; POOL_DOWN_D: int; POOL_DENSE_BYTES: int; POOL_DOWN_M: int; POOL_MOE_BYTES: int
    POOL_BYTES: int; KV_ROW: int
    LM_ROWS: int; LMHEAD_POOL_BYTES: int


@dataclass(frozen=True)
class XingRecipe:
    spec: ModelSpec
    layout: Layout
    geo: Geometry
    max_ctx: int
    kind: str = "xing"


def _check(spec: ModelSpec) -> None:
    if spec.family != "xing":
        raise OpRangeError(f"xing recipe given a {spec.family!r} spec")
    if not spec.mla or not spec.hyper:
        raise OpRangeError("xing: the spec carries no MLA / hyper-connection geometry")
    bad = sorted(set(spec.layer_types) - set(KINDS))
    if bad:
        raise OpRangeError(f"xing: layer types {bad} are not dense / moe")
    if spec.activation != "silu" or not spec.num_experts or not spec.moe_intermediate:
        raise OpRangeError("xing: expected routed SiLU experts")
    ff = spec.moe_intermediate
    if spec.shared_expert_intermediate != ff:
        raise OpRangeError(f"xing: the shared expert {spec.shared_expert_intermediate} is not one {ff}-wide slot")
    if DENSE in spec.layer_types and (not spec.intermediate or spec.intermediate % ff):
        raise OpRangeError(f"xing: the dense MLP {spec.intermediate} is not whole {ff}-wide slots")
    m, h = spec.mla, spec.hyper
    if h["mult"] != 4:
        raise OpRangeError(f"xing: {h['mult']} hyper-connection streams (the host steps hold 4)")
    if spec.rotary_dim != m["qk_rope_dim"] or spec.head_dim != m["qk_nope_dim"] + m["qk_rope_dim"]:
        raise OpRangeError("xing: head_dim / rotary_dim do not match the MLA dims")
    slots = max(spec.experts_per_tok + 1, (spec.intermediate // ff) if DENSE in spec.layer_types else 0)
    if slots > WV_SLOTS:
        raise OpRangeError(f"xing: {slots} expert slots exceed the {WV_SLOTS}-slot record")
    widths = (("hidden", spec.hidden), ("q_lora", m["q_lora_rank"]), ("expert width", ff),
              ("attention output", spec.num_heads * m["v_head_dim"]))
    for what, w in widths:
        if w % 256:
            raise OpRangeError(f"xing: {what} {w} is not whole 256-wide k-tiles")
    for what, w in (("q_a | kv_a", m["q_lora_rank"] + m["kv_lora_rank"] + m["qk_rope_dim"]),
                    ("q_b", spec.num_heads * spec.head_dim), ("hidden", spec.hidden), ("expert width", ff)):
        if w % BAND_ROWS:
            raise OpRangeError(f"xing: {what} {w} is not whole 64-row bands")
    if spec.qk_norm or spec.attn_gate or spec.sliding_window:
        raise OpRangeError("xing: the attention has no q/k norm, no gate, no window")


def geometry(spec: ModelSpec) -> Geometry:
    _check(spec)
    m, h = spec.mla, spec.hyper
    ff = spec.moe_intermediate
    hc = h["mult"]
    return Geometry(
        HID=spec.hidden, EPS=spec.norm_eps,
        NH=spec.num_heads, NOPE=m["qk_nope_dim"], ROPE=m["qk_rope_dim"], VD=m["v_head_dim"],
        QL=m["q_lora_rank"], KVL=m["kv_lora_rank"],
        QKVA_N=m["q_lora_rank"] + m["kv_lora_rank"] + m["qk_rope_dim"], QB_N=spec.num_heads * spec.head_dim,
        O_K=spec.num_heads * m["v_head_dim"],
        FF=ff, DFF=spec.intermediate, DSLOTS=spec.intermediate // ff if spec.intermediate else 0,
        TOPK=spec.experts_per_tok, MSLOTS=spec.experts_per_tok + 1, EXPERTS=spec.num_experts,
        MEXPERTS=spec.num_experts + 1,
        HC=hc, HC_MIX=(2 + hc) * hc, HC_ITERS=h["sinkhorn_iters"], HC_EPS=h["eps"], HC_CLAMP=h["clamp"],
        KV_DIM=m["kv_lora_rank"] + m["qk_rope_dim"], SCALE=spec.attn_scale or spec.head_dim ** -0.5,
        ROUTER_SCALE=float(spec.extra.get("router_scale", 1.0)),
    )


def lm_rows(spec: ModelSpec) -> int:
    return roundup(spec.vocab, LM_BAND_ROWS)


def slots_max(G: Geometry) -> int:
    return max(G.MSLOTS, G.DSLOTS)


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Layout:
    G = geometry(spec)
    a: dict[str, int] = {}
    off = 0
    xm_elems = roundup(G.HID * 2 + 4 * WV_SLOTS, ELEM) // ELEM
    w_off = roundup(G.HID * 2 - (xm_elems - 1) * ELEM, 64)          # the weights: past xm, in its last element
    if w_off + 4 * WV_SLOTS > ELEM:
        raise OpRangeError("xing: the routing weights do not fit xm's last element")
    h_stride = roundup(G.FF * 4, ELEM)
    xf = max(G.HID, G.QL, G.O_K) * 4
    for name, size in (("y", max(G.QKVA_N, G.QB_N) * 4), ("o", G.HID * 4), ("xm", xm_elems * ELEM),
                       ("h", slots_max(G) * h_stride), ("moe", G.HID * 4), ("xf", xf)):
        a[name] = off
        off += roundup(size, ELEM)
    # the attention's three GEMVs at the head of every layer pool (the same offsets in both kinds), then the
    # experts: every expert's gate | up (band-interleaved) at POOL_EXP + e UPS, then the down projections
    bb_h, bb_q, bb_o, bb_f = band_bytes(G.HID), band_bytes(G.QL), band_bytes(G.O_K), band_bytes(G.FF)
    p_qb = G.QKVA_N // BAND_ROWS * bb_h
    p_o = p_qb + G.QB_N // BAND_ROWS * bb_q
    p_exp = roundup(p_o + G.HID // BAND_ROWS * bb_o, MB)
    ups = 2 * (G.FF // BAND_ROWS) * bb_h
    dns = G.HID // BAND_ROWS * bb_f
    pd_d = p_exp + G.DSLOTS * ups
    pool_d = roundup(pd_d + G.DSLOTS * dns, MB)
    pd_m = p_exp + G.MEXPERTS * ups
    pool_m = roundup(pd_m + G.MEXPERTS * dns, MB)
    if max(pool_d, pool_m) >= 1 << 32:
        raise OpRangeError("xing: a layer pool past 4 GiB does not fit a 32-bit fill offset")
    rows = lm_rows(spec)
    bands = rows // LM_BAND_ROWS
    lm_band = 4 * (G.HID // 256) * LM_TILE
    pools = [pool_m] + ([pool_d] if DENSE in spec.layer_types else [])
    return Layout(
        AD_Y=a["y"], AD_O=a["o"], AD_XM=a["xm"], AD_W=a["xm"] + (xm_elems - 1) * ELEM + w_off, AD_H=a["h"],
        AD_MOE=a["moe"], AD_XF=a["xf"], AD_BYTES=roundup(off, ELEM), XM_ELEMS=xm_elems, H_STRIDE=h_stride,
        W_OFF=w_off, POOL_QKVA=0, POOL_QB=p_qb, POOL_O=p_o, POOL_EXP=p_exp,
        UPS=ups, DNS=dns, POOL_DOWN_D=pd_d, POOL_DENSE_BYTES=pool_d, POOL_DOWN_M=pd_m, POOL_MOE_BYTES=pool_m,
        POOL_BYTES=max(pools), KV_ROW=G.KV_DIM * 4,
        LM_ROWS=rows, LMHEAD_POOL_BYTES=roundup(roundup(bands, LM_CORES) * lm_band, MB),
    )


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> XingRecipe:
    from .catalogue import require
    G = geometry(spec)
    require("xg", hidden=G.HID, q_lora=G.QL, kv_a=G.KV_DIM, q_b=G.QB_N, o_k=G.O_K, ff=G.FF, moe_slots=G.MSLOTS,
            dense_slots=G.DSLOTS)
    require("lm_head_q8", K=G.HID, vocab=lm_rows(spec))
    return XingRecipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def gemv_shapes(spec: ModelSpec) -> dict[str, dict]:
    """The three attention GEMVs: rows, K, the pool offset of the weight, the act region out (the x is always the
    f32 at AD_XF; designs/xing/xg.py XG_PART)."""
    G, L = geometry(spec), layout(spec)
    return {"qkva": {"n": G.QKVA_N, "k": G.HID, "pool": L.POOL_QKVA, "ay": L.AD_Y},
            "qb": {"n": G.QB_N, "k": G.QL, "pool": L.POOL_QB, "ay": L.AD_Y},
            "o": {"n": G.HID, "k": G.O_K, "pool": L.POOL_O, "ay": L.AD_O}}


def ffn_shape(spec: ModelSpec, kind: str) -> dict:
    """The expert stream of a layer kind: its slot count and the pool offsets slot j is compiled at (pool expert j)."""
    G, L = geometry(spec), layout(spec)
    if kind == "dense":
        return {"slots": G.DSLOTS, "pool_down": L.POOL_DOWN_D, "pool_bytes": L.POOL_DENSE_BYTES}
    return {"slots": G.MSLOTS, "pool_down": L.POOL_DOWN_M, "pool_bytes": L.POOL_MOE_BYTES}


def _q40(tensor: str, dst: int, rows: int, in_dim: int, src_dim: int, row0: int = 0, col0: int = 0,
         experts: int = 1, n_total: int = 0, expert_bytes: int = 0, band_stride: int = 0) -> dict:
    """runtime pack op nh_q40 (nemotronh-runtime.patch; band_stride: xing-runtime.patch): for e < experts, `rows`
    file rows from row0 + e n_total and `src_dim` columns from col0 of a Q4_0 GGUF tensor, as Q4_0-native chunks of a
    [roundup(rows, 64), in_dim] matrix in std_perm's band law, expert e at dst + e expert_bytes, band b at
    + b band_stride (default: one band after another)."""
    bands = roundup(rows, BAND_ROWS) // BAND_ROWS
    if in_dim % 256 or src_dim > in_dim or src_dim % 32 or col0 % 32:
        raise OpRangeError(f"xing: {tensor}: {rows} x {src_dim} at column {col0} into K {in_dim}")
    op = {"op": "nh_q40", "tensor": tensor, "dst": dst, "nch": bands * (in_dim // 128), "in_dim": in_dim,
          "src_dim": src_dim, "col0": col0, "rows": rows, "row0": row0, "experts": experts,
          "n_total": n_total or rows, "expert_bytes": expert_bytes}
    if band_stride:
        op["band_stride"] = band_stride
    return op


def _layer_plan(spec: ModelSpec, kind: str) -> dict:
    G, L = geometry(spec), layout(spec)
    b = "blk.{l}."
    hid, ff = G.HID, G.FF
    bb = band_bytes(hid)
    pool = [_q40(b + "attn_q_a.weight", L.POOL_QKVA, G.QL, hid, hid),
            _q40(b + "attn_kv_a_mqa.weight", L.POOL_QKVA + G.QL // BAND_ROWS * bb, G.KVL + G.ROPE, hid, hid),
            _q40(b + "attn_q_b.weight", L.POOL_QB, G.QB_N, G.QL, G.QL),
            _q40(b + "attn_output.weight", L.POOL_O, hid, G.O_K, G.O_K)]
    # gate | up: band b of slot e at POOL_EXP + e UPS + 2 b bb (gate) and + (2 b + 1) bb (up)
    if kind == DENSE:
        # the dense MLP as DSLOTS 1024-wide slots: rows j ff.. of gate / up, columns j ff.. of down
        n, pd = G.DSLOTS, L.POOL_DOWN_D
        pool += [_q40(b + "ffn_gate.weight", L.POOL_EXP, ff, hid, hid, experts=n, n_total=ff, expert_bytes=L.UPS,
                      band_stride=2 * bb),
                 _q40(b + "ffn_up.weight", L.POOL_EXP + bb, ff, hid, hid, experts=n, n_total=ff, expert_bytes=L.UPS,
                      band_stride=2 * bb)]
        pool += [_q40(b + "ffn_down.weight", pd + j * L.DNS, hid, ff, ff, col0=j * ff) for j in range(n)]
    else:
        E, pd = G.EXPERTS, L.POOL_DOWN_M
        pool += [_q40(b + "ffn_gate_exps.weight", L.POOL_EXP, ff, hid, hid, experts=E, n_total=ff, expert_bytes=L.UPS,
                      band_stride=2 * bb),
                 _q40(b + "ffn_up_exps.weight", L.POOL_EXP + bb, ff, hid, hid, experts=E, n_total=ff,
                      expert_bytes=L.UPS, band_stride=2 * bb),
                 _q40(b + "ffn_down_exps.weight", pd, hid, ff, ff, experts=E, n_total=hid, expert_bytes=L.DNS),
                 # the shared expert: pool expert E
                 _q40(b + "ffn_gate_shexp.weight", L.POOL_EXP + E * L.UPS, ff, hid, hid, band_stride=2 * bb),
                 _q40(b + "ffn_up_shexp.weight", L.POOL_EXP + E * L.UPS + bb, ff, hid, hid, band_stride=2 * bb),
                 _q40(b + "ffn_down_shexp.weight", pd + E * L.DNS, hid, ff, ff)]
    return {"pool": pool, "consts": []}


def pack_plan(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    return {
        "pool_bytes": L.POOL_BYTES, "chunk_bytes": CH,
        "layer_types": {k: _layer_plan(spec, k) for k in KINDS if k in spec.layer_types},
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "nh_lmhead", "tensor": "output.weight", "dst": 0, "rows": L.LM_ROWS,
                             "in_dim": G.HID}]},
        "embed": {"tensor": "token_embd.weight", "dim": G.HID},
        "norm": {"tensor": "output_norm.weight", "bytes": G.HID * 2},
    }


def _hc(G: Geometry, which: str) -> dict:
    b = "blk.{l}."
    return {"hc_fn": b + f"hc_{which}_fn.weight", "hc_base": b + f"hc_{which}_base.weight",
            "hc_scale": b + f"hc_{which}_scale.weight", "hc": G.HC, "hc_iters": G.HC_ITERS, "hc_eps": G.HC_EPS,
            "hc_clamp": G.HC_CLAMP}


def pre_attn_step(spec: ModelSpec) -> dict:
    G, L = geometry(spec), layout(spec)
    return {"op": "xg_pre", "kernel": "xg_qkva", "mode": "x", **_hc(G, "attn"), "norm": "blk.{l}.attn_norm.weight",
            "eps": G.EPS, "hidden": G.HID, "a_x": L.AD_XF, "k": G.HID}


def qnorm_step(spec: ModelSpec) -> dict:
    G, L = geometry(spec), layout(spec)
    b = "blk.{l}."
    return {"op": "xg_qnorm", "kernel": "xg_qb", "a_y": L.AD_Y, "a_x": L.AD_XF, "q_norm": b + "attn_q_a_norm.weight",
            "kv_norm": b + "attn_kv_a_norm.weight", "q_lora": G.QL, "kv_lora": G.KVL, "rope": G.ROPE, "eps": G.EPS,
            "k": G.QL}


def mla_step(spec: ModelSpec) -> dict:
    G, L = geometry(spec), layout(spec)
    b = "blk.{l}."
    return {"op": "xg_mla", "kernel": "xg_o", "a_y": L.AD_Y, "a_x": L.AD_XF, "k_b": b + "attn_k_b.weight",
            "v_b": b + "attn_v_b.weight", "heads": G.NH, "nope": G.NOPE, "rope": G.ROPE, "v_dim": G.VD,
            "kv_lora": G.KVL, "scale": G.SCALE, "k": G.O_K}


def post_step(spec: ModelSpec, kernel: str, a_o: int) -> dict:
    G = geometry(spec)
    return {"op": "xg_post", "kernel": kernel, "a_o": a_o, "hidden": G.HID, "hc": G.HC}


def pre_ffn_step(spec: ModelSpec, kind: str) -> dict:
    """xg_pre of the FFN: xm (bf16) + the slots' weights; `route` (moe): the router and the slot law the expert
    stream's fills are patched with (slot j compiled as pool expert j: gate | up at POOL_EXP + j UPS, down at
    pool_down + j DNS; slot TOPK is the shared expert, pool expert EXPERTS); `dense`: slot j = slice j, weight 1."""
    G, L = geometry(spec), layout(spec)
    sh = ffn_shape(spec, kind)
    st = {"op": "xg_pre", "kernel": "xg_dense" if kind == DENSE else "xg_moe", **_hc(G, "ffn"),
          "norm": "blk.{l}.ffn_norm.weight", "eps": G.EPS, "hidden": G.HID, "a_xm": L.AD_XM, "a_w": L.AD_W,
          "k": G.HID, "slots": sh["slots"]}
    if kind == DENSE:
        st["mode"] = "dense"
    else:
        st.update({"mode": "route", "router": "blk.{l}.ffn_gate_inp.weight", "bias": "blk.{l}.exp_probs_b.bias",
                   "experts": G.EXPERTS, "topk": G.TOPK, "scale": G.ROUTER_SCALE, "pool_exp": L.POOL_EXP,
                   "ups": L.UPS, "dns": L.DNS, "pool_down": sh["pool_down"]})
    return st


GEMV_ARGS = ["pool", "act"]


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    ks = set(spec.layer_types)
    kern = {"lm": {"context": "lm", "insts": "lm_head_q8/insts.bin", "build": "lm_head_q8"}}
    for name in [f"xg_{p}" for p in GEMVS] + [f"xg_{k}" for k in ("dense", "moe") if
                                               (k == "dense" and DENSE in ks) or (k == "moe" and MOE_BLOCK in ks)]:
        kern[name] = {"context": "xg_one", "insts": f"{name}/insts.bin", "build": name}
    if MOE_BLOCK in ks:
        kern["xg_moe"]["patch"] = "nhroute"
    out = {"contexts": {"xg_one": "xg_o/final.xclbin", "lm": "lm_head_q8/final.xclbin"},
           "kernels": kern, "layer_types": {},
           "tail": [{"op": "xg_final", "kernel": "lm", "norm": "output_norm.weight", "eps": G.EPS, "hidden": G.HID,
                     "hc": G.HC, "k": G.HID, "dst": "hn"},
                    {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
           "globals": {"xres": G.HC * G.HID * 4, "hn": G.HID * 2, "logits": L.LM_ROWS * 4,
                       "lmpool": L.LMHEAD_POOL_BYTES}}

    def program(kind: str) -> list[dict]:
        ffn = "xg_dense" if kind == DENSE else "xg_moe"
        return [pre_attn_step(spec), {"op": "run", "kernel": "xg_qkva", "args": GEMV_ARGS},
                qnorm_step(spec), {"op": "run", "kernel": "xg_qb", "args": GEMV_ARGS},
                mla_step(spec), {"op": "run", "kernel": "xg_o", "args": GEMV_ARGS},
                post_step(spec, "xg_o", L.AD_O),
                pre_ffn_step(spec, kind), {"op": "run", "kernel": ffn, "args": GEMV_ARGS},
                post_step(spec, ffn, L.AD_MOE)]
    for kind, pool in ((DENSE, L.POOL_DENSE_BYTES), (MOE_BLOCK, L.POOL_MOE_BYTES)):
        if kind in ks:
            out["layer_types"][kind] = {
                "buffers": {"consts": CONSTS, "act": L.AD_BYTES, "pool": pool,
                            "state": {"kind": "kv", "row": L.KV_ROW}},
                "program": program(kind)}
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    G, L = geometry(spec), layout(spec)
    out = {"lm_head_q8": {"design": "lm_head_q8/lm_head_q8.py",
                          "build_dir": f"lm_head_q8/build_gguf_n{L.LM_ROWS}_k{G.HID}_c{LM_CORES}",
                          "env": {"LMHEAD_N": str(L.LM_ROWS), "LMHEAD_K": str(G.HID), "LMHEAD_CORES": str(LM_CORES),
                                  "LMHEAD_SCALES_F32": "1"}}}
    for p, sh in gemv_shapes(spec).items():
        out[f"xg_{p}"] = {"design": "xing/xg.py", "build_dir": f"xing/build_{p}_n{sh['n']}_k{sh['k']}",
                          "env": {"XG_PART": p}}
    ks = set(spec.layer_types)
    if DENSE in ks:
        out["xg_dense"] = {"design": "xing/xg.py", "build_dir": f"xing/build_dense_h{G.HID}_f{G.FF}_s{G.DSLOTS}",
                           "env": {"XG_PART": "dense"}}
    if MOE_BLOCK in ks:
        out["xg_moe"] = {"design": "xing/xg.py", "build_dir": f"xing/build_moe_h{G.HID}_f{G.FF}_s{G.MSLOTS}",
                         "env": {"XG_PART": "moe"}}
    return out


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    return {"hidden": G.HID, "vocab": L.LM_ROWS, "real_vocab": spec.real_vocab, "chunk_bytes": CH,
            "pool_bytes": L.POOL_BYTES, "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": LM_TILE,
            "kv_row": L.KV_ROW, "ptab_row": 1024, "rotary_dim": G.ROPE, "rope_theta": spec.rope_theta,
            "rope_inv_freq": spec.rope_inv_freq()}


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
            "num_attention_heads": spec.num_heads, "q_lora_rank": G.QL, "kv_lora_rank": G.KVL,
            "qk_nope_head_dim": G.NOPE, "qk_rope_head_dim": G.ROPE, "v_head_dim": G.VD,
            "moe_intermediate_size": spec.moe_intermediate, "n_routed_experts": spec.num_experts,
            "num_experts_per_tok": spec.experts_per_tok, "n_shared_experts": 1,
            "first_k_dense_replace": spec.layer_types.count(DENSE), "hc_mult": G.HC,
            "routed_scaling_factor": G.ROUTER_SCALE, "norm_topk_prob": True, "tie_word_embeddings": False}


GEN_KERNELS = "designs/xing/gen_kernels.py"
KERNEL_SOURCES = [
    "designs/xing/*.py", "designs/xing/*.cc",
    "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h", "designs/gemv_q4/gemv_q4.py",
    "designs/lm_head_q8/*.py", "designs/lm_head_q8/*.h", "designs/lm_head_q8/*.cc",
    "include/vecmath.h", "ironutil.py", "build_design.py",
]
