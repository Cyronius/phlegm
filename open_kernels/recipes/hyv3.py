"""Tencent HY V3 (OPEN-FAMILY-HYV3; Hy-Memory-30B-A3B) on the open engine, from the GGUF.

A layer is two dispatches with the router on the host between them:

  hy_att  (designs/hyv3/hy.py, HY_PART=att; attnpos):
          ln -> q | k | v -> per-head RMSNorm of q and k -> NEOX RoPE (every layer, every dim)
          -> causal attention (no sink, no window) -> o -> res = x + o; xm = ffn_norm(res)
  hy_route (host): sigmoid scores of the F32 router [128, 2048] over xm; the top 8 of score + the
          per-expert selection bias; their UNBIASED scores renormalised and scaled by
          router_scaling_factor; slot 8 is the ungated shared expert (pool expert 128) at weight 1.
          The weights go into act beside xm, the nine experts into hy_moe's fills (moeroute2's patch).
          The leading dense layer runs the same step in its dense mode: slots 0..8 = its MLP's nine
          768-wide slices (pool experts 0..8) at weight 1, no router.
  hy_moe  (HY_PART=moe; moeroute2): gate | up of each slot -> w_e * silu(gate) * up -> h_e; down
          sums the nine slots in each core's accumulator; xres = res + down

Then the final norm (`ln`) and the q8 head (`lm_head_q8`, f32 scales: gx_lmhead keeps the GGUF's
Q8_0 codes and fp16 scales exactly).

Weights. The GGUF is our own pure Q4_0 (llama quantize --pure, Q8_0 head; .plans/hymemory-native.md):
from a BF16 source every Q4_0 scale is a block max / -8, a bf16 value, so the q4_1 pool law
(5120-B chunks, d and m = -8 d as bf16; flint-runtime.patch fl_q4) holds every matrix EXACTLY. The
embedding (Q4_0) stays on the host; the norms and q / k norms go to bf16 in the consts.

The expert width 768 is twelve 64-row bands: not a whole number per core over 8 cores. So the
gate | up phase takes the slots in PAIRS: cores 0-3 run slot 2v's bands (3 each), cores 4-7 slot
2v + 1's, and each core's fill is compiled as the pool expert of its own slot (so moeroute2's table
reads it as that slot, intra = the core's band offset, and the stock patch re-points it). The ninth
slot runs on cores 0-3 alone. Down is per slot over all eight cores (K = 768: three k-tiles).

Block prefill (on by default; HYV3_BLOCK=0 drops it and leaves the sequential emission as it was): LFM2.5's
route (recipes/lfm2moe.py gemm_route, runtime kind `lfm`, mixer `att`) over the SAME pool bytes -- q | k | v
and o as q4_1 GEMMs (gemm_q4_prefill), the entry norm, attention (q / k norms, NEOX RoPE, causal, no gate) and
ffn_norm on the host, then every FFN through LFM2.5's token-batched expert kernel `mbl`. The pair law above is
only the sequential program's CORE mapping: the pool itself is mbl's law (per expert [gate band b | up band b]
x 12, then the down slices), so mbl reads it in place at FF 768 / 129 pool experts. The shared expert is pool
expert 128 at weight 1 beside the router's eight (route `shared`); the dense layer is its nine 768-wide slices,
pool experts 0..8, at weight 1 (route `dense`). hyv3-block-runtime.patch adds those two fields to kind `lfm`.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from .catalogue import OpRangeError, check_buffer_args
from .qwen36moe import BAND_ROWS, ELEM, GEMM_T, MB, roundup, tab_bytes
from .spec import DENSE, ModelSpec

CH = 5120                 # a q4_1 chunk: 32 rows x 256 K, bf16 d | m planes (gemv_q4.h)
PER_CALL = 2              # chunks per w element (10240 B)
N_CORES = 8
CPS = 4                   # cores a slot's gate | up runs on (the pair law: two slots across the 8)
WV = 16                   # floats of the routing-weight record beside xm (the slots, padded)
ATTN_CORES = 4            # attnknobs.attn_cores(32): the largest divisor of 32 heads <= 6
STACK = 0x1800
LM_BAND_ROWS = 128        # lm_head_q8: 128-row bands
LM_TILE = 9216            # lm_head_q8.h's f32-scale chunk: 32 rows x 256 K
LM_CORES = 8
KIND_DENSE, KIND_MOE = "dense", "moe"


def band_bytes(K: int) -> int:
    """64 rows x K: K/256 k-tiles x 2 thirty-two-row chunks of CH bytes."""
    return K // 128 * CH


@dataclass(frozen=True)
class Geometry:
    HID: int; FF: int; DFF: int; SFF: int
    NH: int; KVH: int; HD: int; ROT: int; QW: int; KVW: int
    N_CORES: int; PER_CALL: int; CALL_BYTES: int
    Q_PC: int; KV_PC: int; O_PC: int; UG_PC: int; DOWN_PC: int; CPS: int
    E_A: int; HPE: int; HPO: int; Q_AIN_ELEMS: int; K_AIN_ELEMS: int
    XN_ELEMS: int; XM_ELEMS: int; OG_ELEMS: int; H_ELEMS: int; H_STRIDE: int
    TAB_BYTES: int; MS_U: int; MS_G: int; MS_FLOATS: int
    EPS: float; TOPK: int; EXPERTS: int; SHARED: int; SLOTS: int; POOL_EXPERTS: int
    ACORES: int; NHL: int; VEXP: int; MLS: int; RB: int
    PTAB_ELEMS: int; W_OFF: int


@dataclass(frozen=True)
class Layout:
    CD_LNW: int; CD_POSTLN: int; CD_META: int; CD_BYTES: int
    AD_XN: int; AD_Q: int; AD_KVN: int; AD_OG: int; AD_OUT: int; AD_RES: int; AD_XM: int; AD_W: int
    AD_H: int; AD_OUT2: int; AD_JUNK: int; AD_BYTES: int
    UP_BYTES: int; EXPERT_UG: int
    POOL_DOWN: int; POOL_ATTN: int; POOL_Q: int; POOL_K: int; POOL_V: int; POOL_O: int; POOL_BYTES: int
    KV_ROW: int; PTAB_ROW: int; MAX_CTX: int; KV_BYTES: int; PTAB_BYTES: int
    LM_ROWS: int; LMHEAD_BANDS: int; LMHEAD_BAND_BYTES: int; LMHEAD_POOL_BYTES: int
    ELN: int


@dataclass(frozen=True)
class HyV3Recipe:
    spec: ModelSpec
    layout: Layout
    geo: Geometry
    max_ctx: int
    kind: str = "hyv3"


def router_scale(spec: ModelSpec) -> float:
    """router_scaling_factor (host arithmetic: spec.extra, outside the kernel hash)."""
    return float(spec.extra.get("router_scale", 1.0))


def _check(spec: ModelSpec) -> None:
    if spec.family != "hyv3":
        raise OpRangeError(f"hyv3 recipe given a {spec.family!r} spec")
    if set(spec.layer_types) != {DENSE}:
        raise OpRangeError("hyv3: every layer is full causal attention (dense)")
    if spec.activation != "silu" or not spec.num_experts or not spec.moe_intermediate:
        raise OpRangeError("hyv3: expected routed SiLU experts")
    ff = spec.moe_intermediate
    if not spec.shared_expert_intermediate or spec.shared_expert_intermediate % ff:
        raise OpRangeError(f"hyv3: a shared expert of {spec.shared_expert_intermediate} is not whole expert widths ({ff})")
    slots = spec.experts_per_tok + spec.shared_expert_intermediate // ff
    if slots > WV:
        raise OpRangeError(f"hyv3: {slots} slots exceed the {WV}-entry weight record")
    if spec.dense_lead and spec.intermediate != slots * ff:
        raise OpRangeError(f"hyv3: the dense MLP {spec.intermediate} is not the MoE layers' {slots} slots of {ff} "
                           "(the dense layers run the same hy_moe program)")
    if not spec.qk_norm or spec.attn_gate:
        raise OpRangeError("hyv3: q/k norms and no attention gate")
    if spec.rotary_dim != spec.head_dim or spec.rope_scaling or spec.sliding_window:
        raise OpRangeError("hyv3: a full-head default RoPE, no window")
    if spec.num_kv_heads % 2 or spec.num_heads % spec.num_kv_heads or spec.num_heads % ATTN_CORES:
        raise OpRangeError(f"hyv3: {spec.num_heads} q / {spec.num_kv_heads} kv heads do not split "
                           f"over the attention elements and {ATTN_CORES} cores")
    qw, kvw = spec.num_heads * spec.head_dim, spec.num_kv_heads * spec.head_dim
    for what, w in (("hidden", spec.hidden), ("q width", qw), ("kv width", kvw)):
        if w % (BAND_ROWS * N_CORES) or w % 256:
            raise OpRangeError(f"hyv3: {what} {w} is not whole 64-row bands over {N_CORES} cores and 256-wide k-tiles")
    if ff % (BAND_ROWS * CPS) or ff % 256:
        raise OpRangeError(f"hyv3: expert width {ff} is not whole 64-row bands over {CPS} cores and 256-wide k-tiles")
    if 4 * spec.head_dim > 2 * kvw:
        raise OpRangeError("hyv3: qn | kn do not fit one meta element")


def geometry(spec: ModelSpec) -> Geometry:
    _check(spec)
    hid, ff = spec.hidden, spec.moe_intermediate
    nh, kvh, hd = spec.num_heads, spec.num_kv_heads, spec.head_dim
    qw, kvw = nh * hd, kvh * hd
    e_a = kvw * 2
    ptab_row = max(1024, e_a)
    nhl = nh // ATTN_CORES
    xm_bytes = hid * 2
    shared = spec.shared_expert_intermediate // ff
    slots = spec.experts_per_tok + shared
    return Geometry(
        HID=hid, FF=ff, DFF=spec.intermediate, SFF=spec.shared_expert_intermediate,
        NH=nh, KVH=kvh, HD=hd, ROT=spec.rotary_dim, QW=qw, KVW=kvw,
        N_CORES=N_CORES, PER_CALL=PER_CALL, CALL_BYTES=PER_CALL * CH,
        Q_PC=qw // BAND_ROWS // N_CORES, KV_PC=kvw // BAND_ROWS // N_CORES, O_PC=hid // BAND_ROWS // N_CORES,
        UG_PC=ff // BAND_ROWS // CPS, DOWN_PC=hid // BAND_ROWS // N_CORES, CPS=CPS,
        E_A=e_a, HPE=e_a // (hd * 4), HPO=e_a // (hd * 2), Q_AIN_ELEMS=nh // (e_a // (hd * 4)),
        K_AIN_ELEMS=kvh // (e_a // (hd * 4)),
        XN_ELEMS=roundup(hid * 2, ELEM) // ELEM, XM_ELEMS=roundup(xm_bytes + 4 * WV, ELEM) // ELEM,
        OG_ELEMS=roundup(qw * 2, ELEM) // ELEM, H_ELEMS=roundup(ff * 4, ELEM) // ELEM, H_STRIDE=roundup(ff * 4, ELEM),
        TAB_BYTES=tab_bytes(max(hid, qw, ff)), MS_U=0, MS_G=BAND_ROWS, MS_FLOATS=3 * BAND_ROWS,   # u | g | a down band
        EPS=spec.norm_eps, TOPK=spec.experts_per_tok, EXPERTS=spec.num_experts, SHARED=shared,
        SLOTS=slots, POOL_EXPERTS=spec.num_experts + shared,
        # Flint's attention point (RB 4, the vector softmax, 8 heads a core at head dim 128), less the sink.
        ACORES=ATTN_CORES, NHL=nhl, VEXP=1, MLS=roundup(nhl, 32), RB=4,
        PTAB_ELEMS=ptab_row // e_a,
        W_OFF=xm_bytes % ELEM,        # the routing weights' byte offset in xm's last x element
    )


def lm_rows(spec: ModelSpec) -> int:
    return roundup(spec.vocab, LM_BAND_ROWS)


def slot_cores(G: Geometry, s: int) -> range:
    """The cores slot s's gate | up runs on: the pair law (slot 2v on cores 0-3, 2v + 1 on 4-7)."""
    h = s % 2
    return range(h * G.CPS, (h + 1) * G.CPS)


def core_slots(G: Geometry, c: int) -> list[int]:
    """The slots core c takes in the gate | up phase, in order."""
    return [s for s in range(G.SLOTS) if c in slot_cores(G, s)]


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Layout:
    G = geometry(spec)
    eln = G.HID * 2
    cd = {"lnw": 0, "postln": eln, "meta": 2 * eln}
    cd_bytes = roundup(2 * eln + G.E_A, ELEM)
    a: dict[str, int] = {}
    off = 0
    for name, size in (("xn", G.XN_ELEMS * ELEM), ("q", G.QW * 4), ("kvn", 2 * G.KVW * 4), ("og", G.OG_ELEMS * ELEM),
                       ("out", G.HID * 4), ("res", G.HID * 4), ("xm", G.XM_ELEMS * ELEM),
                       ("h", G.SLOTS * G.H_STRIDE), ("out2", G.HID * 4), ("junk", eln)):
        a[name] = off
        off += size
    ad_bytes = roundup(off, ELEM)
    up = (G.FF // BAND_ROWS) * band_bytes(G.HID)       # one slot's gate (= up) projection
    down = (G.HID // BAND_ROWS) * band_bytes(G.FF)
    if up != down:
        raise OpRangeError("hyv3: a slot's gate and down must be the same size for the routed-expert patch (moe2)")
    n = G.POOL_EXPERTS
    pool_down = n * 2 * up
    p_attn = pool_down + n * up
    p = {"q": p_attn}
    p["k"] = p["q"] + (G.QW // BAND_ROWS) * band_bytes(G.HID)
    p["v"] = p["k"] + (G.KVW // BAND_ROWS) * band_bytes(G.HID)
    p["o"] = p["v"] + (G.KVW // BAND_ROWS) * band_bytes(G.HID)
    pool_bytes = roundup(p["o"] + (G.HID // BAND_ROWS) * band_bytes(G.QW), MB)
    if pool_bytes >= 1 << 32:
        raise OpRangeError("hyv3: a layer pool past 4 GiB does not fit a 32-bit fill offset")
    rows = lm_rows(spec)
    bands = rows // LM_BAND_ROWS
    lm_band = 4 * (G.HID // 256) * LM_TILE
    kv_row = 2 * G.E_A
    ptab_row = G.PTAB_ELEMS * G.E_A
    return Layout(
        CD_LNW=cd["lnw"], CD_POSTLN=cd["postln"], CD_META=cd["meta"], CD_BYTES=cd_bytes,
        AD_XN=a["xn"], AD_Q=a["q"], AD_KVN=a["kvn"], AD_OG=a["og"], AD_OUT=a["out"], AD_RES=a["res"],
        AD_XM=a["xm"], AD_W=a["xm"] + (G.XM_ELEMS - 1) * ELEM + G.W_OFF, AD_H=a["h"], AD_OUT2=a["out2"],
        AD_JUNK=a["junk"], AD_BYTES=ad_bytes,
        UP_BYTES=up, EXPERT_UG=2 * up,
        POOL_DOWN=pool_down, POOL_ATTN=p_attn, POOL_Q=p["q"], POOL_K=p["k"], POOL_V=p["v"], POOL_O=p["o"],
        POOL_BYTES=pool_bytes,
        KV_ROW=kv_row, PTAB_ROW=ptab_row, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row, PTAB_BYTES=max_ctx * ptab_row,
        LM_ROWS=rows, LMHEAD_BANDS=bands, LMHEAD_BAND_BYTES=lm_band,
        LMHEAD_POOL_BYTES=roundup(roundup(bands, LM_CORES) * lm_band, MB),
        ELN=eln,
    )


def kinds(spec: ModelSpec) -> list[str]:
    return [KIND_DENSE if i < spec.dense_lead else KIND_MOE for i in range(spec.num_layers)]


def layer_type_names(spec: ModelSpec) -> list[str]:
    """The manifest's per-layer type names: `dense_mlp` for the dense-MLP layers, else the spec's own."""
    return ["dense_mlp" if k == KIND_DENSE else t for k, t in zip(kinds(spec), spec.layer_types)]


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> HyV3Recipe:
    return HyV3Recipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def attn_flags(G: Geometry, probe_env=()) -> list[str]:
    """attn.h for HY V3: Qwen3's q / k norm before the rotation, no gate, no sink, the vector softmax
    over 4 cores of 8 heads, RB 4."""
    flags = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
             "-DATTN_QKNORM=1", "-DATTN_QKNORM_POST=0", f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}",
             f"-DATTN_NHL={G.NHL}"]
    if G.RB > 1:
        flags.append(f"-DATTN_RB={G.RB}")
    if G.PTAB_ELEMS > 1:
        flags.append("-DATTN_PTAB_SPLIT=1")
    for k, v in probe_env:
        flags.append(f"-D{k}={v}")
    return flags


def ln_flags(G: Geometry) -> list[str]:
    return [f"-DLN_N={G.HID}", f"-DLN_EPS={G.EPS:g}f"]


PROBE_VARS = ("ATTN_NULL", "HY_NULL")


def probe_env() -> dict[str, str]:
    return {k: os.environ[k] for k in PROBE_VARS if os.environ.get(k)}


def gguf_quant(spec: ModelSpec) -> str:
    """The law the GGUF view's pools hold: q4_1 (our pure Q4_0 exactly: m = -8 d)."""
    return "q4_1"


def finalize_gguf_manifest(spec: ModelSpec, manifest: dict) -> None:
    """The manifest IS the GGUF view (there is no q4nx container): core.cpp reads a model.gguf through
    the nested `gguf` section, so the section is the same manifest, nested once. And the per-layer type
    names: the dense-MLP layer is its own type (`dense_mlp`)."""
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
        raise OpRangeError(f"hyv3: {tensor}: {rows} x {in_dim} at column {col0} is not whole bands and k-tiles")
    return {"op": "fl_q4", "tensor": tensor, "dst": dst, "rows": rows, "in_dim": in_dim, "row0": row0,
            "chunk0": col0, "experts": experts, "n_total": row_step, "cols": col_step,
            "expert_bytes": expert_bytes, "stripe_bytes": band_stride}


def _layer_plan(spec: ModelSpec, dense: bool) -> dict:
    G, L = geometry(spec), layout(spec)
    b = "blk.{l}."
    ff, hid, bbh = G.FF, G.HID, band_bytes(G.HID)
    ug = dict(expert_bytes=L.EXPERT_UG, band_stride=2 * bbh)
    if dense:
        n = G.SLOTS          # the MLP's rows (gate, up) and columns (down) in expert-wide slices
        ffn = [
            _q4_op(b + "ffn_gate.weight", 0, ff, hid, experts=n, row_step=ff, **ug),
            _q4_op(b + "ffn_up.weight", bbh, ff, hid, experts=n, row_step=ff, **ug),
            _q4_op(b + "ffn_down.weight", L.POOL_DOWN, hid, ff, experts=n, col_step=ff, expert_bytes=L.UP_BYTES),
        ]
    else:
        E = spec.num_experts     # the shared expert's slices are pool experts E .. E + SHARED - 1
        ffn = [
            _q4_op(b + "ffn_gate_exps.weight", 0, ff, hid, experts=E, row_step=ff, **ug),
            _q4_op(b + "ffn_up_exps.weight", bbh, ff, hid, experts=E, row_step=ff, **ug),
            _q4_op(b + "ffn_down_exps.weight", L.POOL_DOWN, hid, ff, experts=E, row_step=hid,
                   expert_bytes=L.UP_BYTES),
            _q4_op(b + "ffn_gate_shexp.weight", E * L.EXPERT_UG, ff, hid, experts=G.SHARED, row_step=ff, **ug),
            _q4_op(b + "ffn_up_shexp.weight", E * L.EXPERT_UG + bbh, ff, hid, experts=G.SHARED, row_step=ff, **ug),
            _q4_op(b + "ffn_down_shexp.weight", L.POOL_DOWN + E * L.UP_BYTES, hid, ff, experts=G.SHARED,
                   col_step=ff, expert_bytes=L.UP_BYTES),
        ]
    attn = [_q4_op(b + "attn_q.weight", L.POOL_Q, G.QW, hid), _q4_op(b + "attn_k.weight", L.POOL_K, G.KVW, hid),
            _q4_op(b + "attn_v.weight", L.POOL_V, G.KVW, hid), _q4_op(b + "attn_output.weight", L.POOL_O, hid, G.QW)]
    consts = [
        {"op": "put", "tensor": b + "attn_norm.weight", "dst": L.CD_LNW, "cap": L.ELN},
        {"op": "put", "tensor": b + "ffn_norm.weight", "dst": L.CD_POSTLN, "cap": L.ELN},
        {"op": "put", "tensor": b + "attn_q_norm.weight", "dst": L.CD_META, "cap": 2 * G.HD},
        {"op": "put", "tensor": b + "attn_k_norm.weight", "dst": L.CD_META + 2 * G.HD, "cap": 2 * G.HD},
    ]
    return {"pool": ffn + attn, "consts": consts}


def pack_plan(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    out = {}
    for nm, k in zip(layer_type_names(spec), kinds(spec)):
        out.setdefault(nm, _layer_plan(spec, k == KIND_DENSE))
    return {
        "pool_bytes": L.POOL_BYTES, "chunk_bytes": CH,
        "layer_types": out,
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "gx_lmhead", "tensor": "output.weight", "dst": 0, "rows": L.LM_ROWS,
                             "in_dim": G.HID}]},
        "embed": {"tensor": "token_embd.weight", "dim": G.HID},
        "norm": {"tensor": "output_norm.weight", "bytes": L.ELN},
    }


def route_step(spec: ModelSpec, dense: bool) -> dict:
    """hy_route (hyv3-runtime.patch). MoE mode: the router; dense mode: slots 0..SLOTS-1 = pool
    experts 0..SLOTS-1 at weight 1 (the dense MLP's slices), so the shared stream is re-pointed there."""
    L, G = layout(spec), geometry(spec)
    st = {"op": "hy_route", "kernel": "hy_moe", "a_xm": L.AD_XM, "a_w": L.AD_W, "hidden_real": G.HID,
          "slots": G.SLOTS}
    if dense:
        st["dense"] = True
    else:
        st.update({"router": "blk.{l}.ffn_gate_inp.weight", "bias": "blk.{l}.exp_probs_b",
                   "experts": spec.num_experts, "topk": spec.experts_per_tok, "scale": router_scale(spec),
                   "shared": [spec.num_experts + i for i in range(G.SHARED)]})
    return st


# ---- block prefill (LFM2.5's route: recipes/lfm2moe.py gemm_route; .plans/hymemory-native.md "block prefill")
def block_on() -> bool:
    """The block prefill route's kernels and manifest keys: on by default (HYV3_BLOCK=0 drops them: sequential
    prefill, and programs() / builds() are the decode set's alone; pack_plan() never changes)."""
    return os.environ.get("HYV3_BLOCK") != "0"


MB_NT = 8             # designs/lfm2moe_batch/mbl.py token slots per expert visit (one 8-token MAC tile)
MB_MAX = 64           # the longest mbl stream (LFM2.5's; built and run there)


def moe_batch_slots(spec: ModelSpec) -> list[int]:
    """mbl's dispatch lengths, longest first: every multiple of 8 up to 64. A layer's visits (~330 for a 256-token
    block: 8 routed + the shared expert, at 8 tokens a visit) go out 64 at a time; the remainder takes the shortest
    stream that holds it, so an unused slot (which streams a real expert, 2.95 MB, for nothing) is at most 7."""
    G = geometry(spec)
    top = min(MB_MAX, G.POOL_EXPERTS // 8 * 8)
    return list(range(top, 0, -8))


def moe_batch(spec: ModelSpec) -> dict:
    """LFM2.5's token-batched expert kernel (designs/lfm2moe_batch/mbl.py, unchanged) at Hy's widths: the
    `moe_batch` entry of every layer type's gemm_block, plus the contexts / kernels / globals / builds it needs.
    One xclbin (context `mb`), one instruction stream per dispatch length; the runtime re-points the slots'
    expert fills (`moebatch`: moeroute2's table with every pool expert a slot, the manifest's moe geometry).
    mbl's taps read [gate b | up b] x FF/64 bands at slot x EXPERT_UG and the down slices at POOL_DOWN + slot x
    UP_BYTES -- what _layer_plan writes for the routed experts, the shared expert (pool expert 128) and the dense
    layer's slices alike. y leaves [rbg][half][row][1024 B] (designs/lfm2moe_batch/layout.py)."""
    G, L = geometry(spec), layout(spec)
    slots = moe_batch_slots(spec)
    args = ["pool", "mb_x", "mb_h", "mb_y"]
    check_buffer_args("moe_batch", args)
    out: dict = {"contexts": {}, "kernels": {}, "globals": {}, "builds": {}}
    for s in slots:
        name = f"mb_s{s}"
        out["contexts"].setdefault("mb", f"mb_s{slots[0]}/final.xclbin")
        out["kernels"][name] = {"context": "mb", "insts": f"{name}/insts.bin", "patch": "moebatch", "build": name}
        out["builds"][name] = {"design": "lfm2moe_batch/mbl.py",
                               "build_dir": f"lfm2moe_batch/build_s{s}_h{G.HID}_f{G.FF}_e{G.POOL_EXPERTS}",
                               "env": {"MBL_SLOTS": str(s), "MBL_NT": str(MB_NT), "MBL_HID": str(G.HID),
                                       "MBL_FF": str(G.FF), "MBL_EXPERTS": str(G.POOL_EXPERTS),
                                       "MBL_POOL_DOWN": str(L.POOL_DOWN), "MBL_POOL_BYTES": str(L.POOL_ATTN)}}
    out["globals"]["mb_x"] = slots[0] * G.HID * MB_NT * 2
    out["globals"]["mb_h"] = slots[0] * G.FF * MB_NT * 2
    out["globals"]["mb_y"] = slots[0] * G.HID * MB_NT * 4
    out["moe_batch"] = {"kernels": {str(s): f"mb_s{s}" for s in slots}, "args": args, "nt": MB_NT,
                        "y_layout": "rbg_half_row"}
    return out


def gemm_route(spec: ModelSpec, max_ctx: int = 4096) -> dict | None:
    """Per layer type the `gemm_block` (kind `lfm`, mixer `att`) the driver reads, plus the contexts / kernels /
    globals / builds the route adds. None under HYV3_BLOCK=0. A layer: the host entry norm, the q | k | v GEMM
    (three contiguous pool ops), host attention over the block (q / k RMSNorm, NEOX RoPE, causal, no gate), the o
    GEMM, the host residual and ffn_norm, then the FFN through mbl: the host router (hy_route's arithmetic) and the
    shared expert at weight 1, or the dense layer's nine slices at weight 1. Nothing is added to the pool."""
    if not block_on():
        return None
    L, G, T = layout(spec, max_ctx), geometry(spec), GEMM_T
    hid, qw, kvw = G.HID, G.QW, G.KVW
    shapes: set[tuple[int, int]] = set()

    def run(N: int, K: int, w: str) -> dict:
        if N % 256 or K % 256:
            raise OpRangeError(f"hyv3 gemm route: [{N}, {K}] is not a multiple of 256 in both dims")
        shapes.add((N, K))
        return {"op": "run", "kernel": f"gemm_n{N}_k{K}", "args": [w, f"gemm_x_k{K}", f"gemm_y_n{N}"]}

    def idx(pool: list[dict], suffix: str) -> int:
        hits = [i for i, o in enumerate(pool) if o["tensor"].endswith(suffix)]
        if len(hits) != 1:
            raise OpRangeError(f"hyv3 gemm route: {len(hits)} pack ops end in {suffix!r}, want one")
        return hits[0]

    def run_of(pool: list[dict], ids: list[int]) -> dict:
        ops = [pool[i] for i in ids]
        for o in ops:
            if o["experts"] != 1 or o["stripe_bytes"]:
                raise OpRangeError("hyv3 gemm route: a GEMM weight must be one contiguous tensor op")
        for a, b in zip(ops, ops[1:]):       # one byte-contiguous run, as the engine needs
            if b["dst"] != a["dst"] + a["rows"] // BAND_ROWS * band_bytes(a["in_dim"]):
                raise OpRangeError("hyv3 gemm route: the pool ops of a GEMM weight are not contiguous")
        return {"from": "pool", "ops": ids}

    mbatch = moe_batch(spec)
    out: dict = {"layer_types": {}, "contexts": {}, "kernels": {}, "globals": {}, "builds": {}}
    for k in ("contexts", "kernels", "globals", "builds"):
        out[k].update(mbatch[k])
    for nm, k in zip(layer_type_names(spec), kinds(spec)):
        if nm in out["layer_types"]:
            continue
        dense = k == KIND_DENSE
        pool = _layer_plan(spec, dense)["pool"]
        w = {"gqkv3_w": run_of(pool, [idx(pool, "attn_q.weight"), idx(pool, "attn_k.weight"), idx(pool, "attn_v.weight")]),
             "go_w": run_of(pool, [idx(pool, "attn_output.weight")])}
        if dense:
            route = {"dense": G.SLOTS}
        else:
            r = route_step(spec, dense=False)
            route = {"router": r["router"], "bias": r["bias"], "experts": r["experts"], "topk": r["topk"],
                     "scale": r["scale"], "shared": r["shared"]}
        out["layer_types"][nm] = {
            "kind": "lfm", "mixer": "att", "t": T, "eps": spec.norm_eps, "cd_ln": L.CD_LNW, "cd_ffn": L.CD_POSTLN,
            "program": [run(qw + 2 * kvw, hid, "gqkv3_w"), run(hid, qw, "go_w")], "weights": w,
            "qw": qw, "kvw": kvw, "nh": G.NH, "kvh": G.KVH, "hd": G.HD, "rot": spec.rotary_dim,
            "cd_qnorm": L.CD_META, "cd_knorm": L.CD_META + 2 * G.HD,
            "route": route, "moe_batch": mbatch["moe_batch"]}
    for N, K in sorted(shapes):
        name = f"gemm_n{N}_k{K}"
        out["contexts"].setdefault("gemm", f"{name}/final.xclbin")
        out["kernels"][name] = {"context": "gemm", "insts": f"{name}/insts.bin", "build": name}
        out["globals"][f"gemm_x_k{K}"] = K * T * 2
        out["globals"][f"gemm_y_n{N}"] = max(out["globals"].get(f"gemm_y_n{N}", 0), N * T * 4)
        out["builds"][name] = {"design": "gemm_q4_prefill/gemm_q4_prefill.py",
                               "build_dir": f"gemm_q4_prefill/build_n{N}_k{K}_t{T}",
                               "env": {"GQP_N": str(N), "GQP_K": str(K), "GQP_T": str(T)}}
    return out


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    args = ["pool", "xres", "consts", "state", "act", "ptab"]
    check_buffer_args("hy_att", args)
    out = {
        "contexts": {"hy_att": "hy_att/final.xclbin", "hy_moe": "hy_moe/final.xclbin",
                     "ln": "ln/final.xclbin", "lm": "lm_head_q8/final.xclbin"},
        "kernels": {
            "hy_att": {"context": "hy_att", "insts": "hy_att/insts.bin", "patch": "attnpos", "build": "hy_att",
                       "window": 0},
            "hy_moe": {"context": "hy_moe", "insts": "hy_moe/insts.bin", "patch": "moeroute2", "build": "hy_moe"},
            "ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
            "lm": {"context": "lm", "insts": "lm_head_q8/insts.bin", "build": "lm_head_q8"},
        },
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": G.HID * 4, "zero": G.HID * 4, "normw": G.HID * 2, "xresf": G.HID * 4, "hn": G.HID * 2,
                    "logits": L.LM_ROWS * 4, "lmpool": L.LMHEAD_POOL_BYTES,
                    "ptab": {"per_row": L.PTAB_ROW, "inv_freq": spec.rope_inv_freq(), "window": 0}},
    }
    for nm, k in zip(layer_type_names(spec), kinds(spec)):
        if nm in out["layer_types"]:
            continue
        out["layer_types"][nm] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": "hy_att", "args": args}, route_step(spec, k == KIND_DENSE),
                        {"op": "run", "kernel": "hy_moe", "args": ["pool", "xres", "act"]}],
        }
    gr = gemm_route(spec, max_ctx)
    if gr:
        for lt, gb in gr["layer_types"].items():
            out["layer_types"][lt]["gemm_block"] = gb
        for k in ("contexts", "kernels"):
            out[k].update(gr[k])
        for k, v in gr["globals"].items():
            out["globals"][k] = max(out["globals"].get(k, 0), v)
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    G, L = geometry(spec), layout(spec)
    out = {
        "hy_att": {"design": "hyv3/hy.py", "build_dir": f"hyv3/build_att_h{G.HID}",
                   "env": {"HY_PART": "att"}},
        "hy_moe": {"design": "hyv3/hy.py", "build_dir": f"hyv3/build_moe_h{G.HID}_f{G.FF}_s{G.SLOTS}",
                   "env": {"HY_PART": "moe"}},
        "ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{G.HID}_{G.EPS:g}",
               "env": {"LN_N": str(G.HID), "LN_EPS": f"{G.EPS:g}"}},
        "lm_head_q8": {"design": "lm_head_q8/lm_head_q8.py",
                       "build_dir": f"lm_head_q8/build_gguf_n{L.LM_ROWS}_k{G.HID}_c{LM_CORES}",
                       "env": {"LMHEAD_N": str(L.LM_ROWS), "LMHEAD_K": str(G.HID), "LMHEAD_CORES": str(LM_CORES),
                               "LMHEAD_SCALES_F32": "1"}},
    }
    gr = gemm_route(spec)
    if gr:
        out.update(gr["builds"])
    return out


def moe_geometry(spec: ModelSpec) -> dict:
    """stream_patch::MoeGeometry for moeroute2 over hy_moe's fills: slot j is compiled as pool expert j
    (each core's gate | up fill inside its own slot's stripe set), and the pool holds the routed experts
    and the shared expert (num_experts + SHARED) from byte 0."""
    L, G = layout(spec), geometry(spec)
    return {"stripe": L.UP_BYTES, "up_bytes": L.UP_BYTES, "down_core": L.UP_BYTES, "pool_down": L.POOL_DOWN,
            "share_up": L.POOL_ATTN, "share_gate": L.POOL_ATTN, "share_down": L.POOL_ATTN,
            "experts": G.POOL_EXPERTS, "topk": G.SLOTS}


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    return {"hidden": G.HID, "vocab": L.LM_ROWS, "real_vocab": spec.real_vocab, "chunk_bytes": CH,
            "pool_bytes": L.POOL_BYTES, "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": LM_TILE,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
            "rope_inv_freq": spec.rope_inv_freq(), "moe": moe_geometry(spec)}


def hf_config_check(spec: ModelSpec) -> dict:
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim, "intermediate_size": spec.intermediate,
            "expert_hidden_dim": spec.moe_intermediate, "num_experts": spec.num_experts,
            "num_experts_per_tok": spec.experts_per_tok,
            "num_shared_experts": spec.shared_expert_intermediate // spec.moe_intermediate,
            "router_scaling_factor": router_scale(spec),
            "moe_router_use_sigmoid": True, "moe_router_enable_expert_bias": True}


GEN_KERNELS = "designs/hyv3/gen_kernels.py"
KERNEL_SOURCES = [
    "designs/hyv3/*.py", "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h",
    "designs/lfm2moe_batch/*.py", "designs/lfm2moe_batch/*.cc", "designs/lfm2moe_batch/*.h",
    "designs/gemm_q4_prefill/*.py", "designs/gemm_q4_prefill/*.h", "designs/gemm_q4_prefill/*.cc",
    "designs/attn/*.cc", "designs/attn/*.h", "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py",
    "designs/lin_layer/ln_nr.cc", "designs/lm_head_q8/*.py", "designs/lm_head_q8/*.h", "designs/lm_head_q8/*.cc",
    "include/vecmath.h", "include/scalar_fp.h", "ironutil.py", "build_design.py",
]
