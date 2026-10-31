"""Gemma 4 12B (OPEN-FAMILY-GEMMA4): the dense recipe, twice, in one model.

The 40 sliding layers ARE Gemma 3 12B's dense_local layer -- hidden 3840, 16 q / 8 kv heads at
head dim 256, a 1024-row window, GeGLU-tanh 15360, sandwich norms -- and run designs/dense/dx.py
unchanged in shape. The 8 global layers have a geometry of their own: head dim 512 and ONE kv
head. `ModelSpec`, `dense.geometry()` and `dense.layout()` all describe one geometry per spec,
so this module derives a second spec for the global layers and builds dx.py once per layer type
(G4_KIND, below): two xclbins, two hardware contexts, two layouts, two position tables. The
engine runs a different context per layer type already (LFM2, Qwen3.5); what it needed was the
attention patch's KV / position-record row per KERNEL rather than per model (manifest
`kernels.*.kv_row` / `ptab_row`).

What Gemma 4 changes inside a layer, every one from transformers 5.15 `modeling_gemma4.py`
and checked in the fp64-free reference before anything here relied on it
(NpuEmbeddings tasks/0184 `g4ref.py`):

* score scale 1.0, not 1/sqrt(HD): q_norm / k_norm carry it (attn.h ATTN_SCALE).
* a weightless RMSNorm on V, every layer (attn.h ATTN_VNORM).
* the global layers have no v_proj (`attention_k_eq_v`): V is the RAW k projection, normed
  without weight and not rotated, while K is k_norm'd and rotated. Packing k_proj into the V
  slot as well gives exactly that, since the attention core already norms (V) and norms +
  rotates (K) the two slots separately.
* one kv head is an odd count, and the attention element carries KVH/2 heads; the head is
  DUPLICATED (two pack ops of the same tensor), which is exact -- every q head of the group
  reads an identical row -- for one extra 512-wide K and V row on 8 layers.
* proportional RoPE on the global layers: 64 pairs turn, pairing dim i with i + 256, at
  theta^(-2i/512) (spec.rope_inv_freq; attn.h ATTN_ROPE_PAIR). The record holds 64 cos / sin.
* a per-layer output scalar, folded into the two post norms by the packer (`div_running`).
* a tanh cap on the final logits (the manifest's layout.final_logit_softcapping, on the host).
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, replace

from . import dense as D
from .catalogue import OpRangeError, check_buffer_args
from .qwen36moe import ATTN_LMAX, MB, gguf_direct, proj_op, q4_chunks, role_bytes, roundup
from .spec import DENSE, DENSE_LOCAL, ModelSpec

band_bytes = D.band_bytes    # noqa: F401  (dx.py: band_bytes(K, quant))
chunk_bytes = D.chunk_bytes  # noqa: F401  (dx.py: chunk_bytes(quant))
attn_flags = D.attn_flags    # noqa: F401  (phlegm's dx.py: attn_flags(G, probe_env))
# The build key's probe variables (cache.py reads it off the family module): the attention ones
# and dense's, DENSE_GGUF_GEMM among them -- it adds the block prefill route's builds.
G4_PROBE_VARS = ("G4_ATTN_BLOCK", "G4_BFP16", "G4_Q40", "G4_ACORES", "G4_RB", "G4_KVSLICE")


def probe_env() -> dict[str, str]:
    """dense's probe variables and this family's own (G4_PROBE_VARS)."""
    env = D.probe_env()
    env.update({k: os.environ[k] for k in G4_PROBE_VARS if os.environ.get(k)})
    return env

KIND_ENV = "G4_KIND"          # which layer type a dx.py build is for: "local" | "global"
# A pool law's build-name suffix: the q4nx section's, the f32-scale twin's, the Q4_0-native view's.
SFX = {"q4_1": "", "q4_1_f32": "_f32", "q4_0": "_q40"}


def q40_on() -> bool:
    """The GGUF view on Q4_0-native pools: 4608-B chunks holding the codes and the bf16 d plane,
    the cores taking m = -8 d (gemv_q4.h GEMV_Q4_0_NATIVE, gemm_q4_dequant.h GQD_Q4_0), so the
    products are the compact view's with a tenth fewer weight bytes. On unless G4_Q40=0, which
    gives the compact view: every prefill and decode logit matched it, and decode took 6-7%
    less time (OPEN-FAMILY-GEMMA4). Every tensor the pools hold must be Q4_0, as in the QAT
    Q4_0 GGUF; the packer refuses anything else, and such a GGUF needs G4_Q40=0."""
    return os.environ.get("G4_Q40") != "0"


def gguf_quant(spec: ModelSpec) -> str:
    """The pool law recipes/manifest.py builds the GGUF section at: Q4_0-native (q40_on, the
    default), else the f32-scale twin, which finalize_gguf_manifest turns into the compact view."""
    return "q4_0" if q40_on() else "q4_1_f32"
KINDS = {"local": DENSE_LOCAL, "global": DENSE}
LAYER_SCALE = "model.layers.{j}.layer_output_scale.weight"   # GGUF blk.{j}.layer_output_scale.weight


def _kind(kind: str | None) -> str:
    k = kind or os.environ.get(KIND_ENV, "local")
    if k not in KINDS:
        raise OpRangeError(f"gemma4: {KIND_ENV}={k!r} (local | global)")
    return k


def sub_spec(spec: ModelSpec, kind: str) -> ModelSpec:
    """The dense spec one layer type is built from. Local: the model's own head fields, every
    layer sliding. Global: head dim 512, the kv head doubled (see the module docstring),
    the proportional rotation's 128 dims, no window."""
    n = spec.num_layers
    if kind == "local":
        return replace(spec, layer_types=(DENSE_LOCAL,) * n)
    return replace(spec, layer_types=(DENSE,) * n, head_dim=spec.global_head_dim or spec.head_dim,
                   num_kv_heads=2 * spec.global_kv_heads if spec.global_kv_heads % 2 else spec.global_kv_heads,
                   rotary_dim=spec.global_rotary_dim or spec.global_head_dim, sliding_window=0)


def _check(spec: ModelSpec) -> None:
    if spec.family != "gemma4":
        raise OpRangeError(f"gemma4 recipe given a {spec.family!r} spec")
    if set(spec.layer_types) != {DENSE, DENSE_LOCAL}:
        raise OpRangeError("gemma4: expected sliding (dense_local) and global (dense) layers")
    if not (spec.v_norm and spec.attn_scale == 1.0 and spec.global_head_dim and spec.global_kv_heads):
        raise OpRangeError("gemma4: the spec lacks the Gemma 4 attention fields (v_norm, attn_scale 1.0, "
                           "global_head_dim, global_kv_heads)")
    if spec.global_kv_heads not in (1, 2):
        raise OpRangeError(f"gemma4: {spec.global_kv_heads} global kv heads (the duplication covers 1)")
    if not spec.attn_k_eq_v:
        raise OpRangeError("gemma4: global layers with their own v_proj are not packed here")
    for kind in KINDS:
        s = sub_spec(spec, kind)
        D._check(s)
        if D.cores_for(s) != D.cores_for(sub_spec(spec, "local")):
            raise OpRangeError("gemma4: the two layer types split over different core counts")


def _require(spec: ModelSpec, kind: str, G: D.DenseGeometry) -> None:
    """The attention tuple, with Gemma 4's three knobs IN it: the catalogue's Gemma 3 12B point
    (256, 16, 8, 256) is not this kernel, because the same shape with a V norm, scale 1.0 and
    another pair distance computes something else."""
    D.require("attn", head_dim=G.HD, num_heads=G.NH, num_kv_heads=G.KVH, rotary_dim=G.ROT,
              rope_theta=spec.rope_theta, qk_norm=True, attn_gate=False, qk_norm_post_rope=False,
              qkv_bias=False, v_norm=bool(G.VNORM), attn_scale=G.SCALE, rope_pair=G.ROPE_PAIR)


def geometry(spec: ModelSpec, kind: str | None = None) -> D.DenseGeometry:
    k = _kind(kind)
    s = sub_spec(spec, k)
    G = D.geometry(s)
    return replace(G, VNORM=1, SCALE=float(spec.attn_scale), QLOCAL=1,
                   ROPE_PAIR=(s.head_dim // 2) if (k == "global" and s.rotary_dim < s.head_dim) else 0)


DX_ENV = "G4_DX"              # set in the decode layer's builds: recipe() then gives dx.py dx_geometry()


def dx_attention() -> tuple[int, int, int]:
    """(cores, cached rows a call, rows a window element) of the decode layer's attention.

    8 cores of 2 heads on row 4 (og joined in memtiles), 4 rows a call, and each core reading
    only its own kv heads of the window on a shim channel of its own (OPEN-ATTN-KVSLICE). The
    four-core, one-row, broadcast walk cost 1.1 ms a local layer at the full 1024-row window and
    1.26 us a row a global layer; this one took decode after a 1450-token prompt from 237 to
    199-200 ms a token on a quiet machine, and costs 1-2% at the start of a context
    (OPEN-FAMILY-GEMMA4). 16 cores (G4_ACORES=16) were as fast deep and slower at the start:
    every switch into a layer's context reloads twice the attention programs. G4_ACORES=4
    G4_RB=1 G4_KVSLICE=0 gives the old walk back. The head split and the windows leave every
    logit as it was; rows a call round the online softmax differently."""
    cores = int(os.environ.get("G4_ACORES", "8"))
    rb = int(os.environ.get("G4_RB", "4"))
    kvs = rb if os.environ.get("G4_KVSLICE", "1") == "1" and rb > 1 else 0
    if cores not in (4, 8, 16) or rb not in (1, 2, 4):
        raise OpRangeError(f"gemma4: G4_ACORES={cores} G4_RB={rb} (4 | 8 | 16 cores, 1 | 2 | 4 rows a call)")
    return cores, rb, kvs


def dx_geometry(spec: ModelSpec, kind: str | None = None) -> D.DenseGeometry:
    """geometry() with dx_attention()'s cores, rows a call and window elements. The block
    prefill's attention dispatch (dx_attn.py) keeps geometry()'s four cores and one row."""
    G = geometry(spec, kind)
    cores, rb, kvs = dx_attention()
    if (cores, rb, kvs) == (G.ACORES, G.RB, 0):
        return G
    if G.NH % cores or (rb > 1 and rb * max(G.NH // cores, 8) not in (8, 16, 32)):
        raise OpRangeError(f"gemma4: {G.NH} heads over {cores} cores at {rb} rows a call")
    return replace(G, ACORES=cores, NHL=G.NH // cores, RB=rb, AWIDE=int(cores > D.MAX_ATTN_CORES), KVSLICE=kvs)


def layout(spec: ModelSpec, max_ctx: int = 4096, kind: str | None = None) -> D.DenseLayout:
    return D.layout(sub_spec(spec, _kind(kind)), max_ctx)


def pool_bytes(spec: ModelSpec) -> int:
    """Both layer types are allocated the larger pool (the global q / o are twice as wide)."""
    return max(layout(spec, kind=k).POOL_BYTES for k in KINDS)


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> D.DenseRecipe:
    """The recipe dx.py builds: the layer type G4_KIND names (local when unset)."""
    _check(spec)
    k = _kind(None)
    G = dx_geometry(spec, k) if os.environ.get(DX_ENV) == "1" else geometry(spec, k)
    _require(spec, k, G)
    return D.DenseRecipe(spec=sub_spec(spec, k), layout=layout(spec, max_ctx, k), geo=G, max_ctx=max_ctx)


def _layer_plan(spec: ModelSpec, kind: str) -> dict:
    s = sub_spec(spec, kind)
    L, G = D.layout(s), geometry(spec, kind)
    hid, ff = spec.hidden, spec.intermediate
    pre = "model.layers.{l}."
    if kind == "local":
        pool = D.pack_plan(s)["layer_types"][DENSE_LOCAL]["pool"]
    else:
        # k_proj is one 512-row head. The attention core wants KVH = 2 heads per row half, so
        # both slots get the head twice, back to back (the pool is band-major in row order:
        # rows [k; k] are two ordinary 8-band projections end to end). The V slot is the SAME
        # tensor: attention_k_eq_v.
        kv1 = spec.global_kv_heads * s.head_dim
        dup = G.KVW // kv1
        step = role_bytes(s, "attn", kv1, hid)
        k_ops = [proj_op(s, "attn", pre + "self_attn.k_proj.weight", L.POOL_K + i * step, kv1, hid, hid)
                 for i in range(dup)]
        v_ops = [proj_op(s, "attn", pre + "self_attn.k_proj.weight", L.POOL_V + i * step, kv1, hid, hid)
                 for i in range(dup)]
        pool = ([proj_op(s, "attn", pre + "self_attn.q_proj.weight", L.POOL_Q, G.QW, hid, hid)] + k_ops + v_ops +
                [proj_op(s, "attn", pre + "self_attn.o_proj.weight", L.POOL_O, hid, G.QW, G.QW),
                 proj_op(s, "ffn", pre + "mlp.up_proj.weight", L.POOL_UP, ff, hid, hid),
                 proj_op(s, "ffn", pre + "mlp.gate_proj.weight", L.POOL_GATE, ff, hid, hid),
                 proj_op(s, "ffn", pre + "mlp.down_proj.weight", L.POOL_DOWN, hid, ff, ff)])
    # layer_output_scale folded in: the residual is carried divided by P_l = prod_{j<l} s_j,
    # which every reader of it -- an RMSNorm -- cannot see (up to eps), so the two outputs added
    # to it are divided by P_l too, and the scalar disappears. `div_running` names the scalar
    # tensor; the packer divides by the product of its value over the layers BEFORE this one.
    run = {"div_running": LAYER_SCALE}
    consts = [
        {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CD_LNW, "cap": L.ELN},
        {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CD_POSTLN, "cap": L.ELN, **run},
        {"op": "put", "tensor": pre + "self_attn.q_norm.weight", "dst": L.CD_META, "cap": G.HD * 2},
        {"op": "put", "tensor": pre + "self_attn.k_norm.weight", "dst": L.CD_META + G.HD * 2, "cap": G.HD * 2},
        {"op": "put", "tensor": pre + "pre_feedforward_layernorm.weight", "dst": L.CD_PREFFN, "cap": L.ELN},
        {"op": "put", "tensor": pre + "post_feedforward_layernorm.weight", "dst": L.CD_POSTFFN, "cap": L.ELN, **run},
    ]
    return {"pool": pool, "consts": consts}


def pack_plan(spec: ModelSpec) -> dict:
    s = sub_spec(spec, "local")
    L = D.layout(s)
    hid = spec.hidden
    gd = gguf_direct(spec.quant)
    return {
        "pool_bytes": pool_bytes(spec), "chunk_bytes": D.chunk_bytes(D.pool_quant(spec)),
        "layer_types": {KINDS[k]: _layer_plan(spec, k) for k in KINDS},
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "std_perm_gguf" if gd else "std_perm", "tensor": "lm_head.weight", "dst": 0,
                             "nch": q4_chunks(D.lm_rows(spec), hid, D.pool_quant(spec)), "in_dim": hid}]},
        # GGUF stores the raw table; HF multiplies by sqrt(hidden) at lookup (as Gemma 3)
        "embed": {"tensor": "model.embed_tokens.weight", "dim": hid, **({"scale": hid ** 0.5} if gd else {})},
        "norm": {"tensor": "model.norm.weight", "bytes": hid * 2},
    }


def _kernel_sets(spec: ModelSpec) -> dict[str, str]:
    sfx = SFX[D.pool_quant(spec)]
    return {"local": "dx" + sfx, "global": "dxg" + sfx}


def gemm_on() -> bool:
    """The block prefill route (gemm_route): on unless DENSE_GGUF_GEMM=0."""
    return os.environ.get("DENSE_GGUF_GEMM") != "0"


def attn_block_on() -> bool:
    """The block route's attention as NPU products per kv head (OPEN-PREFILL-ATTN, `attn_block`)
    instead of one dxB dispatch a token: on unless G4_ATTN_BLOCK=0. It gave the dxB route's greedy
    tokens on both prompts, a lighter KL tail and the 1024-row window (OPEN-FAMILY-GEMMA4)."""
    return os.environ.get("G4_ATTN_BLOCK") != "0"


def _attn_block(spec: ModelSpec, kind: str, L: D.DenseLayout, G: D.DenseGeometry, T: int, max_ctx: int):
    """One layer type's `attn_block`: S = Q_g K_g^T and O = P_g V_g as bf16 GEMMs per kv head
    (designs/attn_block/attn_gemm.py), the softmax and everything before it on the host
    (core.cpp dense_attention_block, prep "qkvnorm_rope"): q / k RMSNorm with the consts'
    weights, the weightless V norm, the kind's own rotation (the global one turns rot / 2 pairs
    half a head apart) and window, the unit score scale, and Q as a bf16 hi / lo pair as dx.py's
    attention core takes it, both halves in one scores product of twice the rows (`kernels_shl`;
    each row is the same dot product either way). Rows per product are the kv group's heads times T; the window tiers
    stop at the widest a block reads: (window - 1) + T rows on the local layers, ATTN_LMAX (the
    host chunks past it) on the global ones."""
    s = sub_spec(spec, kind)
    hd, grp = G.HD, G.NH // G.KVH
    m = grp * T
    window = spec.sliding_window if kind == "local" else 0
    widest = roundup(window - 1 + T, 256) if window else min(ATTN_LMAX, roundup(max_ctx, 256))
    tiers = list(range(256, widest + 1, 256))
    return {"m": m, "hd": hd, "l_max": widest, "args": ["ag_a", "ag_b", "ag_c"],
            "kernels_s": {str(n): f"ag_s{n}_{kind}" for n in tiers},
            "kernels_pv": {str(n): f"ag_pv{n}_{kind}" for n in tiers},
            # Q's hi and lo halves stacked as one product's 2m rows: the scores in one dispatch, not two
            "kernels_shl": {str(n): f"ag_shl{n}_{kind}" for n in tiers},
            "prep": "qkvnorm_rope", "scale": float(spec.attn_scale), "window": window,
            "rot": s.rotary_dim, "rope_pair": hd // 2, "inv_freq": spec.rope_inv_freq(local=kind == "local"),
            "qsplit": True, "cd_qnorm": L.CD_META, "cd_knorm": L.CD_META + hd * 2}, tiers


def bfp16_on() -> bool:
    """The block route's five projections on gemm_bfp16 (OPEN-PREFILL-BFP16) as `gemm_block`, the
    route `fast` (the default prefill mode) runs, with the q4 route kept as the `lean` variant
    for those who want its accuracy (`--prefill-mode lean`): on unless G4_BFP16=0, which emits
    the q4 route alone, byte for byte what it was."""
    return os.environ.get("G4_BFP16") != "0"


def _bfp16_route(q4: dict, plans: dict) -> dict:
    """The q4 route's steps on gemm_bfp16, per layer type: bfp16ebs8 activations against
    bfp16ebs8 weights, fp32 sums, one image for every shape (designs/gemm_bfp16, 10.6-12.1
    TFLOPS on Granite's shapes against the q4 GEMM's ~1.4 here). Each weight is its pool ops'
    tensors as `bfp16_gemm` pack ops, read from the GGUF's own blocks at load and stacked in the
    pool's row order (q, the k_proj copies, v), its rows padded to whole 1024-row groups with
    zero blocks (o and down: 3840 -> 4096). The copy is made only when the route is chosen and
    lives beside the decode pools, which stay as they are. Everything around the GEMMs -- the
    attention products, the host stages, the schedule -- is the q4 route's."""
    T = D.GEMM_T
    out: dict = {"layer_types": {}, "contexts": {}, "kernels": {}, "globals": {}, "builds": {}}
    for lt, gb in q4["layer_types"].items():
        pool = plans[lt]["pool"]
        b = copy.deepcopy(gb)
        n_pad = {}
        b["weights"] = {}
        for wname, w in gb["weights"].items():
            ops, row0 = [], 0
            for i in w["ops"]:
                o = pool[i]
                if o["in_dim"] % 256:
                    raise OpRangeError(f"gemma4 bfp16 route: {o['tensor']}'s K {o['in_dim']} is not a multiple of 256")
                ops.append({"op": "bfp16_gemm", "tensor": o["tensor"], "nch": o["nch"], "in_dim": o["in_dim"],
                            "dst": 0, "row0": row0})
                row0 += o["nch"] // (o["in_dim"] // 256) * 32
            n_pad[wname] = roundup(row0, D.BFP16_ROWS)
            for op in ops:
                op["n_total"] = n_pad[wname]
            b["weights"][wname] = {"from": "pack", "pack": ops}
        b["program"] = []
        for st in gb["program"]:
            wname = st["args"][0]
            K, N = int(st["args"][1].removeprefix("gemm_x_k")), n_pad[wname]
            kname = f"gemm_bfp16_n{N}_k{K}"
            x, y = f"gbfp_x_k{K}", f"gbfp_y_n{N}"
            b["program"].append({"op": "run", "kernel": kname, "args": [wname, x, y], "fmt": "bfp16", "n_pad": N})
            out["contexts"].setdefault("gemmbfp", f"{kname}/final.xclbin")
            out["kernels"][kname] = {"context": "gemmbfp", "insts": f"{kname}/insts.bin", "build": kname}
            out["globals"][x] = T * K // 8 * 9        # bfp16ebs8 A, nine bytes per eight values
            out["globals"][y] = N * T * 4             # fp32 C, c-tiled
            out["builds"][kname] = {"design": "gemm_bfp16/gemm_bfp16.py",
                                    "build_dir": f"gemm_bfp16/build_n{N}_k{K}_m{T}",
                                    "env": {"GB_M": str(T), "GB_K": str(K), "GB_N": str(N)}}
        out["layer_types"][lt] = b
    return out


def compact_on() -> bool:
    """The GGUF view on 5120-B q4_1-law pools (finalize_gguf_manifest): on unless
    DENSE_GGUF_COMPACT=0, unlike dense's opt-in Llama view. Every logit matched the f32-scale
    twin's, sequential and block (OPEN-FAMILY-GEMMA4)."""
    return os.environ.get("DENSE_GGUF_COMPACT") != "0"


def gemm_route(spec: ModelSpec, max_ctx: int = 4096, pool_quant: str | None = None) -> dict | None:
    """The block prefill route (OPEN-PREFILL-BATCH, kind "dense"), per layer type: dense's
    five-GEMM chain (qkv3, o, gate, up, down) with T single-token attention dispatches between
    the first two, which the engine's host chain runs with Gemma 3's GeGLU-tanh and sandwich
    norms. What dense.gemm_route cannot express is two geometries in one model, so each kind has
    its own qkv3 and o shapes, attention build (designs/dense/dx_attn.py at that kind's G4_KIND,
    so the V norm, unit scale, pair distance and local q are dx.py's), position table, act
    offsets and KV row. The global qkv3 spans q and the four k_proj copies the pack plan puts in
    the K and V slots, so the GEMM writes the same q | k | v rows dx.py's q / k / v GEMVs do.
    `layer_output_scale` needs nothing here: it is folded into the consts' post norms, which is
    where the host chain reads them.

    GGUF-direct only (the q4nx route has not run for this model). On by default, unlike
    dense's opt-in GGUF route: it gave the sequential route's greedy tokens on both prompts
    and the 1024-row window at 1023 / 1024 / 1025 (OPEN-FAMILY-GEMMA4). DENSE_GGUF_GEMM=0
    leaves it out: None means the sequential path, byte for byte what it was. `pool_quant`
    is the chunk law the GEMMs read: the spec's own (the f32-scale twin's or the Q4_0-native
    view's), or q4_1's for the compact view (finalize_gguf_manifest)."""
    if not gguf_direct(spec.quant) or not gemm_on():
        return None
    pool_quant = pool_quant or D.pool_quant(spec)
    if pool_quant not in SFX:
        raise OpRangeError(f"gemma4 gemm route: no GEMM build reads {pool_quant!r} pools")
    sfx = SFX[pool_quant]
    T = D.GEMM_T
    hid, ff = spec.hidden, spec.intermediate
    plans = pack_plan(spec)["layer_types"]
    shapes: set[tuple[int, int]] = set()

    def ctx(N: int, K: int) -> str:
        if N % 256 or K % 256:
            raise OpRangeError(f"gemma4 gemm route: [{N}, {K}] is not a multiple of 256 in both dims")
        shapes.add((N, K))
        return f"gemm_n{N}_k{K}{sfx}"

    def run(N: int, K: int, w: str) -> dict:
        return {"op": "run", "kernel": ctx(N, K), "args": [w, f"gemm_x_k{K}", f"gemm_y_n{N}"]}

    out: dict = {"layer_types": {}, "contexts": {}, "kernels": {}, "globals": {}, "builds": {}}
    act_max = 0
    for kind, lt in KINDS.items():
        L, G = layout(spec, max_ctx, kind), geometry(spec, kind)
        local = kind == "local"
        pool = plans[lt]["pool"]
        tensors = [op["tensor"] for op in pool]

        def ops(name: str) -> list[int]:
            idx = [i for i, t in enumerate(tensors) if t.endswith(name)]
            if not idx:
                raise OpRangeError(f"gemma4 gemm route: {lt} pack plan has no {name}")
            return idx

        # q, then every k_proj op (the global kind's K and V slots hold the head twice each), then
        # v_proj where the kind has one: one byte-contiguous run, as the engine needs
        qkv = ops("self_attn.q_proj.weight") + ops("self_attn.k_proj.weight") + (ops("self_attn.v_proj.weight") if local else [])
        if qkv != list(range(qkv[0], qkv[0] + len(qkv))):
            raise OpRangeError(f"gemma4 gemm route: {lt}'s q / k / v pack ops are not consecutive")
        weights = {"gqkv3_w": {"from": "pool", "ops": qkv},
                   "go_w": {"from": "pool", "ops": ops("self_attn.o_proj.weight")},
                   "ggate_w": {"from": "pool", "ops": ops("mlp.gate_proj.weight")},
                   "gup_w": {"from": "pool", "ops": ops("mlp.up_proj.weight")},
                   "gdown_w": {"from": "pool", "ops": ops("mlp.down_proj.weight")}}
        qw, kvw = G.QW, G.KVW
        program = [run(qw + 2 * kvw, hid, "gqkv3_w"), run(hid, qw, "go_w"),
                   run(ff, hid, "ggate_w"), run(ff, hid, "gup_w"), run(hid, ff, "gdown_w")]
        kn, tab, ctxn = ("dxB_local", "ptab_local", "dxa") if local else ("dxB", "ptab", "dxag")
        build = f"dx_attn_{kind}"
        args = ["pool", "xres", "consts", "state", "act", tab]
        check_buffer_args(kn, args)
        out["contexts"][ctxn] = f"{build}/final.xclbin"
        out["kernels"][kn] = {"context": ctxn, "insts": f"{build}/insts.bin", "patch": "attnpos", "build": build,
                              "window": spec.sliding_window if local else 0,
                              "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW}
        out["builds"][build] = {"design": "dense/dx_attn.py",
                                "build_dir": f"dense/build_dxa_gemma4_{kind}_h{hid}", "env": {KIND_ENV: kind}}
        out["layer_types"][lt] = {"kind": "dense", "t": T, "eps": spec.norm_eps,
                                  "program": program, "weights": weights,
                                  "qw": qw, "kvw": kvw, "ff": ff,
                                  "ad_q": L.AD_Q, "ad_kvn": L.AD_KVN, "ad_og": L.AD_OG,
                                  "attn_kernel": kn, "attn_args": args,
                                  "sandwich": True, "act": spec.activation, "norm_groups": spec.norm_groups,
                                  # the FFN-side norms from the consts, where post_feedforward carries the
                                  # folded scale; the file's copy (Gemma 3's source) does not
                                  "cd_preffn": L.CD_PREFFN, "cd_postffn": L.CD_POSTFFN}
        act_max = max(act_max, L.AD_BYTES)
        if attn_block_on():
            # the dxB kernel above stays: OFLM_OPEN_ATTN_BLOCK=0 at run time goes back to it
            ab, tiers = _attn_block(spec, kind, L, G, T, max_ctx)
            out["layer_types"][lt]["attn_block"] = ab
            for n in tiers:
                # S is [m, hd] x [hd, n]; O is [m, n] x [n, hd]. Both tile the head over 8
                # columns (hd 256 and 512 are multiples of 32 x 8), so every product of both kinds
                # is one image: attn_gemm.py takes M and K at run time and N only shapes the
                # instruction stream, and the 42 xclbins differ in their UUIDs and timestamps
                # alone. One context, then: a layer's S -> P.V step stops costing a context
                # switch (~3.6 ms each, 48 a block), leaving the two between it and the GEMMs.
                for tag, M, K, N in (("s", ab["m"], ab["hd"], n), ("pv", ab["m"], n, ab["hd"]), ("shl", 2 * ab["m"], ab["hd"], n)):
                    name = f"ag_{tag}{n}_{kind}"
                    out["contexts"].setdefault("ag", f"{name}/final.xclbin")
                    out["kernels"][name] = {"context": "ag", "insts": f"{name}/insts.bin", "build": name}
                    out["builds"][name] = {"design": "attn_block/attn_gemm.py",
                                           "build_dir": f"attn_block/build_{tag}{n}_m{M}_h{ab['hd']}",
                                           "env": {"AG_M": str(M), "AG_K": str(K), "AG_N": str(N), "AG_COLS": "8"}}
                    for g_, size in (("ag_a", M * K * 2), ("ag_b", K * N * 2), ("ag_c", M * N * 4)):
                        out["globals"][g_] = max(out["globals"].get(g_, 0), size)
    # the T-wide attention scratch the engine shuttles each token's act slice through, at the
    # larger kind's act size (core.cpp indexes it per layer with that layer's own act_bytes)
    out["globals"]["gact"] = T * act_max
    # one GEMM context for every shape: the core program depends on neither N nor K (dense)
    for N, K in sorted(shapes):
        name = ctx(N, K)
        out["contexts"].setdefault("gemm", f"{name}/final.xclbin")
        out["kernels"][name] = {"context": "gemm", "insts": f"{name}/insts.bin", "build": name}
        out["globals"][f"gemm_x_k{K}"] = K * T * 2      # bf16, pre-tiled [K, T]
        out["globals"][f"gemm_y_n{N}"] = N * T * 4      # f32 [N, T]
        out["builds"][name] = {"design": "gemm_q4_prefill/gemm_q4_prefill.py",
                               "build_dir": f"gemm_q4_prefill/build_n{N}_k{K}_t{T}{sfx}",
                               "env": {"GQP_N": str(N), "GQP_K": str(K), "GQP_T": str(T),
                                       **({"GQP_SCALES_F32": "1"} if pool_quant == "q4_1_f32" else
                                          {"GQP_FMT": "q4_0"} if pool_quant == "q4_0" else {})}}
    out["variants"] = {}
    if bfp16_on():
        # bfp16 is gemm_block (fast, the default); the q4 route above becomes `lean`
        bf = _bfp16_route(out, plans)
        for lt, gb in bf["layer_types"].items():
            out["variants"][lt] = {"lean": out["layer_types"][lt]}
            out["layer_types"][lt] = gb
        for k in ("contexts", "kernels", "globals", "builds"):
            out[k].update(bf[k])
    return out


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    """dx_local on the local design's context, dx on the global one's. Each attnpos kernel
    names its own KV and position-record row (the engine's per-kernel geometry); each layer
    type its own consts / act / state sizes and position table."""
    sets = _kernel_sets(spec)
    lm_set = "lm_head_q4" + SFX[D.pool_quant(spec)]
    Ll = layout(spec, max_ctx, "local")
    out = {
        "contexts": {"dx": f"{sets['local']}/final.xclbin", "dxg": f"{sets['global']}/final.xclbin",
                     "ln": "ln/final.xclbin", "lm": f"{lm_set}/final.xclbin"},
        "kernels": {"ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
                    "lm": {"context": "lm", "insts": f"{lm_set}/insts.bin", "build": lm_set}},
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": spec.hidden * 4, "zero": spec.hidden * 4, "normw": spec.hidden * 2,
                    "xresf": spec.hidden * 4, "hn": spec.hidden * 2, "logits": D.lm_rows(spec) * 4,
                    "lmpool": Ll.LMHEAD_POOL_BYTES},
    }
    for kind, lt in KINDS.items():
        L = layout(spec, max_ctx, kind)
        local = kind == "local"
        kn, tab, ctx = ("dx_local", "ptab_local", "dx") if local else ("dx", "ptab", "dxg")
        window = spec.sliding_window if local else 0
        args = ["pool", "xres", "consts", "state", "act", tab]
        check_buffer_args(kn, args)
        out["kernels"][kn] = {"context": ctx, "insts": f"{sets[kind]}/insts.bin", "patch": "attnpos",
                              "build": sets[kind], "window": window,
                              "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW}
        if dx_geometry(spec, kind).KVSLICE:            # per-core window fills of whole blocks (OPEN-ATTN-KVSLICE)
            out["kernels"][kn]["kvslice"] = dx_geometry(spec, kind).KVSLICE
        out["globals"][tab] = {"per_row": L.PTAB_ROW, "inv_freq": spec.rope_inv_freq(local=local), "window": window,
                               "rotary_dim": sub_spec(spec, kind).rotary_dim}
        out["layer_types"][lt] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": kn, "args": args}],
        }
    r = gemm_route(spec, max_ctx)
    if r:
        for k in ("contexts", "kernels", "globals"):
            out[k].update(r[k])
        for lt, gb in r["layer_types"].items():
            out["layer_types"][lt]["gemm_block"] = gb
            if r["variants"].get(lt):
                out["layer_types"][lt]["gemm_block_variants"] = r["variants"][lt]
    if spec.quant == "q4_1":
        g = programs(replace(spec, quant=gguf_quant(spec)), max_ctx)
        out["gguf"] = {k: g[k] for k in ("contexts", "kernels", "layer_types", "tail", "globals")}
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    n = D.cores_for(sub_spec(spec, "local"))
    rows = D.lm_rows(spec)
    b = {"ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{spec.hidden}_{spec.norm_eps:g}",
                "env": {"LN_N": str(spec.hidden), "LN_EPS": f"{spec.norm_eps:g}"}},
         "lm_head_q4": {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{rows}",
                        "env": {"LMHEAD_N": str(rows), "LMHEAD_K": str(spec.hidden), "LMHEAD_CORES": str(n)}},
         "lm_head_q4_f32": {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{rows}_f32",
                            "env": {"LMHEAD_N": str(rows), "LMHEAD_K": str(spec.hidden),
                                    "LMHEAD_CORES": str(n), "LMHEAD_SCALES_F32": "1"}}}
    if q40_on():
        b["lm_head_q4_q40"] = {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{rows}_q40",
                               "env": {"LMHEAD_N": str(rows), "LMHEAD_K": str(spec.hidden),
                                       "LMHEAD_CORES": str(n), "LMHEAD_Q4_0": "1"}}
    for kind, name in (("local", "dx"), ("global", "dxg")):
        bd = f"dense/build_gemma4_{kind}_h{spec.hidden}"
        b[name] = {"design": "dense/dx.py", "build_dir": bd, "env": {KIND_ENV: kind, DX_ENV: "1"}}
        b[name + "_f32"] = {"design": "dense/dx.py", "build_dir": bd + "_f32",
                            "env": {KIND_ENV: kind, DX_ENV: "1", "GEMV_SCALES_F32": "1"}}
        if q40_on():
            b[name + "_q40"] = {"design": "dense/dx.py", "build_dir": bd + "_q40",
                                "env": {KIND_ENV: kind, DX_ENV: "1", "GEMV_Q4_0": "1"}}
    # the GEMMs the GGUF view runs: Q4_0-native pools, else the compact view's q4_1-law ones or
    # the twin's f32 ones
    if q40_on():
        r = gemm_route(replace(spec, quant="q4_0"))
    else:
        r = gemm_route(replace(spec, quant="q4_1_f32"), pool_quant="q4_1" if compact_on() else "q4_1_f32")
    if r:
        b.update(r["builds"])
    return b


def _pairs(gguf_ops: list[dict], base_ops: list[dict], what: str) -> None:
    """The two sections' projection ops are the same tensors at the same chunk counts; only the
    byte offsets move with the chunk law. Anything else is not a view of the same pool."""
    if len(gguf_ops) != len(base_ops) or any(
            a["tensor"] != b["tensor"] or a["op"] != "std_perm_gguf" or b["op"] != "std_perm"
            or a["nch"] != b["nch"] or a["in_dim"] != b["in_dim"] for a, b in zip(gguf_ops, base_ops)):
        raise OpRangeError(f"gemma4 compact GGUF view: {what}'s pack ops do not pair with the q4_1 section's")


def finalize_gguf_manifest(spec: ModelSpec, manifest: dict) -> None:
    """The GGUF view on the q4_1 law's 5120-B chunks and kernels (DENSE_GGUF_COMPACT=0: the twin's).

    The f32-scale kernels narrow each f32 scale to bf16 with RNE on the core (gemv_q4.h
    load_scale, gemm_q4_dequant.h gqd_scale), so storing that bf16 in the pool instead
    (pools.cpp std_perm_gguf under a 5120-B chunk) feeds the q4_1 builds the same products:
    a representation change, 1/6 fewer weight bytes per token, with the same codes. The
    GGUF section keeps its reader (std_perm_gguf, the scaled embedding, the f32 norms and
    their folded layer scale) and takes the q4_1 section's pool geometry, buffers, decode
    kernels and head. The block prefill route's GEMMs become the q4_1-law builds."""
    if spec.quant != "q4_1" or "gguf" not in manifest or not compact_on() or q40_on():
        return
    g, b = manifest["gguf"], manifest
    g["spec"] = copy.deepcopy(b["spec"])      # the pool law the view's kernels compile for
    g["spec_hash"] = b["spec_hash"]
    for key in ("chunk_bytes", "pool_bytes", "lmhead_pool_bytes"):
        g["layout"][key] = b["layout"][key]
    for key in ("chunk_bytes", "pool_bytes"):
        g["pack"][key] = b["pack"][key]
    _pairs(g["pack"]["lm_head"]["ops"], b["pack"]["lm_head"]["ops"], "lm_head")
    g["pack"]["lm_head"] = copy.deepcopy(b["pack"]["lm_head"])
    for op in g["pack"]["lm_head"]["ops"]:
        op["op"] = "std_perm_gguf"
    g["globals"]["lmpool"] = b["globals"]["lmpool"]
    route = gemm_route(replace(spec, quant="q4_1_f32"), b["max_ctx_default"], pool_quant="q4_1")
    if route:
        # drop the twin's f32-scale GEMMs; the compact ones take the same names minus _f32
        for name in [k for k in g["kernels"] if k.startswith("gemm_n") and k.endswith("_f32")]:
            del g["kernels"][name]
            g["builds"].pop(name, None)     # the nested builds() already names the compact ones
        for k in ("contexts", "kernels", "globals", "builds"):
            g[k].update(route[k])
    for name, layer in g["layer_types"].items():
        base = b["layer_types"][name]
        if layer["pack"]["consts"] != base["pack"]["consts"] or layer["buffers"] != base["buffers"]:
            raise OpRangeError(f"gemma4 compact GGUF view: {name}'s consts or buffers depend on the chunk law")
        _pairs(layer["pack"]["pool"], base["pack"]["pool"], name)
        layer["pack"] = copy.deepcopy(base["pack"])
        for op in layer["pack"]["pool"]:
            op["op"] = "std_perm_gguf"
        layer["program"] = copy.deepcopy(base["program"])
        if route:
            layer["gemm_block"] = route["layer_types"][name]
            layer.pop("gemm_block_variants", None)
            if route["variants"].get(name):
                layer["gemm_block_variants"] = route["variants"][name]
    # the decode kernels and the head: the q4_1 section's builds, in its contexts
    for kernel in {st["kernel"] for lt in g["layer_types"].values() for st in lt["program"]} | {"lm"}:
        g["kernels"][kernel] = copy.deepcopy(b["kernels"][kernel])
        ctx = g["kernels"][kernel]["context"]
        g["contexts"][ctx] = b["contexts"][ctx]
        build = g["kernels"][kernel]["build"]
        g["builds"][build] = copy.deepcopy(b["builds"][build])


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    """The local layers' rows are the model-wide defaults; the global kernels override theirs."""
    out = D.manifest_layout(sub_spec(spec, "local"), max_ctx)
    # The layout's frequencies must be the table its rotary_dim describes (manifest.cpp refuses
    # a mismatch); rope_inv_freq() unqualified is the GLOBAL table, 64 pairs, not the local 128.
    out["rope_inv_freq"] = spec.rope_inv_freq(local=True)
    out["rope_theta"] = spec.rope_local_theta
    out["pool_bytes"] = pool_bytes(spec)
    if spec.final_softcap:
        out["final_logit_softcapping"] = spec.final_softcap
    return out


def hf_config_check(spec: ModelSpec) -> dict:
    """The text tower's fields the kernels bake in (config.json nests them under text_config)."""
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "intermediate_size": spec.intermediate, "head_dim": spec.head_dim,
            "global_head_dim": spec.global_head_dim or spec.head_dim,
            "num_global_key_value_heads": spec.global_kv_heads, "attention_k_eq_v": spec.attn_k_eq_v,
            "sliding_window": spec.sliding_window, "final_logit_softcapping": spec.final_softcap or None,
            # Both position tables and the global rotation width are baked in at export time, so a
            # same-shaped config with other RoPE settings must refuse. Emitted whole, from the spec's
            # values; check_model compares the sub-object as it compares K2's.
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": spec.global_rotary_dim / (spec.global_head_dim or spec.head_dim),
                                   "rope_theta": spec.rope_theta, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": spec.rope_local_theta, "rope_type": "default"}}}


GEN_KERNELS = D.GEN_KERNELS
KERNEL_SOURCES = list(D.KERNEL_SOURCES)
