"""GGUF -> q4nx converter for Qwen3.6-MoE (qwen3_5_moe arch) targeting the OPEN
NPU engine (npu-engine/) which reads FLM's q4_1 5120B / q8 8704B FILE format.

Pipeline per tensor:  GGUF k-quant block  --ggml.dequantize-->  f32
                      --arch transform (transpose / q_proj deinterleave / -exp)-->
                      --re-quant (q4nx_format.pack_q4_1 / pack_q8_0 / bf16)-->
                      safetensors tensor (I8 / BF16 / F32)

The tensor-name map, shapes, dtypes and the single non-identity weight reorder
(full-attn q_proj deinterleave) were all verified byte-for-byte against Cyrus's
installed model_3LiF.q4nx (see README.md "Verified mapping").

Usage:
    python convert.py -i model.gguf -o out_dir            # writes out_dir/model.q4nx (+ config.json)
    python convert.py -i model.gguf -o out_dir --layers 3 # only first 3 layers (debug slice)
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4nx_format import pack_q4_1, pack_q8_0, f32_to_bf16_u16

from gguf import GGUFReader, dequantize
from gguf.constants import GGMLQuantizationType

HEAD_DIM = 256          # value_length (per-head dim of q/gate on full-attn)
NUM_HEADS = 16          # num_attention_heads


# --------------------------------------------------------------------------- #
#  GGUF reading helpers
# --------------------------------------------------------------------------- #
def read_meta(reader):
    meta = {}
    for f in reader.fields.values():
        try:
            meta[f.name] = f.contents()
        except Exception:
            pass
    return meta


def dequant_f32(t):
    """GGUFReaderTensor -> f32 ndarray in logical [out, in] (or original) order."""
    w = dequantize(t.data, t.tensor_type)
    return np.ascontiguousarray(w, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  per-tensor arch transforms (HF/GGUF order -> FLM FILE order)
# --------------------------------------------------------------------------- #
def deinterleave_qgate(W):
    """full-attn q_proj [2*H*hd, in] interleaved per head [q_h|gate_h] ->
    planar [q(all heads) | gate(all heads)].  (g p h) c -> (p g h) c."""
    two_hd = 2 * HEAD_DIM
    rows = W.shape[0]
    g = rows // two_hd                       # heads
    return W.reshape(g, 2, HEAD_DIM, W.shape[1]).transpose(1, 0, 2, 3).reshape(rows, W.shape[1])


# Linear-attention (gated DeltaNet) head geometry: 16 k-heads x 128, 32 v-heads
# x 128 (2 v-heads per k-head). llama.cpp's HF->GGUF stores every v-head-indexed
# axis GROUP-major, (g q) with g=2, q=16 — HF/FLM order is HEAD-major (q g).
# Established 2026-09-01 by cosine-matching Cyrus's converted 27B against the
# base 35B q4nx (LoRA-healed weights still ~0.99 vs their source layer): the
# identity mapping left qkv's v-half, z-gate, ssm_out cols, alpha/beta, dt.bias
# and conv1d's v-cols at cos 0.06-0.6; regrouping (g q)->(q g) brings all of
# them to 0.99-1.00.  (These are the same tensors the reference converter
# calls "reorder_linear"; its rearrange goes the other way because it targets
# the newer FLM 1.0.3 layout.)
LIN_K_HEADS = 16
LIN_V_HEADS = 32
LIN_HEAD_DIM = 128
LIN_G = LIN_V_HEADS // LIN_K_HEADS   # 2


def regroup_vheads_axis(W, axis):
    """(g q) -> (q g) on an axis of length LIN_V_HEADS*LIN_HEAD_DIM (per-head
    128-blocks) or LIN_V_HEADS (per-head scalars)."""
    W = np.asarray(W)
    n = W.shape[axis]
    W = np.moveaxis(W, axis, 0)
    rest = W.shape[1:]
    if n == LIN_V_HEADS * LIN_HEAD_DIM:
        W = W.reshape(LIN_G, LIN_K_HEADS, LIN_HEAD_DIM, *rest).transpose(1, 0, 2, *range(3, 3 + len(rest)))
    elif n == LIN_V_HEADS:
        W = W.reshape(LIN_G, LIN_K_HEADS, *rest).transpose(1, 0, *range(2, 2 + len(rest)))
    else:
        raise ValueError(f"regroup_vheads_axis: axis len {n} is not a v-head axis")
    W = W.reshape(n, *rest)
    return np.ascontiguousarray(np.moveaxis(W, 0, axis))


def regroup_qkv_rows(W):
    """attn_qkv [q(2048) | k(2048) | v(4096), in]: only the v rows are v-head indexed."""
    qk = LIN_K_HEADS * LIN_HEAD_DIM * 2
    out = np.array(W, np.float32, copy=True)
    out[qk:] = regroup_vheads_axis(W[qk:], 0)
    return out


def regroup_conv_vcols(W_kd):
    """conv1d after the [k, dim] transpose: dim = [q 2048 | k 2048 | v 4096]."""
    qk = LIN_K_HEADS * LIN_HEAD_DIM * 2
    out = np.array(W_kd, np.float32, copy=True)
    out[:, qk:] = regroup_vheads_axis(W_kd[:, qk:], 1)
    return out


# --------------------------------------------------------------------------- #
#  name mapping (GGUF llama.cpp name -> q4nx name) + how to emit
# --------------------------------------------------------------------------- #
# emit kinds: 'q4' (pack_q4_1), 'q8' (pack_q8_0), 'bf16', 'f32'
GLOBAL_MAP = {
    "token_embd.weight":  ("model.embed_tokens.weight", "bf16", None),
    "output_norm.weight": ("model.norm.weight",         "bf16", None),
    "output.weight":      ("lm_head.weight",            "q8",   None),
}

# per-layer: gguf suffix -> (q4nx suffix, emit, transform)
LAYER_MAP = {
    # linear-attn block
    "attn_qkv.weight":  ("linear_attn.qkv_proj.weight",         "q4",   "qkv_vheads"),
    "attn_gate.weight": ("self_attn.gate_proj.weight",          "q4",   "rows_vheads"),   # z-gate on linear layers
    "ssm_out.weight":   ("linear_attn.ssm_out_proj.weight",     "q4",   "cols_vheads"),
    # GGUF already carries -exp(A_log) (all-negative); do NOT exponentiate again.
    "ssm_a":            ("linear_attn.ssm_a",                   "f32",  "vec_vheads"),
    "ssm_dt.bias":      ("linear_attn.ssm_dt.bias",            "f32",  "vec_vheads"),
    "ssm_alpha.weight": ("linear_attn.ssm_alpha_proj.weight",   "bf16", "T_cols_vheads"),
    "ssm_beta.weight":  ("linear_attn.ssm_beta_proj.weight",    "bf16", "T_cols_vheads"),
    "ssm_conv1d.weight":("linear_attn.ssm_conv1d.weight",       "bf16", "conv_vheads"),
    "ssm_norm.weight":  ("linear_attn.ssm_norm.weight",         "bf16", None),
    # full-attn block
    "attn_q.weight":    ("self_attn.q_proj.weight",             "q4",   "deint_q"),
    "attn_k.weight":    ("self_attn.k_proj.weight",             "q4",   None),
    "attn_v.weight":    ("self_attn.v_proj.weight",             "q4",   None),
    "attn_output.weight":("self_attn.o_proj.weight",            "q4",   None),
    "attn_q_norm.weight":("self_attn.q_norm.weight",            "bf16", None),
    "attn_k_norm.weight":("self_attn.k_norm.weight",            "bf16", None),
    # norms / moe control
    "attn_norm.weight":        ("input_layernorm.weight",             "bf16", None),
    "post_attention_norm.weight":("post_attention_layernorm.weight",  "bf16", None),
    "ffn_gate_inp.weight":     ("moe_router.weight",                  "bf16", "T"),
    "ffn_gate_inp_shexp.weight":("shared_expert_gate.weight",         "bf16", "squeeze"),
    # experts (3D [n_expert, out, in])
    "ffn_up_exps.weight":      ("mlp.up_exps_proj.weight",            "q4_experts", None),
    "ffn_gate_exps.weight":    ("mlp.gate_exps_proj.weight",          "q4_experts", None),
    "ffn_down_exps.weight":    ("mlp.down_exps_proj.weight",          "q4_experts", None),
    "ffn_up_shexp.weight":     ("mlp.share_up_exps_proj.weight",      "q4",   None),
    "ffn_gate_shexp.weight":   ("mlp.share_gate_exps_proj.weight",    "q4",   None),
    "ffn_down_shexp.weight":   ("mlp.share_down_exps_proj.weight",    "q4",   None),
}


def apply_transform(W, kind):
    if kind is None:
        return W
    if kind == "T":
        return np.ascontiguousarray(W.T)
    if kind == "neg_exp":
        return (-np.exp(W.astype(np.float32))).astype(np.float32)
    if kind == "conv":
        # GGUF conv1d is [dim, 1, k] or [dim, k]; FILE wants [k, dim]
        w = np.asarray(W, np.float32)
        w = w.reshape(w.shape[0], -1)          # [dim, k]
        return np.ascontiguousarray(w.T)       # [k, dim]
    if kind == "squeeze":
        return np.asarray(W, np.float32).reshape(-1)
    if kind == "deint_q":
        return deinterleave_qgate(W)
    if kind == "qkv_vheads":
        return regroup_qkv_rows(np.asarray(W, np.float32))
    if kind == "rows_vheads":
        return regroup_vheads_axis(np.asarray(W, np.float32), 0)
    if kind == "cols_vheads":
        return regroup_vheads_axis(np.asarray(W, np.float32), 1)
    if kind == "vec_vheads":
        return regroup_vheads_axis(np.asarray(W, np.float32).reshape(-1), 0)
    if kind == "T_cols_vheads":
        return regroup_vheads_axis(np.ascontiguousarray(np.asarray(W, np.float32).T), 1)
    if kind == "conv_vheads":
        return regroup_conv_vcols(apply_transform(W, "conv"))
    raise ValueError(kind)


def emit(name, W, kind, torch_mod):
    """Return a torch tensor with the right dtype for safetensors."""
    import torch
    if kind == "bf16":
        u16 = f32_to_bf16_u16(np.asarray(W, np.float32))
        return torch.from_numpy(u16.copy()).view(torch.bfloat16)
    if kind == "f32":
        return torch.from_numpy(np.ascontiguousarray(W, np.float32))
    if kind == "q4":
        packed = pack_q4_1(W)
        return torch.from_numpy(packed.astype(np.int8))
    if kind == "q8":
        packed = pack_q8_0(W)
        return torch.from_numpy(packed.astype(np.int8))
    raise ValueError(kind)


def pack_experts(W3d, torch_mod):
    """[n_expert, out, in] f32 -> I8 [n_expert*(out//32), in//256, 5120] (contiguous per expert)."""
    import torch
    n_e = W3d.shape[0]
    pieces = [pack_q4_1(np.ascontiguousarray(W3d[e])) for e in range(n_e)]
    stacked = np.concatenate(pieces, axis=0)        # [n_e*out//32, in//256, 5120]
    return torch.from_numpy(stacked.astype(np.int8))


# --------------------------------------------------------------------------- #
def convert(gguf_path, out_dir, max_layers=None, warn_norms=True):
    import torch
    from safetensors.torch import save_file

    reader = GGUFReader(gguf_path)
    meta = read_meta(reader)
    gt = {t.name: t for t in reader.tensors}
    print(f"[INFO] GGUF: {len(gt)} tensors, arch={meta.get('general.architecture')}")

    tensors = {}
    norm_warnings = []

    def process(gguf_name, q4nx_name, kind, transform, layer_dbg=False):
        t = gt[gguf_name]
        W = dequant_f32(t)
        if kind == "q4_experts":
            W = apply_transform(W, transform)  # usually None
            tensors[q4nx_name] = pack_experts(W, torch)
        else:
            W = apply_transform(W, transform)
            if kind == "bf16" and ("layernorm" in q4nx_name or "q_norm" in q4nx_name
                                   or "k_norm" in q4nx_name or q4nx_name == "model.norm.weight"):
                mu = float(np.asarray(W, np.float32).mean())
                if warn_norms and abs(mu) < 0.2:
                    norm_warnings.append((q4nx_name, mu))
            tensors[q4nx_name] = emit(q4nx_name, W, kind, torch)
        if layer_dbg:
            print(f"    {gguf_name:32s} -> {q4nx_name:44s} {kind}")

    # globals
    for gn, (qn, kind, tr) in GLOBAL_MAP.items():
        if gn in gt:
            process(gn, qn, kind, tr)
            print(f"[INFO] {gn} -> {qn} ({kind})")

    # detect layer count
    layer_ids = set()
    for name in gt:
        if name.startswith("blk."):
            try:
                layer_ids.add(int(name.split(".")[1]))
            except ValueError:
                pass
    n_layers = max(layer_ids) + 1 if layer_ids else 0
    # MTP ("nextn") layers: llama.cpp appends them as extra blk.N entries that
    # carry a full attention+MoE sub-layer set plus blk.N.nextn.* tensors, and
    # counts them in block_count. FLM drops MTP entirely, so exclude them here —
    # otherwise the last (MTP) block would be emitted as a real decoder layer.
    nextn = int(meta.get("qwen35moe.nextn_predict_layers", 0) or 0)
    nextn_blocks = sorted({int(k.split(".")[1]) for k in gt if k.startswith("blk.") and ".nextn." in k})
    if nextn_blocks:
        n_layers = min(n_layers, min(nextn_blocks))
    elif nextn:
        n_layers -= nextn
    if nextn or nextn_blocks:
        print(f"[INFO] MTP/nextn layers excluded: {nextn_blocks or nextn}; decoder layers = {n_layers}")
    if max_layers:
        n_layers = min(n_layers, max_layers)
    print(f"[INFO] Converting {n_layers} layers")

    for L in range(n_layers):
        present = [k for k in gt if k.startswith(f"blk.{L}.")]
        is_full = any(k == f"blk.{L}.attn_q.weight" for k in present)
        for gn in present:
            suffix = gn[len(f"blk.{L}."):]
            if suffix not in LAYER_MAP:
                print(f"[WARN] no map for {gn}, skipping")
                continue
            qsuf, kind, tr = LAYER_MAP[suffix]
            process(gn, f"model.layer.{L}.{qsuf}", kind, tr, layer_dbg=(L == 0))
        print(f"[INFO] layer {L} ({'full-attn' if is_full else 'linear'}) done")

    if norm_warnings:
        print("\n[WARN] These norms look zero-centered (mean~0); FLM expects EFFECTIVE (1+w).")
        print("       If your GGUF did NOT bake the +1, they are WRONG. Re-run with --add-norm-one.")
        for n, mu in norm_warnings[:8]:
            print(f"         {n}  mean={mu:+.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.q4nx")
    print(f"[INFO] Saving {len(tensors)} tensors -> {out_path}")
    save_file(tensors, out_path)
    _write_config(out_dir, meta, n_layers)
    print("[INFO] Done.")
    return out_path


def _write_config(out_dir, meta, n_layers):
    cfg = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "num_hidden_layers": n_layers,
        "full_attention_interval": int(meta.get("qwen35moe.full_attention_interval", 3)),
        "hidden_size": int(meta.get("qwen35moe.embedding_length", 2048)),
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_HEADS,
        "num_experts": int(meta.get("qwen35moe.expert_count", 256)),
        "num_experts_per_tok": int(meta.get("qwen35moe.expert_used_count", 8)),
        "vocab_size": int(meta.get("qwen35moe.vocab_size", 248320)),
        "rms_norm_eps": 1e-6,
        "note": "Converted by tools/q4nx-convert (q4_1 5120B / q8 8704B FILE format).",
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--layers", type=int, default=None, help="convert only first N layers")
    args = ap.parse_args()
    convert(args.input, args.output, max_layers=args.layers)
