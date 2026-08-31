"""FLM 1.0.3 (Q4_K) q4nx FILE reader.

The open engine and tooling were reverse-engineered from FLM 1.0.2, whose q4nx
files use q4_1 chunks (5120 B) with a 16-lane nibble interleave.  FLM 1.0.3's
converter (FastFlowLM/FLM_Q4NX_Converter, `default_tensor_type: Q4_K`) emits a
DIFFERENT file:

  1. Q4_K-refit quant, 4736 B/chunk (vs q4_1 5120 B).
  2. plain column-major intra-chunk order (vs the 1.0.2 16-lane interleave).
  3. head-pairing reorders `(q g p)->(g q p)` baked into the LINEAR-attention
     tensors that 1.0.2 does NOT have -- this reader UNDOES them so the logical
     [out,in] matrix matches what the 1.0.2 pipeline (build_pools / full_forward)
     already consumes.

Byte layout read straight from the reference converter's `_pack_q4k`
(q4nx/model_converter.py) and `unpack_q4_k`/`_refit_one_side`
(q4nx/gguf_tensor.py); see .claude/plans/qwen36-1.0.3-format-support.md.

The container (safetensors header + raster [out//32, in//256, CHUNK] I8 tensors)
is unchanged from 1.0.2, so this module only supplies the per-chunk dequant and
the reorder-undo.  Tile assembly is PLAIN RASTER, identical to
full_forward.dequant_std: tile f -> rows 32*(f//ncol), cols 256*(f%ncol).

Two facts are INFERRED (no 1.0.3 sample existed at write time) and must be
confirmed on the first real file:
  - ssm.state_size = 128  (used by the reorder dims below)
  - whether stored `ssm_a` is -exp(A_log) (as 1.0.2) or raw A_log.
"""
import numpy as np

Q4K_CHUNK = 4736
Q8_CHUNK = 8704            # lm_head: same size as 1.0.2, different (column-major) order
STATE_SIZE = 128          # ssm.state_size (INFERRED); reorder p-dim
N_HEADS = 16              # g: 16 KV heads across the paired halves


def bf16_to_f32(u16):
    return (np.asarray(u16, np.uint32) << 16).view(np.float32)


# ---- Q4_K-refit chunk dequant -------------------------------------------------
# One 4736 B chunk = one 32-row x 256-col tile, concatenation order [s8|m8|q|S|M]:
#   [0:256]     s8  uint8  per-group(32-col) scale, index g*32+r  (g 0..7, r 0..31)
#   [256:512]   m8  uint8  per-group min,           index g*32+r
#   [512:4608]  q   4096 B nibbles, byte = C*16 + R//2 (C col 0..255, R row 0..31);
#                   low nibble = even row, high nibble = odd row
#   [4608:4672] S   32 bf16 super-block scale, index r
#   [4672:4736] M   32 bf16 super-block min, index r  (STORED NEGATED)
# value(r,c) = S[r]*s8[g,r]*q(r,c) + M[r]*m8[g,r],  g=c//32   (M already negative)

def _q4k_index_tables():
    r = np.arange(32)
    c = np.arange(256)
    g = (c // 32)
    # scale/min slot per (r,c): g*32 + r
    sm_slot = (g[None, :] * 32 + r[:, None]).astype(np.int64)        # [32,256]
    # nibble byte slot per (r,c): c*16 + r//2
    byte_slot = (c[None, :] * 16 + (r[:, None] // 2)).astype(np.int64)  # [32,256]
    hi = (r[:, None] % 2 == 1)                                       # odd row -> high nibble
    return sm_slot, byte_slot, hi


_SM, _BYTE, _HI = _q4k_index_tables()


def dequant_q4k_chunks(chunks):
    """[nch,4736] uint8 -> [nch,32,256] f32 (one 32x256 tile per chunk)."""
    chunks = np.ascontiguousarray(chunks).reshape(-1, Q4K_CHUNK)
    nch = chunks.shape[0]
    s8 = chunks[:, 0:256].astype(np.float32)                        # [nch,256]
    m8 = chunks[:, 256:512].astype(np.float32)
    qb = chunks[:, 512:4608]                                        # [nch,4096]
    S = bf16_to_f32(np.ascontiguousarray(chunks[:, 4608:4672]).view(np.uint16))  # [nch,32]
    M = bf16_to_f32(np.ascontiguousarray(chunks[:, 4672:4736]).view(np.uint16))  # [nch,32]

    smf = _SM.reshape(-1)                                           # [32*256]
    s8v = s8[:, smf].reshape(nch, 32, 256)
    m8v = m8[:, smf].reshape(nch, 32, 256)
    byte = qb[:, _BYTE.reshape(-1)].reshape(nch, 32, 256)
    lo = (byte & 0x0F).astype(np.float32)
    hin = (byte >> 4).astype(np.float32)
    q = np.where(_HI[None, :, :], hin, lo)                         # [nch,32,256]

    Sr = S[:, :, None]                                             # [nch,32,1] broadcast over cols
    Mr = M[:, :, None]
    return Sr * s8v * q + Mr * m8v


def _raster_assemble(w, out_dim, in_dim):
    """[nch,32,256] tiles -> [out,in], plain raster (rows 32*(f//ncol), cols 256*(f%ncol))."""
    ncol = in_dim // 256
    W = np.empty((out_dim, in_dim), np.float32)
    for f in range(w.shape[0]):
        W[32 * (f // ncol):32 * (f // ncol) + 32,
          256 * (f % ncol):256 * (f % ncol) + 256] = w[f]
    return W


def dequant_q4k_file(packed_u8, out_dim, in_dim):
    """Verbatim q4nx 1.0.3 Q4_K bytes -> f32 [out_dim, in_dim] (pre-reorder-undo)."""
    return _raster_assemble(dequant_q4k_chunks(packed_u8), out_dim, in_dim)


# ---- Q8_0 lm_head (1.0.3 column-major) ---------------------------------------
# chunk 8704 B = [scales 512 B (256 bf16, index g*32+r) | data 8192 int8, index c*32+r]
# value(r,c) = scale[g,r]*q,  g=c//32
def dequant_q8_q4k_file(packed_u8, out_dim, in_dim):
    chunks = np.ascontiguousarray(packed_u8).reshape(-1, Q8_CHUNK)
    nch = chunks.shape[0]
    d = bf16_to_f32(np.ascontiguousarray(chunks[:, :512]).view(np.uint16))       # [nch,256] index g*32+r
    data = np.ascontiguousarray(chunks[:, 512:]).view(np.int8).astype(np.float32)  # [nch,8192] index c*32+r
    r = np.arange(32); c = np.arange(256); g = c // 32
    sm_slot = (g[None, :] * 32 + r[:, None]).reshape(-1)          # [32*256]
    data_slot = (c[None, :] * 32 + r[:, None]).reshape(-1)        # [32*256]
    dv = d[:, sm_slot].reshape(nch, 32, 256)
    qv = data[:, data_slot].reshape(nch, 32, 256)
    return _raster_assemble(dv * qv, out_dim, in_dim)


# ---- reorder undo (§4 of the plan) -------------------------------------------
# 1.0.3 bakes the head-pairing perm `(q g p)->(g q p)` (q=2 halves, g=16 heads,
# p=block) into linear-attention tensors; inverse on read is `(g q p)->(q g p)`.
# All operate on a numpy [out,in] (already dequantized, pre-assembled) matrix.

def _undo_qgp_rows(W, p):
    """rows laid as (g q p) -> (q g p). g=16, q=2."""
    out = W.shape[0]
    assert out == N_HEADS * 2 * p, (out, N_HEADS, p)
    return W.reshape(N_HEADS, 2, p, -1).transpose(1, 0, 2, 3).reshape(out, -1)


def _undo_qgp_cols(W, p):
    """cols laid as (g q p) -> (q g p)."""
    in_ = W.shape[1]
    assert in_ == N_HEADS * 2 * p, (in_, N_HEADS, p)
    return W.reshape(-1, N_HEADS, 2, p).transpose(0, 2, 1, 3).reshape(-1, in_)


def _undo_qg_rows(W):
    """rows laid as (g q) -> (q g). g=16, q=2 (for ssm_a / ssm_dt / alpha-beta out-dim)."""
    out = W.shape[0]
    assert out == N_HEADS * 2, out
    return W.reshape(N_HEADS, 2, *W.shape[1:]).swapaxes(0, 1).reshape(out, *W.shape[1:])


def _undo_qg_vec(v):
    """1-D (g q) -> (q g). g=16,q=2 -> length 32."""
    assert v.shape[0] == N_HEADS * 2, v.shape
    return v.reshape(N_HEADS, 2).T.reshape(-1)


def _undo_qg_cols(W):
    """cols laid as (g q) -> (q g). g=16,q=2 (alpha/beta out-dim=32, stored transposed)."""
    in_ = W.shape[1]
    assert in_ == N_HEADS * 2, in_
    return W.reshape(-1, N_HEADS, 2).swapaxes(1, 2).reshape(-1, in_)


# Which tensors 1.0.3 reorders vs 1.0.2, and how to invert on read.  Keyed by
# q4nx tensor-name substring; only applied to LINEAR-attention layers (the full-
# attn q_proj reorder is identical in both formats, so it passes through).  p is
# the ssm state size (128, inferred).
def apply_undo(name, arr, layer_type):
    """arr: dequantized fp weight/vector in the 1.0.3 STORED order.
    Returns it permuted into the 1.0.2 logical order the pipeline expects."""
    p = STATE_SIZE
    if layer_type == "linear":
        if "linear_attn.qkv_proj" in name:            # [8192,2048]: v-half (rows 4096:) reordered
            out = arr.copy()
            out[4096:8192] = _undo_qgp_rows(arr[4096:8192], p)
            return out
        if "self_attn.gate_proj" in name:             # z-gate [4096,2048]: whole
            return _undo_qgp_rows(arr, p)
        if "linear_attn.ssm_out_proj" in name:        # [2048,4096]: columns
            return _undo_qgp_cols(arr, p)
        if "ssm_alpha_proj" in name or "ssm_beta_proj" in name:  # bf16 [2048,32] (transposed): out-dim cols
            return _undo_qg_cols(arr)
        if "ssm_conv1d" in name:                      # bf16 [4,8192]: v-half cols (4096:) reordered
            out = arr.copy()
            out[:, 4096:8192] = _undo_qgp_cols(arr[:, 4096:8192], p)
            return out
        if name.endswith("linear_attn.ssm_a"):        # f32 [32]
            return _undo_qg_vec(arr)
        if "linear_attn.ssm_dt.bias" in name:         # f32 [32]
            return _undo_qg_vec(arr)
    return arr


# Tensors stored quantized (Q4_K 4736) in 1.0.3.  lm_head is Q8_0 (8704).
def is_q4k_quant_name(name):
    q = ("qkv_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj",
         "ssm_out_proj", "exps_proj")
    return name != "lm_head.weight" and any(k in name for k in q)
