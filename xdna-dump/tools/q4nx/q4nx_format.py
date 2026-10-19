"""q4nx FILE byte-layout writer -- the exact inverse of the verified reader in
tools/kernel-interp/q4nx.py (dequant_q4nx_packed) and full_forward.py.

FILE layout (verified 2026-08-30 vs HF reference weights, see npu-open-engine.md
"FILE vs POOL layouts"):

  A quantized tensor [out, in] is stored as an I8 safetensors tensor of shape
  [out//32, in//256, CHUNK] -- one CHUNK per 32-row x 256-col tile, tiles in
  PLAIN RASTER order (tile f -> rows 32*(f//ncol), cols 256*(f%ncol),
  ncol = in//256).  The loader applies all the exotic pool tilings at model-load
  time; the FILE itself is raster.

  q4_1 CHUNK = 5120 B:
    256 bf16 scales d   (planar, order j = bc*32 + r)
    256 bf16 mins  m    (planar, same order)
    4096 B nibbles      (16-lane interleave)
      nibble[(r//16)*4096 + bc*512 + i*16 + (r%16)] = quant(row r, col bc*32 + i)
      byte b holds nibble 2b (low) and 2b+1 (high)
    per-32 block along in-dim: value = q * d[j] + m[j], j = bc*32 + r, block=32.

  q8_0 CHUNK = 8704 B (lm_head):
    256 bf16 scales d   (order j = bc*32 + r)  -> 512 B
    8192 int8 values    (SAME 16-lane interleave, one byte per value)
      int8[(r//16)*4096 + bc*512 + i*16 + (r%16)] = quant(row r, col bc*32 + i)
    value = q * d[j]  (symmetric, no min).

Here r = row within tile (0..31), bc = 32-col block within tile (0..7),
i = col within block (0..31).

The actual numeric quantization is delegated to ggml's own quantize() (Q4_1 /
Q8_0), so the per-block scale/min choice is bit-identical to llama.cpp's; this
module only performs the FLM byte re-layout.  Verified round-trip: bytes written
here, read back by tools/kernel-interp/q4nx.py, reconstruct the ggml-dequantized
weights exactly (see validate.py).
"""
import numpy as np
from gguf import quantize
from gguf.constants import GGMLQuantizationType


# ---- bf16 helpers (round-to-nearest-even, matches q4nx.py) -------------------
def f32_to_bf16_u16(f32):
    u = np.asarray(f32, dtype=np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_u16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


# ---- 16-lane interleave index (shared by q4 nibbles and q8 bytes) -----------
def _lane_index():
    """Return a (32,8,32) int array P where P[r,bc,i] is the destination
    nibble/byte slot for element (row r, col bc*32+i) within a 32x256 tile."""
    r = np.arange(32)[:, None, None]
    bc = np.arange(8)[None, :, None]
    i = np.arange(32)[None, None, :]
    return ((r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)).astype(np.int64)


def _scale_index():
    """Return a (32,8) int array J where J[r,bc] = bc*32+r is the planar slot
    for the scale/min of block (row r, col-block bc)."""
    r = np.arange(32)[:, None]
    bc = np.arange(8)[None, :]
    return (bc * 32 + r).astype(np.int64)


_P = _lane_index()                          # (32,8,32) nibble/byte slot
_J = _scale_index()                         # (32,8)    scale planar slot
_J3 = np.broadcast_to(_J[:, :, None], (32, 8, 32))   # (32,8,32) scale slot per element


def _unpack_q4_1_ggml(qbytes, out_dim, in_dim):
    """ggml Q4_1 blob -> (d[out,ng], m[out,ng], q[out,in]) with ng=in//32.
    q in 0..15, value = q*d + m (ggml Q4_1 semantics)."""
    blk = np.frombuffer(qbytes, dtype=np.uint8).reshape(-1, 20)   # 2 d + 2 m + 16 qs
    nb = blk.shape[0]
    d = blk[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb)
    m = blk[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nb)
    qs = blk[:, 4:]                                   # (nb, 16)
    # ggml Q4_1 packs a 32-block as: low nibble of byte j = elem j, high = elem j+16
    q = np.empty((nb, 32), dtype=np.uint8)
    q[:, 0:16] = qs & 0x0F
    q[:, 16:32] = qs >> 4
    ng = in_dim // 32
    return (d.reshape(out_dim, ng), m.reshape(out_dim, ng),
            q.reshape(out_dim, in_dim))


def _unpack_q8_0_ggml(qbytes, out_dim, in_dim):
    """ggml Q8_0 blob -> (d[out,ng], q[out,in] int8), value = q*d."""
    blk = np.frombuffer(qbytes, dtype=np.uint8).reshape(-1, 34)   # 2 d + 32 int8
    nb = blk.shape[0]
    d = blk[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb)
    q = blk[:, 2:].copy().view(np.int8)
    ng = in_dim // 32
    return d.reshape(out_dim, ng), q.reshape(out_dim, in_dim)


def pack_q4_1(W):
    """f32 weight [out, in] -> uint8 array [out//32, in//256, 5120] (q4nx q4_1).

    out must be a multiple of 32 and in a multiple of 256 (all FLM matmul
    tensors satisfy this).  Quantization is ggml Q4_1 (per-32 block along in-dim,
    fp16 scale+min), re-laid into the FLM 16-lane FILE layout with bf16 scales.
    """
    W = np.ascontiguousarray(W, dtype=np.float32)
    out_dim, in_dim = W.shape
    assert out_dim % 32 == 0 and in_dim % 256 == 0, (out_dim, in_dim)
    d, m, q = _unpack_q4_1_ggml(quantize(W, GGMLQuantizationType.Q4_1), out_dim, in_dim)
    return _lay_q4(d, m, q, out_dim, in_dim)


def _tiles(a, nrow_t, ncol_t, w):
    """[out, cols] -> [nch, 32, w] with chunk f = tr*ncol_t + tc (raster tiles)."""
    return (a.reshape(nrow_t, 32, ncol_t, w).transpose(0, 2, 1, 3)
            .reshape(nrow_t * ncol_t, 32, w))


def _scatter_scale(vals):
    """vals [nch,32,8] (r,bc) -> [nch,256] at planar slot j=bc*32+r."""
    nch = vals.shape[0]
    out = np.empty((nch, 256), np.float32)
    out[:, _J.reshape(-1)] = vals.reshape(nch, -1)
    return out


_BATCH = 4096          # chunks per batch (bounds peak memory to ~a few hundred MB)
_PF = _P.reshape(-1)
_JF = _J.reshape(-1)


def _lay_q4(d, m, q, out_dim, in_dim):
    nrow_t, ncol_t = out_dim // 32, in_dim // 256
    nch = nrow_t * ncol_t
    dd = _tiles(d, nrow_t, ncol_t, 8)          # [nch,32,8]
    mm = _tiles(m, nrow_t, ncol_t, 8)
    qq = _tiles(q, nrow_t, ncol_t, 256).reshape(nch, 32, 8, 32)   # [nch,r,bc,i]
    out = np.empty((nch, 5120), np.uint8)
    for s in range(0, nch, _BATCH):
        e = min(s + _BATCH, nch)
        b = e - s
        sc = np.empty((b, 256), np.float32); sc[:, _JF] = dd[s:e].reshape(b, -1)
        mn = np.empty((b, 256), np.float32); mn[:, _JF] = mm[s:e].reshape(b, -1)
        out[s:e, 0:512] = f32_to_bf16_u16(sc).view(np.uint8).reshape(b, 512)
        out[s:e, 512:1024] = f32_to_bf16_u16(mn).view(np.uint8).reshape(b, 512)
        nib = np.empty((b, 8192), np.uint8); nib[:, _PF] = qq[s:e].reshape(b, -1)
        out[s:e, 1024:] = (nib[:, 0::2] & 0x0F) | ((nib[:, 1::2] & 0x0F) << 4)
    return out.reshape(nrow_t, ncol_t, 5120)


def pack_q8_0(W):
    """f32 weight [out, in] -> uint8 array [out//32, in//256, 8704] (q4nx q8, lm_head)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    out_dim, in_dim = W.shape
    assert out_dim % 32 == 0 and in_dim % 256 == 0, (out_dim, in_dim)
    d, q = _unpack_q8_0_ggml(quantize(W, GGMLQuantizationType.Q8_0), out_dim, in_dim)
    nrow_t, ncol_t = out_dim // 32, in_dim // 256
    nch = nrow_t * ncol_t
    dd = _tiles(d, nrow_t, ncol_t, 8)
    qq = _tiles(q.astype(np.int8), nrow_t, ncol_t, 256).reshape(nch, 32, 8, 32)
    out = np.empty((nch, 8704), np.uint8)
    for s in range(0, nch, _BATCH):
        e = min(s + _BATCH, nch)
        b = e - s
        sc = np.empty((b, 256), np.float32); sc[:, _JF] = dd[s:e].reshape(b, -1)
        out[s:e, 0:512] = f32_to_bf16_u16(sc).view(np.uint8).reshape(b, 512)
        buf = np.empty((b, 8192), np.int8); buf[:, _PF] = qq[s:e].reshape(b, -1)
        out[s:e, 512:] = buf.view(np.uint8)
    return out.reshape(nrow_t, ncol_t, 8704)


# ---- self-check readers (mirror q4nx.py, for validate.py) -------------------
def dequant_q4_1_file(packed_u8, out_dim, in_dim):
    """[.,.,5120] uint8 (or flat) -> f32 [out,in].  Mirrors q4nx.py dq path."""
    chunks = np.ascontiguousarray(packed_u8).reshape(-1, 5120)
    nch = chunks.shape[0]
    meta = bf16_u16_to_f32(np.ascontiguousarray(chunks[:, :1024]).view(np.uint16))
    d, m = meta[:, :256], meta[:, 256:]
    qb = chunks[:, 1024:]
    n = np.empty((nch, 8192), dtype=np.float32)
    n[:, 0::2] = qb & 0xF
    n[:, 1::2] = qb >> 4
    vals = n[:, _P.reshape(-1)].reshape(nch, 32, 8, 32)
    dd = d[:, _J3.reshape(-1)].reshape(nch, 32, 8, 32)
    mm = m[:, _J3.reshape(-1)].reshape(nch, 32, 8, 32)
    w = (vals * dd + mm).reshape(nch, 32, 256)
    ncol = in_dim // 256
    W = np.empty((out_dim, in_dim), np.float32)
    for f in range(nch):
        W[32 * (f // ncol):32 * (f // ncol) + 32, 256 * (f % ncol):256 * (f % ncol) + 256] = w[f]
    return W


def dequant_q8_0_file(packed_u8, out_dim, in_dim):
    chunks = np.ascontiguousarray(packed_u8).reshape(-1, 8704)
    nch = chunks.shape[0]
    d = bf16_u16_to_f32(np.ascontiguousarray(chunks[:, :512]).view(np.uint16))
    qb = np.ascontiguousarray(chunks[:, 512:]).view(np.int8).astype(np.float32)
    vals = qb[:, _P.reshape(-1)].reshape(nch, 32, 8, 32)
    dd = d[:, _J3.reshape(-1)].reshape(nch, 32, 8, 32)
    w = (vals * dd).reshape(nch, 32, 256)
    ncol = in_dim // 256
    W = np.empty((out_dim, in_dim), np.float32)
    for f in range(nch):
        W[32 * (f // ncol):32 * (f // ncol) + 32, 256 * (f % ncol):256 * (f % ncol) + 256] = w[f]
    return W
