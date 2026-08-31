"""q4nx container access: header parse + raw tensor reads + dequant hypotheses.

Layout facts derived from the safetensors header of model_3LiF.q4nx:
  - I8 tensors are packed chunks: shape [A, B, CHUNK_BYTES], A*B chunks.
  - CHUNK_BYTES=5120 -> q4 chunk: 8192 elems (4096B nibbles + 128 blocks x (4B scale + 4B zero))
  - CHUNK_BYTES=8704 -> q8 chunk: 8192 elems (8192B int8   + 128 blocks x 4B scale)  [lm_head]
  - BF16 tensors are plain row-major.
Block size = 64 elements. Exact intra-chunk ordering TBD (this module hosts the
hypotheses; verified against NPU weight-pool dumps).
"""
import json, struct, mmap, os
import numpy as np
import q4nx_v103 as v103

MODEL_DIR = r"C:/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2"


class Q4NX:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.data_base = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.tensors = {k: v for k, v in self.header.items() if k != "__metadata__"}
        self.fmt, self.chunk_bytes = self._detect_format()

    def _detect_format(self):
        """FLM 1.0.2 (q4_1, 5120 B) vs 1.0.3 (Q4_K, 4736 B), by a non-lm_head I8
        tensor's chunk size.  No version field exists in the header."""
        for k, v in self.tensors.items():
            if v.get("dtype") == "I8" and k != "lm_head.weight":
                cb = v["shape"][-1]
                if cb == v103.Q4K_CHUNK:
                    return "q4k", v103.Q4K_CHUNK
                return "q4_1", 5120
        return "q4_1", 5120

    def _layer_type(self, name):
        """'linear' | 'full' | None for the layer a tensor belongs to."""
        if ".layer." not in name:
            return None
        l = name.split(".layer.")[1].split(".")[0]
        if f"model.layer.{l}.linear_attn.qkv_proj.weight" in self.tensors:
            return "linear"
        if f"model.layer.{l}.self_attn.q_proj.weight" in self.tensors:
            return "full"
        return None

    def raw(self, name):
        t = self.tensors[name]
        o0, o1 = t["data_offsets"]
        return self.mm[self.data_base + o0 : self.data_base + o1]

    def bf16(self, name):
        t = self.tensors[name]
        assert t["dtype"] == "BF16", t
        a = np.frombuffer(self.raw(name), dtype=np.uint16).reshape(t["shape"])
        w = bf16_to_f32(a)
        if self.fmt == "q4k":
            w = v103.apply_undo(name, np.ascontiguousarray(w), self._layer_type(name))
        return w

    def f32(self, name):
        t = self.tensors[name]
        assert t["dtype"] == "F32", t
        w = np.frombuffer(self.raw(name), dtype=np.float32).reshape(t["shape"])
        if self.fmt == "q4k":
            w = v103.apply_undo(name, np.ascontiguousarray(w), self._layer_type(name))
        return w

    # ---- format-aware quantized reads (q4_1 path is byte-identical to before) --
    def dq_tile(self, raw_bytes, out_dim, in_dim):
        """Raw quantized chunk bytes -> [out,in] fp, PLAIN RASTER, NO reorder.
        Used for expert slices (experts carry no head-pairing reorder)."""
        b = np.frombuffer(raw_bytes, dtype=np.uint8)
        if self.fmt == "q4k":
            return v103.dequant_q4k_file(b, out_dim, in_dim)
        chunks = b.reshape(-1, 5120)
        w = _dq_chunks_q4_1(chunks).reshape(-1, 32, 256)
        ncol = in_dim // 256
        W = np.empty((out_dim, in_dim), np.float32)
        for f in range(w.shape[0]):
            W[32 * (f // ncol):32 * (f // ncol) + 32,
              256 * (f % ncol):256 * (f % ncol) + 256] = w[f]
        return W

    def matmul_w(self, name, out_dim, in_dim):
        """Dequant a full quantized matmul tensor to logical [out,in] (reorder-undone)."""
        W = self.dq_tile(self.raw(name), out_dim, in_dim)
        if self.fmt == "q4k":
            W = v103.apply_undo(name, W, self._layer_type(name))
        return W

    def lmhead_logits(self, hn):
        """Stream the q8 lm_head against hidden hn[hidden] -> logits[vocab]."""
        t = self.tensors["lm_head.weight"]
        out_dim = t["shape"][0] * 32
        lm = np.frombuffer(self.raw("lm_head.weight"), dtype=np.uint8)
        hn = np.asarray(hn, np.float32)
        logits = np.zeros(out_dim, np.float32)
        if self.fmt == "q4k":
            lmb = lm.reshape(-1, v103.Q8_CHUNK)
            nch = lmb.shape[0]
            d = v103.bf16_to_f32(np.ascontiguousarray(lmb[:, :512]).view(np.uint16))  # [nch,256] g*32+r
            data = np.ascontiguousarray(lmb[:, 512:]).view(np.int8).astype(np.float32)  # [nch,8192] c*32+r
            r = np.arange(32); c = np.arange(256); g = c // 32
            sm = (g[None, :] * 32 + r[:, None]).reshape(-1)
            ds = (c[None, :] * 32 + r[:, None]).reshape(-1)
            for cc in range(nch):
                w = (d[cc][sm] * data[cc][ds]).reshape(32, 256)
                logits[32 * (cc // 8):32 * (cc // 8) + 32] += w @ hn[256 * (cc % 8):256 * (cc % 8) + 256]
            return logits
        # q4_1: 16-lane interleave (matches full_forward's inline path)
        lmb = lm.reshape(-1, 8704)
        nch = lmb.shape[0]
        d = bf16_to_f32(np.ascontiguousarray(lmb[:, :512]).view(np.uint16))
        qq = np.ascontiguousarray(lmb[:, 512:]).view(np.int8)
        r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
        p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)
        j = bc * 32 + r + 0 * i
        for cc in range(nch):
            vals = qq[cc][p.reshape(-1)].reshape(32, 8, 32).astype(np.float32)
            dd = d[cc][j.reshape(-1)].reshape(32, 8, 32)
            w = (vals * dd).reshape(32, 256)
            logits[32 * (cc // 8):32 * (cc // 8) + 32] += w @ hn[256 * (cc % 8):256 * (cc % 8) + 256]
        return logits


def _dq_chunks_q4_1(chunks):
    """[n,5120]->[n,32,8,32] f32 (q4_1 16-lane; mirror of moe_forward.dq_chunks)."""
    nch = chunks.shape[0]
    meta = bf16_to_f32(np.ascontiguousarray(chunks[:, :1024]).view(np.uint16))
    d, mn = meta[:, :256], meta[:, 256:]
    q = chunks[:, 1024:]
    n = np.empty((nch, 8192), dtype=np.float32)
    n[:, 0::2] = q & 0xF
    n[:, 1::2] = q >> 4
    r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)
    j = bc * 32 + r + 0 * i
    vals = n[:, p.reshape(-1)].reshape(nch, 32, 8, 32)
    dd = d[:, j.reshape(-1)].reshape(nch, 32, 8, 32)
    mm_ = mn[:, j.reshape(-1)].reshape(nch, 32, 8, 32)
    return vals * dd + mm_


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    """round-to-nearest-even bf16 encode -> uint16"""
    u = np.asarray(f32, dtype=np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


# ---- chunk dequant hypotheses ----------------------------------------------
def dequant_q4_chunk(chunk, order="nibbles_first", scale_dtype="f32"):
    """One 5120-byte chunk -> 8192 float elems, under a layout hypothesis."""
    c = np.frombuffer(chunk, dtype=np.uint8)
    assert c.size == 5120
    if order == "nibbles_first":
        q, meta = c[:4096], c[4096:]
    else:  # meta_first
        meta, q = c[:1024], c[1024:]
    lo = (q & 0xF).astype(np.int32)
    hi = (q >> 4).astype(np.int32)
    # interleave: assume elem 2i = lo, 2i+1 = hi of byte i
    vals = np.empty(8192, dtype=np.int32)
    vals[0::2] = lo
    vals[1::2] = hi
    if scale_dtype == "f32":
        sc = np.frombuffer(meta[:512].tobytes(), dtype=np.float32)
        zp = np.frombuffer(meta[512:].tobytes(), dtype=np.int32)
    else:  # bf16 scales then i32 zp, padded
        sc = bf16_to_f32(np.frombuffer(meta[:256].tobytes(), dtype=np.uint16))
        zp = np.frombuffer(meta[256:768].tobytes(), dtype=np.int32)
    vals = vals.reshape(128, 64)
    return (vals - zp[:, None]) * sc[:, None]


def dequant_q4nx_packed(packed, out_dim, in_dim):
    """Verbatim q4nx packed bytes -> f32 [out_dim, in_dim].

    Verified byte-level layout (2026-08-30, vs HF reference weights):
      chunk = 5120 B: 256 bf16 scales d + 256 bf16 mins m (planar, q4_1 style,
      block=32 along in_dim), then 4096 B nibbles, 16-lane interleaved:
        nibble[(r//16)*4096 + bc*512 + i*16 + (r%16)] = elem(row=r, col=bc*32+i)
      (byte b -> lo nibble = even nibble index, hi = odd)
      chunk c covers (per_band = in_dim//128 chunks per 64-row band):
        rows 64*(c//per_band)+32*(c%2) .. +31
        cols 1024*((c//8)%(in_dim//1024))+256*((c//2)%4) .. +255
      (verified for in_dim 2048 [qkv, z] and 4096 [out_proj])
    """
    chunks = packed.reshape(-1, 5120)
    nch = chunks.shape[0]
    assert nch * 8192 == out_dim * in_dim
    meta = bf16_to_f32(np.ascontiguousarray(chunks[:, :1024]).view(np.uint16))
    d, m = meta[:, :256], meta[:, 256:]
    q = chunks[:, 1024:]
    n = np.empty((nch, 8192), dtype=np.float32)
    n[:, 0::2] = q & 0xF
    n[:, 1::2] = q >> 4
    r = np.arange(32)[:, None, None]
    bc = np.arange(8)[None, :, None]
    i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)
    j = (bc * 32 + r + 0 * i)
    vals = n[:, p.reshape(-1)].reshape(nch, 32, 8, 32)
    dd = d[:, j.reshape(-1)].reshape(nch, 32, 8, 32)
    mm_ = m[:, j.reshape(-1)].reshape(nch, 32, 8, 32)
    w = (vals * dd + mm_).reshape(nch, 32, 256)
    c = np.arange(nch)
    per_band = in_dim // 128
    rows0 = 64 * (c // per_band) + 32 * (c % 2)
    cols0 = 1024 * ((c // 8) % (in_dim // 1024)) + 256 * ((c // 2) % 4)
    W = np.empty((out_dim, in_dim), dtype=np.float32)
    for ci in range(nch):
        W[rows0[ci]:rows0[ci] + 32, cols0[ci]:cols0[ci] + 256] = w[ci]
    return W
