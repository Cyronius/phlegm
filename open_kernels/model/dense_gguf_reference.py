"""Independent fp64 dense GGUF forward and NPU-like weight-carriage control.

Supports llama (adjacent RoPE) and qwen3 (split RoPE, q/k RMS norms), tied
or untied heads, and GGUF rope_freqs.weight. Other architectures are refused
until their forward is implemented. --carriage applies the dense pool rounding
law, not device activation arithmetic; it is an offline control, not G2.
Run with python -I and a trusted --gguf-py path.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_gguf_check import KEPT, INT8, q8_target
from dense_gguf_npulike import bf16, kept_q4


class Gguf:
    def __init__(self, path: str, gguf_py: str | None):
        if gguf_py:
            sys.path.insert(0, gguf_py)
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.dq = dequantize
        self.r = GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        self.kv = {f.name: f for f in self.r.fields.values()}
        self.carriage = False

    def meta(self, key: str):
        f = self.kv[key]
        v = f.parts[f.data[0]]
        return v.tobytes().decode() if f.types and f.types[0].name == "STRING" else v[0].item()

    def w(self, name: str, *, head=False) -> np.ndarray:
        """The tensor as f32 [rows, cols] (GGUF's fastest dim is the row's)."""
        t = self.t[name]
        a = self.dq(t.data, t.tensor_type).astype(np.float32, copy=False)
        a = a.reshape([int(d) for d in reversed(t.shape)])
        if self.carriage:
            ty = t.tensor_type.name
            if head or (a.ndim == 2 and ty in INT8):
                return q8_target(t.data, ty, a, head=head)
            if a.ndim == 2 and ty in KEPT:
                return kept_q4(t.data, ty).reshape(a.shape)
            if name.endswith('norm.weight'):
                return bf16(a)
        return a

    def rows(self, name: str, ids) -> np.ndarray:
        t = self.t[name]
        cols = int(t.shape[0])
        data = t.data.reshape(int(t.shape[1]), -1)
        return np.stack([self.dq(data[i], t.tensor_type).reshape(-1)[:cols] for i in ids]).astype(np.float64)


def rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope_pairs(x: np.ndarray, pos: np.ndarray, theta: float, hd: int, freqs=None, split=False) -> np.ndarray:
    """llama.cpp's normal RoPE: x [T, heads, hd], pairs (2i, 2i + 1) rotated by pos * theta^(-2i/hd)."""
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    if freqs is not None: inv /= freqs
    ang = pos[:, None].astype(np.float64) * inv[None, :]                # [T, hd/2]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    if split:
        a, b = np.split(x, 2, axis=-1)
        return np.concatenate((a * c - b * s, a * s + b * c), axis=-1)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def attention(q, k, v, nh, nkv, hd):
    T = q.shape[0]
    q = q.reshape(T, nh, hd)
    k = k.reshape(T, nkv, hd)
    v = v.reshape(T, nkv, hd)
    grp = nh // nkv
    out = np.empty((T, nh, hd))
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for h in range(nh):
        s = q[:, h, :] @ k[:, h // grp, :].T / np.sqrt(hd) + mask
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        out[:, h, :] = p @ v[:, h // grp, :]
    return out.reshape(T, nh * hd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--carriage", action="store_true", help="NPU-like weights, fp64 activations")
    ap.add_argument("--logits-file", help="also write one contiguous f32 [positions,vocab] file")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="the first N layers (0: all)")
    ap.add_argument("--positions", default="all", help="all | last | comma-separated positions")
    ap.add_argument("--dump-res", action="store_true", help="write every layer's output")
    ap.add_argument("--gguf-py", default=None)
    a = ap.parse_args()
    g = Gguf(a.gguf, a.gguf_py)
    arch = g.meta("general.architecture")
    g.carriage = a.carriage
    if arch not in ("llama", "qwen3"):
        raise SystemExit(f"{a.gguf}: architecture {arch!r}; this reference supports llama and qwen3")
    nl = int(g.meta(arch + ".block_count"))
    hid = int(g.meta(arch + ".embedding_length"))
    nh = int(g.meta(arch + ".attention.head_count"))
    nkv = int(g.meta(arch + ".attention.head_count_kv"))
    eps = float(g.meta(arch + ".attention.layer_norm_rms_epsilon"))
    theta = float(g.meta(arch + ".rope.freq_base"))
    hd = hid // nh
    if arch + ".attention.key_length" in g.kv:
        hd = int(g.meta(arch + ".attention.key_length"))
    N = a.layers or nl
    if not 1 <= N <= nl: raise ValueError("layers out of range")
    freqs = g.w('rope_freqs.weight') if 'rope_freqs.weight' in g.t else None
    if freqs is None and arch + '.rope.scaling.type' in g.kv:
        kind = g.meta(arch + '.rope.scaling.type')
        if kind not in ('none',): raise ValueError(f"unsupported RoPE scaling without rope_freqs: {kind}")
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()]
    T = len(ids)
    pos = np.arange(T)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{arch} GGUF: {nl} layers ({N} run), hidden {hid}, heads {nh}/{nkv} x {hd}, theta {theta:g}, "
          f"eps {eps:g}; {T} tokens", flush=True)
    t0 = time.time()
    x = g.rows("token_embd.weight", ids)
    for l in range(N):
        p = f"blk.{l}."
        h = rms(x, g.w(p + "attn_norm.weight").astype(np.float64), eps)
        q = h @ g.w(p + "attn_q.weight").T
        k = h @ g.w(p + "attn_k.weight").T
        v = h @ g.w(p + "attn_v.weight").T
        q, k = q.reshape(T, nh, hd), k.reshape(T, nkv, hd)
        if arch == 'qwen3':
            q = rms(q, g.w(p + 'attn_q_norm.weight'), eps)
            k = rms(k, g.w(p + 'attn_k_norm.weight'), eps)
        q = rope_pairs(q, pos, theta, hd, freqs, arch == 'qwen3').reshape(T, -1)
        k = rope_pairs(k, pos, theta, hd, freqs, arch == 'qwen3').reshape(T, -1)
        x = x + attention(q, k, v, nh, nkv, hd) @ g.w(p + "attn_output.weight").T
        m = rms(x, g.w(p + "ffn_norm.weight").astype(np.float64), eps)
        gt = m @ g.w(p + "ffn_gate.weight").T
        up = m @ g.w(p + "ffn_up.weight").T
        x = x + (gt / (1.0 + np.exp(-gt)) * up) @ g.w(p + "ffn_down.weight").T
        if a.dump_res:
            x.astype(np.float32).tofile(out / f"ref_res{l}.bin")
        print(f"  layer {l}: |x| {np.abs(x).max():.3f}  {time.time() - t0:.0f} s", flush=True)
    if a.positions == "all":
        ps = list(range(T))
    elif a.positions == "last":
        ps = [T - 1]
    else:
        ps = [int(s) for s in a.positions.split(",")]
    hn = rms(x[ps], g.w("output_norm.weight").astype(np.float64), eps)
    W = g.w("output.weight" if "output.weight" in g.t else "token_embd.weight", head=True)
    logits = np.empty((len(ps), W.shape[0]))
    for r0 in range(0, W.shape[0], 16384):
        logits[:, r0:r0 + 16384] = hn @ W[r0:r0 + 16384].T
    for i, pp in enumerate(ps):
        logits[i].astype(np.float32).tofile(out / f"ref_logits_p{pp}.bin")
    if a.logits_file: logits.astype('<f4').tofile(a.logits_file)
    am = logits.argmax(axis=1)
    print(f"logits at {len(ps)} positions ({time.time() - t0:.0f} s); argmax {am[:8].tolist()} ... last {int(am[-1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
