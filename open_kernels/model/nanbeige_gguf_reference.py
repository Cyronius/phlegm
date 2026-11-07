r"""The fp64 reference for a dense llama.cpp `llama` GGUF (Nanbeige4.3-33B, OPEN-FAMILY-NANBEIGE): the
published LlamaForCausalLM forward over the GGUF's own values.

Every tensor is dequantized by llama.cpp's gguf-py (`gguf.quants.dequantize`), the arithmetic definition
of the file's weights. Q and K stay in the GGUF's row order, which llama.cpp's converter permuted so a
head's rotary pairs are ADJACENT rows (convert_hf_to_gguf LlamaModel.permute); the rotation here is
therefore llama.cpp's "normal" RoPE over (2i, 2i + 1), not HF's rotate_half over (i, i + hd / 2) -- the
same function of the HF weights, computed without the runtime's un-permute (pools.cpp std_perm_gguf), so
this checks that too. Nothing here shares code with the runtime, the recipe or llama.cpp's compute path.
Activations are fp64; weights are held as f32 (gguf-py's output) and widened per product.

  per layer:  h = rms(x) * attn_norm;  q, k, v = h Wq^T, h Wk^T, h Wv^T
              q, k: rotate pairs (2i, 2i+1) of every head by pos * theta^(-2i / hd)
              a = softmax(q k^T / sqrt(hd), causal) v, q head j on kv head j // (heads / kv heads)
              x += a Wo^T;  m = rms(x) * ffn_norm;  x += (silu(m Wg^T) * (m Wu^T)) Wd^T
  head:       logits = rms(x) * output_norm @ output^T

Writes, for the first N layers (all of them by default):

    <out>/ref_logits_p{pos}.bin    f32[vocab]       the N-layer model's logits at each requested position
    <out>/ref_res{l}.bin           f32[T, hidden]   layer l's output at every position (--dump-res)

    python -I open_kernels/model/nanbeige_gguf_reference.py --gguf model.gguf --ids-file short.cids \
        --out OUT [--layers 8] [--positions all|last|a,b,...] [--dump-res] [--gguf-py <llama.cpp>/gguf-py]

The NPU side for the same truncated model: `open_qwen36_cli --layers N --ids-file ... --max-tokens 1
--prefill-logits --dump-logits P` (P_p{pos}.bin); compare with saluki/scripts/cmp_ref.py-style statistics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


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

    def meta(self, key: str):
        f = self.kv[key]
        v = f.parts[f.data[0]]
        return v.tobytes().decode() if f.types and f.types[0].name == "STRING" else v[0].item()

    def w(self, name: str) -> np.ndarray:
        """The tensor as f32 [rows, cols] (GGUF's fastest dim is the row's)."""
        t = self.t[name]
        a = self.dq(t.data, t.tensor_type).astype(np.float32, copy=False)
        return a.reshape([int(d) for d in reversed(t.shape)]) if a.ndim == 1 and len(t.shape) > 1 else a

    def rows(self, name: str, ids) -> np.ndarray:
        t = self.t[name]
        cols = int(t.shape[0])
        data = t.data.reshape(int(t.shape[1]), -1)
        return np.stack([self.dq(data[i], t.tensor_type).reshape(-1)[:cols] for i in ids]).astype(np.float64)


def rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope_pairs(x: np.ndarray, pos: np.ndarray, theta: float, hd: int) -> np.ndarray:
    """llama.cpp's normal RoPE: x [T, heads, hd], pairs (2i, 2i + 1) rotated by pos * theta^(-2i/hd)."""
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = pos[:, None].astype(np.float64) * inv[None, :]                # [T, hd/2]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
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
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="the first N layers (0: all)")
    ap.add_argument("--positions", default="all", help="all | last | comma-separated positions")
    ap.add_argument("--dump-res", action="store_true", help="write every layer's output")
    ap.add_argument("--gguf-py", default=None)
    a = ap.parse_args()
    g = Gguf(a.gguf, a.gguf_py)
    arch = g.meta("general.architecture")
    if arch != "llama":
        raise SystemExit(f"{a.gguf}: architecture {arch!r}; this reference is the llama forward")
    nl = int(g.meta("llama.block_count"))
    hid = int(g.meta("llama.embedding_length"))
    nh = int(g.meta("llama.attention.head_count"))
    nkv = int(g.meta("llama.attention.head_count_kv"))
    eps = float(g.meta("llama.attention.layer_norm_rms_epsilon"))
    theta = float(g.meta("llama.rope.freq_base"))
    hd = hid // nh
    if "llama.attention.key_length" in g.kv:
        hd = int(g.meta("llama.attention.key_length"))
    N = a.layers or nl
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()]
    T = len(ids)
    pos = np.arange(T)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"llama GGUF: {nl} layers ({N} run), hidden {hid}, heads {nh}/{nkv} x {hd}, theta {theta:g}, "
          f"eps {eps:g}; {T} tokens", flush=True)
    t0 = time.time()
    x = g.rows("token_embd.weight", ids)
    for l in range(N):
        p = f"blk.{l}."
        h = rms(x, g.w(p + "attn_norm.weight").astype(np.float64), eps)
        q = h @ g.w(p + "attn_q.weight").T
        k = h @ g.w(p + "attn_k.weight").T
        v = h @ g.w(p + "attn_v.weight").T
        q = rope_pairs(q.reshape(T, nh, hd), pos, theta, hd).reshape(T, -1)
        k = rope_pairs(k.reshape(T, nkv, hd), pos, theta, hd).reshape(T, -1)
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
    W = g.w("output.weight")
    logits = np.empty((len(ps), W.shape[0]))
    for r0 in range(0, W.shape[0], 16384):
        logits[:, r0:r0 + 16384] = hn @ W[r0:r0 + 16384].T
    for i, pp in enumerate(ps):
        logits[i].astype(np.float32).tofile(out / f"ref_logits_p{pp}.bin")
    am = logits.argmax(axis=1)
    print(f"logits at {len(ps)} positions ({time.time() - t0:.0f} s); argmax {am[:8].tolist()} ... last {int(am[-1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
