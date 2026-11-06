r"""The fp64 reference for a Qwen3.5 dense model straight from a llama.cpp `qwen35` GGUF
(OPEN-FAMILY-SALUKI): replica_qwen35.py's layer math over the GGUF's own values.

Every tensor is dequantized by llama.cpp's gguf-py (`gguf.quants.dequantize`, the i-quants included),
which is the arithmetic definition of the published weights; the conversion's maps are undone in NumPy
(the per-head [q | gate] interleave of attn_q, the tiled linear-attention value heads of attn_qkv /
attn_gate / ssm_out / ssm_alpha / ssm_beta / ssm_a / ssm_dt.bias / ssm_conv1d, conv1d's [channels, taps]
layout). Nothing here shares code with the runtime's q35 ops or with llama.cpp's compute path, so it is
independent of both. Activations are fp64; weights are held as f32 (gguf-py's output).

Runs the prompt layer-major through the first N layers (all positions), then the final norm and head at
every position, and writes:

    <out>/ref_logits_p{pos}.bin    f32[vocab]  the N-layer model's logits at each position
    <out>/ref_res{l}.bin           f32[T, hidden]  layer l's output at every position

    python -I open_kernels/model/saluki_gguf_reference.py --gguf model.gguf --model-dir DIR \
        --ids-file short.cids --layers 8 --out OUT --gguf-py <llama.cpp>/gguf-py

The NPU side for the same truncated model: `open_qwen36_cli --layers N --ids-file ... --max-tokens 1
--prefill-logits --dump-logits P` (P_p{pos}.bin).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from recipes.load import spec_from_model_dir  # noqa: E402
import replica_qwen35 as R35  # noqa: E402


def untile(n: int, kheads: int, vperk: int, vdim: int, v0: int) -> np.ndarray:
    """HF (grouped) index -> GGUF (tiled) index for the value-head axis from element v0."""
    i = np.arange(n)
    rel = i - v0
    h = rel // vdim
    k, j = h // vperk, h % vperk
    return np.where(i >= v0, v0 + (j * kheads + k) * vdim + rel % vdim, i)


class GgufWeights:
    """replica_qwen35's weight interface (`bf16`, `f32`, `matmul_w`, `embed`, `lmhead_logits`) over a GGUF."""

    def __init__(self, path: str, spec, gguf_py: str | None):
        if gguf_py:
            sys.path.insert(0, gguf_py)
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.dq = dequantize
        self.t = {t.name: t for t in GGUFReader(path).tensors}
        self.spec = spec
        s = spec
        self.kh, self.vperk = s.lin_key_heads, s.lin_value_heads // s.lin_key_heads

    def raw(self, name: str, rows: slice | None = None) -> np.ndarray:
        t = self.t[name]
        shape = tuple(int(x) for x in reversed(t.shape))
        data = t.data if rows is None else t.data[rows]
        a = self.dq(data, t.tensor_type).astype(np.float32)
        return a.reshape(shape if rows is None else (-1,) + shape[1:])

    def _vt(self, a: np.ndarray, v0: int, vdim: int, axis: int = 0) -> np.ndarray:
        if self.kh == self.spec.lin_value_heads:
            return a
        idx = untile(a.shape[axis], self.kh, self.vperk, vdim, v0)
        return a[idx] if axis == 0 else a[:, idx]

    def _hf(self, name: str) -> np.ndarray:
        s = self.spec
        layer, stem = name.split(".", 3)[2], name.split(".", 3)[3]
        g = f"blk.{layer}."
        kd, vd, kh = s.lin_key_dim, s.lin_value_dim, s.lin_key_heads
        m = {
            "input_layernorm.weight": ("attn_norm.weight", None),
            "post_attention_layernorm.weight": ("post_attention_norm.weight", None),
            "self_attn.q_norm.weight": ("attn_q_norm.weight", None),
            "self_attn.k_norm.weight": ("attn_k_norm.weight", None),
            "linear_attn.ssm_norm.weight": ("ssm_norm.weight", None),
            "self_attn.k_proj.weight": ("attn_k.weight", None),
            "self_attn.v_proj.weight": ("attn_v.weight", None),
            "self_attn.o_proj.weight": ("attn_output.weight", None),
            "mlp.up_proj.weight": ("ffn_up.weight", None),
            "mlp.gate_proj.weight": ("ffn_gate.weight", None),
            "mlp.down_proj.weight": ("ffn_down.weight", None),
        }
        if stem in m:
            return self.raw(g + m[stem][0])
        if stem == "self_attn.q_proj.weight":
            W = self.raw(g + "attn_q.weight")                      # [nh * (q_h | gate_h), hid]
            nh, hd = s.num_heads, s.head_dim
            W = W.reshape(nh, 2, hd, -1)
            return np.concatenate([W[:, 0].reshape(nh * hd, -1), W[:, 1].reshape(nh * hd, -1)], 0)
        if stem == "linear_attn.qkv_proj.weight":
            return self._vt(self.raw(g + "attn_qkv.weight"), 2 * kh * kd, vd)
        if stem == "self_attn.gate_proj.weight":                   # a linear layer's z
            return self._vt(self.raw(g + "attn_gate.weight"), 0, vd)
        if stem == "linear_attn.ssm_out_proj.weight":
            return self._vt(self.raw(g + "ssm_out.weight"), 0, vd, axis=1)
        if stem in ("linear_attn.ssm_alpha_proj.bf16.weight", "linear_attn.ssm_beta_proj.bf16.weight"):
            return self._vt(self.raw(g + ("ssm_alpha.weight" if "alpha" in stem else "ssm_beta.weight")), 0, 1)
        if stem in ("linear_attn.ssm_a", "linear_attn.ssm_dt.bias"):
            return self._vt(self.raw(g + ("ssm_a" if stem.endswith("ssm_a") else "ssm_dt.bias")).ravel(), 0, 1)
        if stem == "linear_attn.ssm_conv1d.weight":                # GGUF [channels, taps] -> [taps, channels]
            return self._vt(self.raw(g + "ssm_conv1d.weight"), 2 * kh * kd, vd).T
        raise KeyError(name)

    def bf16(self, name: str) -> np.ndarray:
        if name == "model.norm.weight":
            return self.raw("output_norm.weight")
        return self._hf(name)

    f32 = bf16

    def matmul_w(self, name: str, rows: int, cols: int) -> np.ndarray:
        W = self._hf(name)
        assert W.shape == (rows, cols), (name, W.shape, rows, cols)
        return W

    def embed(self, tok: int, hidden: int) -> np.ndarray:
        return self.raw("token_embd.weight", slice(tok, tok + 1))[0]

    def lmhead_logits(self, hn: np.ndarray) -> np.ndarray:
        V = self.spec.vocab
        out = np.empty(V, np.float64)
        h = np.asarray(hn, np.float64)
        for r0 in range(0, V, 16384):
            out[r0:r0 + 16384] = self.raw("output.weight", slice(r0, min(V, r0 + 16384))).astype(np.float64) @ h
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True, help="config.json + tokenizer.json (the spec)")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = spec_from_model_dir(Path(a.model_dir))
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()]
    m = GgufWeights(a.gguf, spec, a.gguf_py)
    cache: dict = {}
    plain = m.matmul_w
    m.matmul_w = lambda name, o, i: cache[name] if name in cache else cache.setdefault(name, plain(name, o, i))
    T = len(ids)
    xs = [m.embed(t, spec.hidden).astype(np.float64) for t in ids]
    t0 = time.time()
    for l in range(min(a.layers, spec.num_layers)):
        cache.clear()
        conv = np.zeros((spec.conv_kernel - 1, spec.lin_qkv_dim))
        S = np.zeros((spec.lin_value_heads, spec.lin_value_dim, spec.lin_value_dim))
        K = np.zeros((0, spec.num_kv_heads, spec.head_dim))
        V = np.zeros((0, spec.num_kv_heads, spec.head_dim))
        for t in range(T):
            xs[t], conv, S, K, V = R35.layer_decode(m, spec, l, xs[t], conv, S, K, V, t)
            xs[t] = np.asarray(xs[t], np.float64)
        np.stack(xs).astype(np.float32).tofile(out / f"ref_res{l}.bin")
        print(f"  layer {l:2d} {spec.layer_types[l]:17s} |res[-1]| {np.linalg.norm(xs[-1]):10.3f}  ({time.time() - t0:.0f} s)",
              flush=True)
    # final norm and head at every position, the head dequantized once (row blocks)
    hn = np.stack([R35.rms(x, spec.norm_eps) * m.bf16("model.norm.weight") for x in xs]).astype(np.float64)
    logits = np.empty((T, spec.vocab), np.float64)
    for r0 in range(0, spec.vocab, 16384):
        Wh = m.raw("output.weight", slice(r0, min(spec.vocab, r0 + 16384))).astype(np.float64)
        logits[:, r0:r0 + Wh.shape[0]] = hn @ Wh.T
    for t in range(T):
        logits[t].astype(np.float32).tofile(out / f"ref_logits_p{t}.bin")
    print(f"{T} positions written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
