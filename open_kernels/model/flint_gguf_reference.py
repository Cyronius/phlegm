"""fp32 numpy forward of Recursive Flint over its GGUF -- the independent reference the NPU route is
checked against (OPEN-FAMILY-FLINT).

It reads the GGUF directly (RecursiveSI's flint-int4-q4_1.gguf: every matrix Q4_1, the embedding and
every norm / sink / router F32; Q4_0, Q8_0 and BF16 are read too) and implements the published model --
transformers' modeling_flint.py (modular_flint.py) and llama.cpp's src/models/flint.cpp, which agree --
not this tree's recipe: nothing is padded, regrouped or requantised.

  per layer:  h = rms(x) * attn_norm
              q, k, v = h Wq, h Wk, h Wv;  q, k = rms over each head * q_norm / k_norm  (Qwen3)
              sliding layers: RoPE (half split, theta 1e4, every dim); full layers: none (NoPE)
              attention with the learned per-head sink (an extra softmax logit with no value);
              a sliding layer sees the newest `sliding_window` rows, itself included (513)
              res = x + a Wo;  xm = rms(res) * ffn_norm
              dense layers (mlp_only_layers): x = res + down(silu(gate xm) * up xm)
              MoE layers: s = sigmoid(xm Wr) (F32 router, no bias); the top k of s, renormalised;
                          x = res + sum_e w_e down_e(silu(gate_e xm) * up_e xm) + shared(xm)
              (the shared expert is ungated, always added in full)
  head:       logits = rms(x) * output_norm @ output

The forward is layer-major over the whole sequence, so each expert a layer routes to is decoded
once for every token that picked it:

    python open_kernels/model/flint_gguf_reference.py --gguf model.gguf --config config.json \
        --ids-file prompt.ids --out DIR [--dump-layers 0,2,48] [--positions all|last|N,M] [--npu-like]

writes `logits.f32` ([T, vocab] at the chosen positions), `positions.json`, `routes.json`, the
per-layer states asked for (`layer{l}_res.f32` after attention, `layer{l}_xm.f32` the FFN norm,
`layer{l}_out.f32` the layer output, `layer{l}_q/k/v.f32` after norm + rotation) and `report.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gguf_pool import read_gguf  # noqa: E402

F32, F16, Q4_0, Q4_1, Q8_0, BF16 = 0, 1, 2, 3, 8, 30
BLOCK_BYTES = {Q4_0: 18, Q4_1: 20, Q8_0: 34}


def bf(x):
    """round to bf16 (round to nearest even) and back"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def decode_blocks(raw: np.ndarray, kind: int, narrow: bool = False) -> np.ndarray:
    """GGUF Q4_0 / Q4_1 / Q8_0 blocks -> flat f32. `narrow`: the block scale (and min) rounded to bf16
    first, as the NPU's q4_1 pool chunks hold them (pools.cpp std_perm_gguf, the 5120-B law)."""
    b = np.asarray(raw).reshape(-1, BLOCK_BYTES[kind])
    sc = (lambda a: bf(a)) if narrow else (lambda a: a)
    d = sc(b[:, 0:2].copy().view(np.float16).astype(np.float32))
    if kind == Q8_0:
        return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)
    if kind == Q4_1:
        m = sc(b[:, 2:4].copy().view(np.float16).astype(np.float32))
        qs = b[:, 4:]
    else:
        m = -8.0 * d
        qs = b[:, 2:]
    q = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.float32)   # value j | value j + 16
    return (q * d + m).reshape(-1)


class Gguf:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.meta, self.tensors, base = read_gguf(self.path)
        align = int(self.meta.get("general.alignment", 32))
        self.base = base + (align - base % align) % align if align != 32 else base
        self.mm = np.memmap(self.path, np.uint8, "r")

    def _bytes(self, kind: int, n: int) -> int:
        if kind == F32:
            return 4 * n
        if kind in (F16, BF16):
            return 2 * n
        if kind in BLOCK_BYTES:
            return n // 32 * BLOCK_BYTES[kind]
        raise ValueError(f"type {kind} not read here")

    def _decode(self, raw, kind: int, n: int, narrow: bool) -> np.ndarray:
        if kind == F32:
            return np.frombuffer(raw, np.float32, n)
        if kind == F16:
            return np.frombuffer(raw, np.float16, n).astype(np.float32)
        if kind == BF16:
            return (np.frombuffer(raw, np.uint16, n).astype(np.uint32) << 16).view(np.float32)
        return decode_blocks(raw, kind, narrow)

    def array(self, name: str, narrow: bool = False) -> np.ndarray:
        """A whole tensor as f32, row-major (dims reversed)."""
        t = self.tensors[name]
        shape = tuple(reversed(t["dims"]))
        n = int(np.prod(shape))
        o = self.base + t["offset"]
        k = t["gguf_type"]
        return self._decode(self.mm[o:o + self._bytes(k, n)], k, n, narrow).reshape(shape)

    def rows(self, name: str, rows) -> np.ndarray:
        """Rows of a [rows, cols] tensor as f32 (the embedding lookup)."""
        t = self.tensors[name]
        cols, k = t["dims"][0], t["gguf_type"]
        rb = self._bytes(k, cols)
        o = self.base + t["offset"]
        out = np.empty((len(rows), cols), np.float32)
        for i, r in enumerate(rows):
            out[i] = self._decode(self.mm[o + rb * r:o + rb * (r + 1)], k, cols, False)
        return out

    def expert(self, name: str, e: int, narrow: bool = False) -> np.ndarray:
        """Expert e of a [experts, rows, cols] tensor as f32 [rows, cols]."""
        t = self.tensors[name]
        cols, rows, _ = t["dims"]
        k = t["gguf_type"]
        per = self._bytes(k, rows * cols)
        o = self.base + t["offset"] + e * per
        return self._decode(self.mm[o:o + per], k, rows * cols, narrow).reshape(rows, cols)


def rope_inv_freq(cfg: dict) -> np.ndarray:
    """transformers' default rope: theta^(-2i/d) over the whole head (no scaling in Flint's config)."""
    hd = cfg["head_dim"]
    theta = float((cfg.get("rope_parameters") or {}).get("rope_theta", cfg.get("rope_theta")))
    return 1.0 / theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd)


def rope(x: np.ndarray, pos: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """Half-split rotation (rotate_half) per head: x [T, H, hd]."""
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    cos = np.cos(ang).astype(np.float32)[:, None, :]
    sin = np.sin(ang).astype(np.float32)[:, None, :]
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return (x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)).astype(np.float32) * w


def attention(q, k, v, sinks, window: int):
    """q [T, H, hd], k / v [T, KVH, hd]: causal; `window` rows (0 = all) -- query t sees keys
    (t - window, t], so 513 means itself and the 512 before it (HF's sliding mask
    `q - kv < window`, llama.cpp's LLAMA_SWA_TYPE_STANDARD `p1 - p0 >= n_swa` masked); the
    learned sinks join each head's softmax with no value vector."""
    T, H, hd = q.shape
    kvh = k.shape[1]
    g = H // kvh
    out = np.empty((T, H, hd), np.float32)
    scale = hd ** -0.5
    for t in range(T):
        s0 = max(0, t - window + 1) if window else 0
        kk, vv = k[s0:t + 1], v[s0:t + 1]                       # [L, KVH, hd]
        qh = q[t].reshape(kvh, g, hd)
        sc = np.einsum("kgd,lkd->kgl", qh.astype(np.float64), kk.astype(np.float64)) * scale
        sk = sinks.reshape(kvh, g, 1).astype(np.float64)
        m = np.maximum(sc.max(axis=-1, keepdims=True), sk)
        p = np.exp(sc - m)
        den = p.sum(axis=-1, keepdims=True) + np.exp(sk - m)
        out[t] = (np.einsum("kgl,lkd->kgd", p / den, vv.astype(np.float64))).reshape(H, hd)
    return out


def silu(x):
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def route(xm: np.ndarray, wr: np.ndarray, topk: int, norm: bool):
    """Flint's router: sigmoid scores of the F32 router, no bias; top k (torch.topk: the larger score
    first, ties to the lower expert as a stable sort gives); renormalised over the chosen."""
    logit = xm.astype(np.float64) @ wr.astype(np.float64).T
    s = 1.0 / (1.0 + np.exp(-logit))
    top = np.argsort(-s, axis=-1, kind="stable")[:, :topk]
    w = np.take_along_axis(s, top, axis=-1)
    if norm:
        w = w / w.sum(axis=-1, keepdims=True)
    return top, w.astype(np.float32)


def forward(g: Gguf, cfg: dict, ids: list[int], dump_layers=(), out: Path | None = None,
            positions=None, npu_like=False, layers=None):
    """npu_like: a control that rounds where the NPU route does (recipes/flint.py) -- the q4_1 block
    scales and mins to bf16, the norms' outputs (xn, xm, hn), K / V as cached, the attention output and
    the sinks / q_norm / k_norm weights to bf16 -- so a gap between the NPU and the fp32 reference can be
    told apart from those roundings."""
    rb = bf if npu_like else (lambda x: x)
    W = (lambda n: g.array(n, narrow=True)) if npu_like else g.array
    X = (lambda n, e: g.expert(n, e, narrow=True)) if npu_like else g.expert
    H, KVH, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    topk, norm_topk = cfg["num_experts_per_tok"], bool(cfg.get("norm_topk_prob", True))
    window = int(cfg["sliding_window"])
    types = cfg["layer_types"]
    dense = set(cfg.get("mlp_only_layers", []))
    nope = set(cfg.get("no_rope_layer_indices", []))
    inv = rope_inv_freq(cfg)
    T = len(ids)
    pos = np.arange(T)
    x = g.rows("token_embd.weight", ids)
    routes = {}
    t0 = time.time()
    n_layers = cfg["num_hidden_layers"] if layers is None else layers
    for l in range(n_layers):
        p = f"blk.{l}."
        sliding = types[l] == "sliding_attention"
        # HF rotates exactly the sliding layers; the config's no_rope list must say the same
        assert sliding == (l not in nope), f"layer {l}: sliding {sliding} but no_rope {l in nope}"
        h = rb(rms(x, g.array(p + "attn_norm.weight"), eps))
        q = (h @ W(p + "attn_q.weight").T).reshape(T, H, hd)
        k = (h @ W(p + "attn_k.weight").T).reshape(T, KVH, hd)
        v = (h @ W(p + "attn_v.weight").T).reshape(T, KVH, hd)
        q = rms(q, rb(g.array(p + "attn_q_norm.weight")), eps)
        k = rms(k, rb(g.array(p + "attn_k_norm.weight")), eps)
        if sliding:
            q, k = rope(q, pos, inv), rope(k, pos, inv)
        k, v = rb(k), rb(v)
        a = rb(attention(q, k, v, rb(g.array(p + "attn_sinks.weight")), window if sliding else 0))
        o = a.reshape(T, H * hd) @ W(p + "attn_output.weight").T
        res = x + o
        xm = rb(rms(res, g.array(p + "ffn_norm.weight"), eps))
        if l in dense:
            f = silu(xm @ W(p + "ffn_gate.weight").T) * (xm @ W(p + "ffn_up.weight").T)
            x = res + f @ W(p + "ffn_down.weight").T
        else:
            top, w = route(xm, g.array(p + "ffn_gate_inp.weight"), topk, norm_topk)
            moe = np.zeros_like(res)
            for e in np.unique(top):
                rows, slot = np.nonzero(top == e)
                xe = xm[rows]
                f = silu(xe @ X(p + "ffn_gate_exps.weight", int(e)).T) * (xe @ X(p + "ffn_up_exps.weight", int(e)).T)
                moe[rows] += w[rows, slot][:, None] * (f @ X(p + "ffn_down_exps.weight", int(e)).T)
            sf = silu(xm @ W(p + "ffn_gate_shexp.weight").T) * (xm @ W(p + "ffn_up_shexp.weight").T)
            x = res + moe + sf @ W(p + "ffn_down_shexp.weight").T
            routes[l] = {"experts": top.tolist(), "weights": w.tolist()}
        if out is not None and l in dump_layers:
            for nm, arr in (("res", res), ("xm", xm), ("out", x), ("q", q), ("k", k), ("v", v), ("a", a)):
                arr.astype(np.float32).tofile(out / f"layer{l}_{nm}.f32")
            if l in routes:
                (out / f"layer{l}_route.json").write_text(json.dumps(routes[l]))
        print(f"layer {l:2d} done {time.time() - t0:7.1f} s", file=sys.stderr, flush=True)
    hn = rb(rms(x, g.array("output_norm.weight"), eps))
    sel = list(range(T)) if positions is None else positions
    Wh = W("output.weight")
    logits = np.empty((len(sel), Wh.shape[0]), np.float32)
    for i0 in range(0, Wh.shape[0], 16384):
        logits[:, i0:i0 + 16384] = hn[sel] @ Wh[i0:i0 + 16384].T
    return logits, sel, routes, x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids-file", required=True, help="token ids, whitespace or comma separated")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dump-layers", default="")
    ap.add_argument("--positions", default="all", help="all | last | comma list")
    ap.add_argument("--npu-like", action="store_true", help="the rounding control (see forward)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()]
    a.out.mkdir(parents=True, exist_ok=True)
    pos = (None if a.positions == "all" else [len(ids) - 1] if a.positions == "last"
           else [int(p) for p in a.positions.split(",")])
    dump = {int(l) for l in a.dump_layers.split(",") if l.strip()}
    g = Gguf(a.gguf)
    t0 = time.time()
    logits, sel, routes, _ = forward(g, cfg, ids, dump, a.out, pos, a.npu_like)
    logits.tofile(a.out / "logits.f32")
    (a.out / "positions.json").write_text(json.dumps(sel))
    (a.out / "routes.json").write_text(json.dumps(routes))
    top = np.argsort(-logits, axis=-1)[:, :5]
    rep = {"gguf": a.gguf, "ids": ids, "positions": sel, "vocab": int(logits.shape[1]),
           "argmax": top[:, 0].tolist(), "top5": top.tolist(), "seconds": round(time.time() - t0, 1),
           "npu_like": a.npu_like}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("argmax", "seconds")}))


if __name__ == "__main__":
    main()
