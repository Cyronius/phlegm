"""fp32 numpy forward of Tencent HY V3 (Hy-Memory-30B-A3B) over its GGUF -- the independent reference the
NPU route is checked against (OPEN-FAMILY-HYV3).

It reads the GGUF directly (our pure Q4_0 with a Q8_0 head; Q4_1, Q8_0, F32, F16 and BF16 are read too)
and implements the published model -- transformers' modular_hy_v3.py and llama.cpp's src/models/hy-v3.cpp,
which agree -- not this tree's recipe: nothing is sliced, paired or requantised.

  per layer:  h = rms(x) * attn_norm
              q, k, v = h Wq, h Wk, h Wv;  q, k = rms over each head * q_norm / k_norm  (Qwen3)
              NEOX RoPE (half split, theta 11158840, every dim), every layer; causal attention
              res = x + a Wo;  xm = rms(res) * ffn_norm
              dense layers (mlp_layer_types "dense"): x = res + down(silu(gate xm) * up xm)
              MoE layers: s = sigmoid(xm Wr) (F32 router); the top k of s + b (b = exp_probs_b, the
                          selection bias); w = s of the chosen (unbiased) / their sum * scale
                          x = res + sum_e w_e down_e(silu(gate_e xm) * up_e xm) + shared(xm)
              (the shared expert is ungated, always added in full)
  head:       logits = rms(x) * output_norm @ output

The expert width is read off the tensors (768 for Hy-Memory, whose config says 1536). The forward is
layer-major over the whole sequence:

    python open_kernels/model/hyv3_gguf_reference.py --gguf model.gguf --config config.json \
        --ids-file prompt.ids --out DIR [--dump-layers 0,1,47] [--positions all|last|N,M] [--npu-like]

writes `logits.f32` ([T, vocab] at the chosen positions), `positions.json`, `routes.json`, the per-layer
states asked for (`layer{l}_res/xm/out/q/k/v/a.f32`) and `report.json`.
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
sys.path.insert(0, str(HERE))
# the GGUF reader and the elementwise helpers (rms, rotate_half RoPE, silu, bf16 rounding) are Flint's
# reference's; they hold no model logic
from flint_gguf_reference import Gguf, bf, rms, rope, silu  # noqa: E402


def rope_inv_freq(cfg: dict) -> np.ndarray:
    """transformers' default rope: theta^(-2i/d) over the whole head (no scaling)."""
    hd = cfg["head_dim"]
    theta = float((cfg.get("rope_parameters") or {}).get("rope_theta", cfg.get("rope_theta")))
    return 1.0 / theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd)


def attention(q, k, v):
    """q [T, H, hd], k / v [T, KVH, hd]: causal, every row; double accumulation."""
    T, H, hd = q.shape
    kvh = k.shape[1]
    g = H // kvh
    out = np.empty((T, H, hd), np.float32)
    scale = hd ** -0.5
    k64, v64 = k.astype(np.float64), v.astype(np.float64)
    for t in range(T):
        qh = q[t].reshape(kvh, g, hd).astype(np.float64)
        sc = np.einsum("kgd,lkd->kgl", qh, k64[:t + 1]) * scale
        p = np.exp(sc - sc.max(axis=-1, keepdims=True))
        out[t] = (np.einsum("kgl,lkd->kgd", p / p.sum(axis=-1, keepdims=True), v64[:t + 1])).reshape(H, hd)
    return out


def route(xm: np.ndarray, wr: np.ndarray, bias: np.ndarray, topk: int, scale: float):
    """HY V3's router (HYV3TopKRouter; llama.cpp build_moe_ffn sigmoid + exp_probs_b + norm + scale):
    s = sigmoid(xm Wr); the top k of s + bias (the larger first, ties to the lower expert as a stable sort
    gives); the weights are the UNBIASED s of the chosen, renormalised, times `scale`."""
    logit = xm.astype(np.float64) @ wr.astype(np.float64).T
    s = 1.0 / (1.0 + np.exp(-logit))
    top = np.argsort(-(s + bias.astype(np.float64)), axis=-1, kind="stable")[:, :topk]
    w = np.take_along_axis(s, top, axis=-1)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-20) * scale
    return top, w.astype(np.float32)


def mlp_types(cfg: dict) -> list[str]:
    n = cfg["num_hidden_layers"]
    t = cfg.get("mlp_layer_types")
    if not t:
        k = int(cfg.get("first_k_dense_replace", 1))
        t = ["dense"] * k + ["sparse"] * (n - k)
    return list(t)


def forward(g: Gguf, cfg: dict, ids: list[int], dump_layers=(), out: Path | None = None,
            positions=None, npu_like=False, layers=None):
    """npu_like: a control that rounds where the NPU route does (recipes/hyv3.py) -- the q4 block scales
    to bf16 (a no-op for a BF16-sourced pure Q4_0), the norms' outputs (xn, xm, hn), K / V as cached, the
    attention output, the q_norm / k_norm weights and each slot's h (the down GEMV's f32 table input is
    rounded to bf16 by gemv_q4_prep_f32) to bf16 -- so a gap between the NPU and the fp32 reference can be
    told apart from those roundings. The head keeps its Q8_0 fp16 scales (f32 on the NPU, exact)."""
    rb = bf if npu_like else (lambda x: x)
    W = (lambda n: g.array(n, narrow=True)) if npu_like else g.array
    X = (lambda n, e: g.expert(n, e, narrow=True)) if npu_like else g.expert
    H, KVH, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    topk = cfg["num_experts_per_tok"]
    scale = float(cfg.get("router_scaling_factor", 1.0))
    kinds = mlp_types(cfg)
    inv = rope_inv_freq(cfg)
    T = len(ids)
    pos = np.arange(T)
    x = g.rows("token_embd.weight", ids)
    routes = {}
    t0 = time.time()
    n_layers = cfg["num_hidden_layers"] if layers is None else layers
    for l in range(n_layers):
        p = f"blk.{l}."
        h = rb(rms(x, g.array(p + "attn_norm.weight"), eps))
        q = (h @ W(p + "attn_q.weight").T).reshape(T, H, hd)
        k = (h @ W(p + "attn_k.weight").T).reshape(T, KVH, hd)
        v = (h @ W(p + "attn_v.weight").T).reshape(T, KVH, hd)
        q = rope(rms(q, rb(g.array(p + "attn_q_norm.weight")), eps), pos, inv)
        k = rope(rms(k, rb(g.array(p + "attn_k_norm.weight")), eps), pos, inv)
        k, v = rb(k), rb(v)
        a = rb(attention(q, k, v))
        o = a.reshape(T, H * hd) @ W(p + "attn_output.weight").T
        res = x + o
        xm = rb(rms(res, g.array(p + "ffn_norm.weight"), eps))
        if kinds[l] == "dense":
            f = silu(xm @ W(p + "ffn_gate.weight").T) * (xm @ W(p + "ffn_up.weight").T)
            x = res + rb(f) @ W(p + "ffn_down.weight").T
        else:
            top, w = route(xm, g.array(p + "ffn_gate_inp.weight"), g.array(p + "exp_probs_b"), topk, scale)
            moe = np.zeros_like(res, dtype=np.float64)
            for e in np.unique(top):
                rows, slot = np.nonzero(top == e)
                xe = xm[rows]
                f = silu(xe @ X(p + "ffn_gate_exps.weight", int(e)).T) * (xe @ X(p + "ffn_up_exps.weight", int(e)).T)
                moe[rows] += rb(w[rows, slot][:, None] * f) @ X(p + "ffn_down_exps.weight", int(e)).T
            sf = silu(xm @ W(p + "ffn_gate_shexp.weight").T) * (xm @ W(p + "ffn_up_shexp.weight").T)
            x = (res + moe + rb(sf) @ W(p + "ffn_down_shexp.weight").T).astype(np.float32)
            routes[l] = {"experts": top.tolist(), "weights": w.tolist()}
        if out is not None and l in dump_layers:
            for nm, arr in (("res", res), ("xm", xm), ("out", x), ("q", q), ("k", k), ("v", v), ("a", a)):
                arr.astype(np.float32).tofile(out / f"layer{l}_{nm}.f32")
            if l in routes:
                (out / f"layer{l}_route.json").write_text(json.dumps(routes[l]))
        print(f"layer {l:2d} done {time.time() - t0:7.1f} s", file=sys.stderr, flush=True)
    hn = rb(rms(x, g.array("output_norm.weight"), eps))
    sel = list(range(T)) if positions is None else positions
    Wh = g.array("output.weight")
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
