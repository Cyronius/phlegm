"""CPU reference for dense Q4NX prompts: float64 attention/residuals, float32 projections.

    python utilities/prefill-ref-q4nx.py --model-dir <dir with model.q4nx + config.json> --ids ids.txt
        --dump "one-token=<prefix>" --dump "batched=<prefix>" [--baseline one-token] [--save ref.npy | --load ref.npy]

The whole prompt runs a layer at a time in float64 residuals (float32 matmuls) on the weights the NPU holds: the q4_1 chunks as
q4nx.py dequantizes them, norms in bf16, the embedding rows exact. Plain and sandwich (Gemma 3) norms, SiLU and GELU-tanh,
q/k norms, sliding windows and the local / global rope tables follow replica_dense.dense_decode. Each --dump is a per-position
logits dump (`open_qwen36_cli --prefill-logits --dump-logits <prefix>`). The gate against the one-token route (--baseline):
median per-position logits cosine >= 0.9999, argmax agreement within 1 point, prompt NLL within 0.5 %.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "open_kernels"))
sys.path.insert(0, str(REPO / "open_kernels" / "model"))
from q4nx import CHUNK_Q4, Q4NX, dq_chunks_q4_1, q4_1_chunks_of  # noqa: E402
from recipes.dense import QKNORM_POST_ROPE  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402
from recipes.spec import DENSE_LOCAL  # noqa: E402
from replica_dense import gelu_tanh, rms, rope, silu  # noqa: E402


def forward(model_dir: Path, ids: list[int]) -> np.ndarray:
    spec = spec_from_model_dir(model_dir)
    m = Q4NX(str(model_dir / "model.q4nx"))
    hid, nh, kvh, hd, ff = spec.hidden, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.intermediate
    T, eps, ng = len(ids), spec.norm_eps, spec.norm_groups
    act = gelu_tanh if spec.activation == "gelu_tanh" else silu
    x = np.stack([m.embed(t, hid) for t in ids])               # float64 [T, hid]
    pos = np.arange(T)
    post = spec.family in QKNORM_POST_ROPE
    for layer in range(spec.num_layers):
        pre = f"model.layers.{layer}."
        local = spec.layer_types[layer] == DENSE_LOCAL
        inv, rsc = spec.rope_inv_freq(local=local, ctx=T), spec.rope_scale()
        xn = (rms(x, eps, ng) * m.bf16(pre + "input_layernorm.weight")).astype(np.float32)
        q = (xn @ m.matmul_w(pre + "self_attn.q_proj.weight", nh * hd, hid).T).reshape(T, nh, hd).astype(np.float64)
        k = (xn @ m.matmul_w(pre + "self_attn.k_proj.weight", kvh * hd, hid).T).reshape(T, kvh, hd).astype(np.float64)
        v = (xn @ m.matmul_w(pre + "self_attn.v_proj.weight", kvh * hd, hid).T).reshape(T, kvh, hd).astype(np.float64)
        if spec.qk_norm:
            qn, kn = m.bf16(pre + "self_attn.q_norm.weight"), m.bf16(pre + "self_attn.k_norm.weight")
            q, k = rms(q, eps), rms(k, eps)
            if not post:
                q, k = q * qn, k * kn
        q = np.stack([rope(q[t], t, spec.rotary_dim, spec.rope_theta, inv, rsc) for t in range(T)])
        k = np.stack([rope(k[t], t, spec.rotary_dim, spec.rope_theta, inv, rsc) for t in range(T)])
        if spec.qk_norm and post:
            q, k = q * qn, k * kn
        grp = nh // kvh
        o = np.zeros((T, nh, hd))
        win = spec.sliding_window if (local and spec.sliding_window) else 0
        mask = pos[None, :] <= pos[:, None]
        if win:
            mask &= pos[None, :] > pos[:, None] - win
        for h in range(nh):
            s = (q[:, h] @ k[:, h // grp].T) / np.sqrt(hd)
            s = np.where(mask, s, -np.inf)
            a = np.exp(s - s.max(1, keepdims=True))
            o[:, h] = (a / a.sum(1, keepdims=True)) @ v[:, h // grp]
        out = o.reshape(T, nh * hd).astype(np.float32) @ m.matmul_w(pre + "self_attn.o_proj.weight", hid, nh * hd).T
        Wg = m.matmul_w(pre + "mlp.gate_proj.weight", ff, hid)
        Wu = m.matmul_w(pre + "mlp.up_proj.weight", ff, hid)
        Wd = m.matmul_w(pre + "mlp.down_proj.weight", hid, ff)
        if spec.sandwich_norms:
            res = x + (rms(out, eps, ng) * m.bf16(pre + "post_attention_layernorm.weight")).astype(np.float32)
            xm = (rms(res, eps, ng) * m.bf16(pre + "pre_feedforward_layernorm.weight")).astype(np.float32)
            d = (act(xm @ Wg.T) * (xm @ Wu.T)).astype(np.float32) @ Wd.T
            x = res + (rms(d, eps, ng) * m.bf16(pre + "post_feedforward_layernorm.weight")).astype(np.float32)
        else:
            res = x + out
            xm = (rms(res, eps, ng) * m.bf16(pre + "post_attention_layernorm.weight")).astype(np.float32)
            x = res + (act(xm @ Wg.T) * (xm @ Wu.T)).astype(np.float32) @ Wd.T
        print(f"layer {layer} done", flush=True)
    hn = (rms(x, eps, ng) * m.bf16("model.norm.weight")).astype(np.float32)
    raw = q4_1_chunks_of(m, "lm_head.weight")
    ncol = hid // 256
    vocab = raw.shape[0] // ncol * 32
    logits = np.zeros((T, vocab), np.float32)
    for r0 in range(0, vocab, 32 * 256):                       # 256 row blocks at a time
        nb = min(256, vocab // 32 - r0 // 32)
        w = dq_chunks_q4_1(raw[(r0 // 32) * ncol:(r0 // 32 + nb) * ncol]).reshape(nb, ncol, 32, 256)
        w = w.transpose(0, 2, 1, 3).reshape(nb * 32, hid)
        logits[:, r0:r0 + nb * 32] = hn @ w.T
    return logits


def load_dump(prefix: str, T: int, vocab: int) -> np.ndarray:
    return np.stack([np.fromfile(f"{prefix}_p{t}.bin", np.float32)[:vocab] for t in range(T)])


def score(ref: np.ndarray, got: np.ndarray, ids: list[int]) -> dict:
    r, g = ref.astype(np.float64), got.astype(np.float64)
    cos = (r * g).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(g, axis=1))
    lse = np.log(np.exp(g - g.max(1, keepdims=True)).sum(1)) + g.max(1)
    nll = float(np.mean([lse[t] - g[t, ids[t + 1]] for t in range(len(ids) - 1)]))
    return {"median": float(np.median(cos)), "min": float(cos.min()), "argmax": float((r.argmax(1) == g.argmax(1)).mean()), "nll": nll}


# Ported from local/llama-bfp16-prefill (1f51c69f); retain its independent CPU math.
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--dump", action="append", default=[], help="label=prefix of a --prefill-logits dump")
    ap.add_argument("--baseline")
    ap.add_argument("--save")
    ap.add_argument("--load")
    a = ap.parse_args()
    ids = [int(t) for t in Path(a.ids).read_text().split(",")]
    ref = np.load(a.load) if a.load else forward(a.model_dir, ids)
    if a.save and not a.load:
        np.save(a.save, ref)
    vocab = ref.shape[1]
    res = {}
    failed = False
    for d in a.dump:
        label, prefix = d.split("=", 1)
        got = load_dump(prefix, len(ids), vocab)
        if got.shape != ref.shape or not np.isfinite(got).all():
            raise ValueError("dump shape or finite-value check failed: " + label)
        # each route's own argmax is compared with the reference's
        r = score(ref, got, ids)
        res[label] = r
        print(f"{label}: {len(ids)} positions, logits cosine vs CPU median {r['median']:.6f} min {r['min']:.6f}, "
              f"argmax agrees {r['argmax'] * 100:.1f}%, prompt NLL {r['nll']:.4f}")
    if a.baseline:
        b = res[a.baseline]
        for label, r in res.items():
            if label == a.baseline:
                continue
            ok = (r["median"] >= 0.9999, r["argmax"] >= b["argmax"] - 0.01, r["nll"] <= b["nll"] * 1.005)
            failed = failed or not all(ok)
            print(f"gate {label} vs {a.baseline}: median cosine {'ok' if ok[0] else 'FAIL'}, argmax {'ok' if ok[1] else 'FAIL'}, "
                  f"NLL {'ok' if ok[2] else 'FAIL'} ({(r['nll'] / b['nll'] - 1) * 100:+.2f}%) -> {'PASS' if all(ok) else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
