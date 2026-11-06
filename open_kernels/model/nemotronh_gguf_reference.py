"""fp32 numpy forward of NVIDIA Nemotron-H MoE (Nemotron 3.5 Lightning 30B-A3B, model_type nemotron_h, GGUF arch
nemotron_h_moe) over its GGUF -- the independent reference the NPU route is checked against
(OPEN-FAMILY-NEMOTRONH).

It reads the GGUF directly (our pure Q4_0 with a Q8_0 head; Q4_1, Q8_0, F32, F16 and BF16 are read too) and
implements the published model -- transformers' models/nemotron_h/modeling_nemotron_h.py (5.22.1) and llama.cpp's
src/models/nemotron-h.cpp + llm_build_mamba_base::build_mamba2_layer (build 12015), which agree except for the dt
floor noted below -- not this tree's recipe: nothing is sliced, padded, regrouped or requantised.

  every block:  h = rms(x) * attn_norm;  x = x + mixer(h)        (52 blocks; layers_block_type picks the mixer)
  mamba (Mamba-2):
      [z | xBC | dt] = h W_in   (z 4096, xBC = x 4096 | B 8x128 | C 8x128, dt 64 heads)
      xBC = silu(causal depthwise conv(xBC; 4 taps, the LAST on this token, zero before t = 0) + conv bias)
      dt = softplus(dt + dt_bias);  dA = exp(dt * A)   (A = -exp(A_log), the GGUF's ssm_a)
      per head h (64 of 64 dims, group g = h // 8):  S_h = S_h dA_h + dt_h x_h B_g^T;  y_h = S_h C_g + D_h x_h
      y = y * silu(z);  y = grouped rms (8 groups of 512) * ssm_norm;  o = y W_out
  attention: q, k, v = h Wq, h Wk, h Wv (32 q / 2 kv heads of 128); NO positional encoding; causal;
      scale 1/sqrt(128);  o = a Wo
  moe:  s = sigmoid(h Wr) (F32 router, 128 experts); the top 6 of s + exp_probs_b; w = s of the chosen
      (unbiased) / their sum * routed_scaling_factor (2.5);
      o = sum_e w_e down_e(relu(up_e h)^2) + down_sh(relu(up_sh h)^2)      (non-gated experts, one shared)
  head:  logits = rms(x) * output_norm @ output^T   (untied)

HF clamps dt at time_step_min (1e-3) in its chunked prefill but not in its decode step; llama.cpp never clamps.
This follows llama.cpp unless --dt-floor. HF adds 1e-20 to the renormalising sum; llama.cpp clamps it at 6.1e-5
(a no-op for six sigmoids); neither changes a value here.

    python open_kernels/model/nemotronh_gguf_reference.py --gguf model.gguf --config config.json \
        --ids-file prompt.ids --out DIR [--dump-layers 0,1,5] [--positions all|last|N,M] [--npu-like]

writes `logits.f32` ([T, vocab] at the chosen positions), `positions.json`, `routes.json`, the per-layer states
asked for (`layer{l}_xn/out.f32` and the mixer's own: mamba `zxbcdt/y`, attention `q/k/v/a`, moe `moe`) and
`report.json`.
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
# the GGUF reader and the elementwise helpers (rms, bf16 rounding) are Flint's reference's; they hold no model
# logic. hyv3's attention is plain causal GQA attention with double accumulation (no RoPE inside it).
from flint_gguf_reference import Gguf, bf, rms  # noqa: E402
from hyv3_gguf_reference import attention  # noqa: E402


def block_types(cfg: dict) -> list[str]:
    """mamba | attention | moe per block (the published config's `layers_block_type`; transformers 5 also
    accepts linear_attention / full_attention)."""
    t = cfg.get("layers_block_type") or cfg.get("layer_types")
    alias = {"linear_attention": "mamba", "full_attention": "attention", "M": "mamba", "*": "attention", "E": "moe"}
    return [alias.get(x, x) for x in t]


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def silu(x):
    return x / (1.0 + np.exp(-x))


def route(h: np.ndarray, wr: np.ndarray, bias: np.ndarray, topk: int, scale: float):
    """NemotronHTopkRouter (n_group 1) / llama.cpp build_moe_ffn (sigmoid, exp_probs_b, norm_w, scale): the top k
    of sigmoid(logits) + bias (ties to the lower expert, a stable sort); weights the unbiased sigmoids of the
    chosen, renormalised, times `scale`."""
    logit = h.astype(np.float64) @ wr.astype(np.float64).T
    s = 1.0 / (1.0 + np.exp(-logit))
    top = np.argsort(-(s + bias.astype(np.float64)), axis=-1, kind="stable")[:, :topk]
    w = np.take_along_axis(s, top, axis=-1)
    w = w / w.sum(axis=-1, keepdims=True) * scale
    return top, w.astype(np.float32)


def relu2(x):
    r = np.maximum(x, 0.0)
    return r * r


def mamba2(h, g: Gguf, p: str, cfg: dict, W, rb, eps: float, dt_floor: float, dump=None, state=None):
    """One Mamba-2 mixer over a whole sequence. `state` (conv [3, 6144], ssm [64, 64, 128]) is the carried
    recurrent state (zero when None); it is updated in place when given."""
    T = h.shape[0]
    nh, hdim = cfg["mamba_num_heads"], cfg["mamba_head_dim"]
    ng, ds, kconv = cfg["n_groups"], cfg["ssm_state_size"], cfg["conv_kernel"]
    di = nh * hdim
    cdim = di + 2 * ng * ds
    zxbcdt = (h @ W(p + "ssm_in.weight").T).astype(np.float32)
    z, xbc, dt = zxbcdt[:, :di], zxbcdt[:, di:di + cdim], zxbcdt[:, di + cdim:]
    wc = g.array(p + "ssm_conv1d.weight").reshape(cdim, kconv).astype(np.float64)    # [channels, taps]
    bc = g.array(p + "ssm_conv1d.bias").astype(np.float64)
    prev = np.zeros((kconv - 1, cdim), np.float64) if state is None else state["conv"].astype(np.float64)
    full = np.concatenate([prev, xbc.astype(np.float64)], axis=0)       # [k - 1 + T, channels]
    conv = np.zeros((T, cdim), np.float64)
    for k in range(kconv):                     # tap k pairs with token t - (kconv - 1 - k)
        conv += full[k:k + T] * wc[:, k]
    xbc_c = silu(conv + bc)
    xs = xbc_c[:, :di].reshape(T, nh, hdim)
    B = xbc_c[:, di:di + ng * ds].reshape(T, ng, ds)
    C = xbc_c[:, di + ng * ds:].reshape(T, ng, ds)
    A = g.array(p + "ssm_a").reshape(nh).astype(np.float64)
    D = g.array(p + "ssm_d").reshape(nh).astype(np.float64)
    dtv = softplus(dt.astype(np.float64) + g.array(p + "ssm_dt.bias").astype(np.float64))
    if dt_floor:
        dtv = np.maximum(dtv, dt_floor)
    S = np.zeros((nh, hdim, ds), np.float64) if state is None else state["ssm"].astype(np.float64)
    grp = np.arange(nh) // (nh // ng)
    y = np.empty((T, nh, hdim), np.float64)
    for t in range(T):
        dA = np.exp(dtv[t] * A)                                        # [nh]
        Bt, Ct = B[t][grp], C[t][grp]                                  # [nh, ds]
        S = S * dA[:, None, None] + (dtv[t][:, None] * xs[t])[:, :, None] * Bt[:, None, :]
        y[t] = np.einsum("hpn,hn->hp", S, Ct) + D[:, None] * xs[t]
    y = y.reshape(T, di) * silu(z.astype(np.float64))
    yg = y.reshape(T, ng, di // ng)
    yn = yg / np.sqrt(np.mean(yg ** 2, axis=-1, keepdims=True) + eps)
    yn = rb((yn.reshape(T, di) * g.array(p + "ssm_norm.weight").reshape(di)).astype(np.float32))
    if state is not None:
        state["conv"] = full[-(kconv - 1):].astype(np.float32)
        state["ssm"] = S
    if dump is not None:
        dump.update({"zxbcdt": zxbcdt, "y": yn})
    return yn @ W(p + "ssm_out.weight").T


def forward(g: Gguf, cfg: dict, ids: list[int], dump_layers=(), out: Path | None = None,
            positions=None, npu_like=False, layers=None, dt_floor=0.0):
    """npu_like: a control that rounds where the NPU route does (recipes/nemotronh.py): the GEMV activations
    (every block's normed input h, the mamba gated-norm output y, the attention output a, an expert's
    relu^2 output) to bf16, and the q4 block scales to bf16 (a no-op for a BF16-sourced pure Q4_0). The head
    keeps its Q8_0 fp16 scales (f32 on the NPU, exact)."""
    rb = bf if npu_like else (lambda x: x)
    W = (lambda n: g.array(n, narrow=True)) if npu_like else g.array
    X = (lambda n, e: g.expert(n, e, narrow=True)) if npu_like else g.expert
    hid = cfg["hidden_size"]
    H, KVH, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = float(cfg.get("layer_norm_epsilon", cfg.get("norm_eps", 1e-5)))
    topk = cfg["num_experts_per_tok"]
    scale = float(cfg.get("routed_scaling_factor", 1.0))
    types = block_types(cfg)
    T = len(ids)
    x = g.rows("token_embd.weight", ids)
    routes = {}
    t0 = time.time()
    n_layers = len(types) if layers is None else layers
    for l in range(n_layers):
        p = f"blk.{l}."
        d: dict = {}
        h = rb(rms(x, g.array(p + "attn_norm.weight"), eps))
        if types[l] == "mamba":
            o = mamba2(h, g, p, cfg, W, rb, eps, dt_floor, d)
        elif types[l] == "attention":
            q = (h @ W(p + "attn_q.weight").T).reshape(T, H, hd)
            k = (h @ W(p + "attn_k.weight").T).reshape(T, KVH, hd)
            v = (h @ W(p + "attn_v.weight").T).reshape(T, KVH, hd)
            a = rb(attention(q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)))
            o = a.reshape(T, H * hd) @ W(p + "attn_output.weight").T
            d.update({"q": q, "k": k, "v": v, "a": a})
        elif types[l] == "moe":
            top, w = route(h, g.array(p + "ffn_gate_inp.weight"), g.array(p + "exp_probs_b.bias"), topk, scale)
            moe = np.zeros_like(x, dtype=np.float64)
            for e in np.unique(top):
                rows, slot = np.nonzero(top == e)
                f = rb(relu2(h[rows] @ X(p + "ffn_up_exps.weight", int(e)).T).astype(np.float32))
                moe[rows] += w[rows, slot][:, None] * (f @ X(p + "ffn_down_exps.weight", int(e)).T)
            fs = rb(relu2(h @ W(p + "ffn_up_shexp.weight").T).astype(np.float32))
            moe += fs @ W(p + "ffn_down_shexp.weight").T
            o = moe
            routes[l] = {"experts": top.tolist(), "weights": w.tolist()}
            d["moe"] = moe
        else:
            raise ValueError(f"layer {l}: block type {types[l]!r}")
        x = (x + o).astype(np.float32)
        if out is not None and l in dump_layers:
            for nm, arr in [("xn", h), ("out", x)] + list(d.items()):
                np.asarray(arr, np.float32).tofile(out / f"layer{l}_{nm}.f32")
            if l in routes:
                (out / f"layer{l}_route.json").write_text(json.dumps(routes[l]))
        print(f"layer {l:2d} ({types[l]}) done {time.time() - t0:7.1f} s", file=sys.stderr, flush=True)
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
    ap.add_argument("--layers", type=int, default=None, help="run only the first N layers (bring-up)")
    ap.add_argument("--positions", default="all", help="all | last | comma list")
    ap.add_argument("--npu-like", action="store_true", help="the rounding control (see forward)")
    ap.add_argument("--dt-floor", type=float, default=0.0, help="HF prefill's dt clamp (time_step_min); 0: none")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()]
    a.out.mkdir(parents=True, exist_ok=True)
    pos = (None if a.positions == "all" else [len(ids) - 1] if a.positions == "last"
           else [int(p) for p in a.positions.split(",")])
    dump = {int(l) for l in a.dump_layers.split(",") if l.strip()}
    g = Gguf(a.gguf)
    t0 = time.time()
    logits, sel, routes, _ = forward(g, cfg, ids, dump, a.out, pos, a.npu_like, a.layers, a.dt_floor)
    logits.tofile(a.out / "logits.f32")
    (a.out / "positions.json").write_text(json.dumps(sel))
    (a.out / "routes.json").write_text(json.dumps(routes))
    top = np.argsort(-logits, axis=-1)[:, :5]
    rep = {"gguf": a.gguf, "ids": ids, "positions": sel, "vocab": int(logits.shape[1]),
           "argmax": top[:, 0].tolist(), "top5": top.tolist(), "seconds": round(time.time() - t0, 1),
           "npu_like": a.npu_like, "dt_floor": a.dt_floor}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("argmax", "seconds")}))


if __name__ == "__main__":
    main()
