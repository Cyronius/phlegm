r"""The fp64 reference for a llama.cpp `xing4_0` GGUF (Xing4.0-29B-A4B, OPEN-FAMILY-XING): the published
Xing4_0ForCausalLM forward (the Hub's modeling_xing4_0.py @ ac0fc307) over the GGUF's own values.

Every tensor is dequantized by llama.cpp's gguf-py (`gguf.quants.dequantize`), the arithmetic definition of the
file's weights. The attention is computed in the HF (NON-absorbed) form: the 512-wide latent is expanded per head
into 128 nope-K and 128 V values through the converter's split kv_b (attn_k_b [32][512][128], attn_v_b
[32][128][512]); the runtime computes the absorbed form (q through W_UK into the latent, the output through
W_UV), so the two agree only if the absorption is right. RoPE turns the pairs (2i, 2i + 1) of the 64 rope dims
(HF's rope_interleave; HF writes the turned halves de-interleaved, a permutation q and k share, so q . k is the
same), at YaRN's inverse frequencies (HF _compute_yarn_parameters, truncate), cos / sin unscaled (attention_factor
mscale / mscale_all_dim = 1) and the score scale 192^-0.5 * (0.1 ln(factor) mscale_all_dim + 1)^2. Nothing here
shares code with the runtime, the recipe or llama.cpp's compute path. Activations are fp64; weights are held as
f32 (gguf-py's output) and widened per product.

  residual:   4 streams S [4, hidden], every one the token's embedding at the start
  mHC block:  f = S flattened (stream-major) / rms;  m = f hc_fn^T  [24]
              pre = sigmoid(m[0:4] s0 + b[0:4]); post = 2 sigmoid(m[4:8] s1 + b[4:8])
              C = Sinkhorn(exp(clamp(m[8:24] s2 + b[8:24], +-30) - rowmax)), 20 x (rows / (sum + eps), then cols)
              y = sublayer(rms(sum_i pre_i S_i) * norm);  S_i <- post_i y + sum_j C_ij S_j
  attention:  qa = rms(h Wqa^T) * q_a_norm; q = qa Wqb^T -> [32][128 nope | 64 rope]
              kv = h Wkva^T -> c [512] | kr [64]; c = rms(c) * kv_a_norm
              k_h = [c k_b[h] | rope(kr)], v_h = v_b[h] c; a = softmax(q k^T * scale, causal) v; o = a Wo^T
  FFN:        layers < first_k_dense: down(silu(gate h) * up h); else sigmoid router over 64 experts, the top 4 of
              s + exp_probs_b, weights the unbiased s renormalised x 2.0, plus the shared expert at weight 1
  head:       logits = rms(mean_i S_i) * output_norm @ output^T

Writes, for the first N layers (all of them by default):

    <out>/ref_logits_p{pos}.bin    f32[vocab]          the N-layer model's logits at each requested position
    <out>/ref_res{l}.bin           f32[T, 4, hidden]   layer l's output streams at every position (--dump-res)

    python -I open_kernels/model/xing_gguf_reference.py --gguf model.gguf --ids-file short.ids --out OUT \
        [--layers 8] [--positions all|last|a,b,...] [--dump-res] [--npu-like] [--gguf-py <llama.cpp>/gguf-py]

--npu-like is a control, not a reference: the runtime's arithmetic (the absorbed attention, bf16 GEMV inputs, f32
GEMV outputs / streams / KV rows, the dense MLP as 1024-wide slots, h rounded to bf16 before each down part) over
the same file, to measure what the NPU route's roundings cost against the fp64 forward.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ARCH = "xing4_0"


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

    def meta(self, key: str, default=None):
        if key not in self.kv:
            if default is not None:
                return default
            raise KeyError(key)
        f = self.kv[key]
        v = f.parts[f.data[0]]
        return v.tobytes().decode() if f.types and f.types[0].name == "STRING" else v[0].item()

    def shape(self, name: str) -> list[int]:
        """numpy order: the GGUF dims reversed (its fastest dim last)."""
        return [int(d) for d in reversed(self.t[name].shape)]

    def w(self, name: str) -> np.ndarray:
        t = self.t[name]
        a = self.dq(t.data, t.tensor_type).astype(np.float32, copy=False)
        return a.reshape(self.shape(name))

    def expert(self, name: str, e: int) -> np.ndarray:
        """Expert e of a [E, rows, cols] tensor as f32 [rows, cols], dequantizing only its bytes."""
        t = self.t[name]
        E, rows, cols = self.shape(name)
        data = t.data.reshape(E, rows, -1)
        return self.dq(data[e], t.tensor_type).reshape(rows, cols).astype(np.float32, copy=False)

    def rows(self, name: str, ids) -> np.ndarray:
        t = self.t[name]
        cols = int(t.shape[0])
        data = t.data.reshape(int(t.shape[1]), -1)
        return np.stack([self.dq(data[i], t.tensor_type).reshape(-1)[:cols] for i in ids]).astype(np.float64)


def bf(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 (round to nearest even) -> fp64: what a GEMV's prep makes of its f32 / bf16 activation."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def rms(x: np.ndarray, w, eps: float) -> np.ndarray:
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x / (1.0 + np.exp(-x))


def yarn_inv_freq(rot: int, theta: float, factor: float, orig: float, beta_fast: float, beta_slow: float) -> np.ndarray:
    """HF _compute_yarn_parameters (truncate=True): the pair i < rot/2 frequency, interpolated / extrapolated."""
    def corr(r):
        return rot * math.log(orig / (r * 2 * math.pi)) / (2 * math.log(theta))
    low, high = math.floor(corr(beta_fast)), math.ceil(corr(beta_slow))
    low, high = max(low, 0), min(high, rot - 1)
    if low == high:
        high += 0.001
    pos_freqs = theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot)
    extrap, interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    ramp = np.clip((np.arange(rot // 2, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
    ext_factor = 1.0 - ramp
    return interp * (1.0 - ext_factor) + extrap * ext_factor


def rope_pairs(x: np.ndarray, pos: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """x [T, heads, rot]: pairs (2i, 2i + 1) turned by pos * inv[i]."""
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def hc_pre(S: np.ndarray, fn: np.ndarray, base: np.ndarray, scale: np.ndarray, eps: float, iters: int,
           hc_eps: float, clamp: float = 30.0):
    """S [T, hc, hidden] -> (collapsed [T, hidden], post [T, hc], comb [T, hc, hc]) (Xing4_0HyperConnection)."""
    T, hc, hid = S.shape
    flat = rms(S.reshape(T, hc * hid), None, eps)
    m = flat @ fn.T.astype(np.float64)                               # [T, (2 + hc) hc]
    pre = sigmoid(m[:, :hc] * scale[0] + base[:hc])
    post = 2.0 * sigmoid(m[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    cl = (m[:, 2 * hc:] * scale[2] + base[2 * hc:]).reshape(T, hc, hc)
    cl = np.clip(cl, -clamp, clamp)
    comb = np.exp(cl - cl.max(axis=-1, keepdims=True))
    for _ in range(iters):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + hc_eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + hc_eps)
    collapsed = np.einsum("th,thd->td", pre, S)
    return collapsed, post, comb


def hc_post(y: np.ndarray, S: np.ndarray, post: np.ndarray, comb: np.ndarray) -> np.ndarray:
    return post[:, :, None] * y[:, None, :] + np.einsum("tij,tjd->tid", comb, S)


def mla(g: Gguf, p: str, h: np.ndarray, pos: np.ndarray, cfg: dict) -> np.ndarray:
    T = h.shape[0]
    nh, nope, rot, vd, kvr = cfg["heads"], cfg["nope"], cfg["rope"], cfg["vdim"], cfg["kv_lora"]
    eps = cfg["eps"]
    qa = rms(h @ g.w(p + "attn_q_a.weight").T, g.w(p + "attn_q_a_norm.weight").astype(np.float64), eps)
    q = (qa @ g.w(p + "attn_q_b.weight").T).reshape(T, nh, nope + rot)
    kv = h @ g.w(p + "attn_kv_a_mqa.weight").T                       # [T, kv_lora + rope]
    c = rms(kv[:, :kvr], g.w(p + "attn_kv_a_norm.weight").astype(np.float64), eps)
    kr = rope_pairs(kv[:, None, kvr:], pos, cfg["inv"])[:, 0, :]     # [T, rope], shared by every head
    qr = rope_pairs(q[:, :, nope:], pos, cfg["inv"])
    kb = g.w(p + "attn_k_b.weight").astype(np.float64)               # [heads, kv_lora, nope]
    vb = g.w(p + "attn_v_b.weight").astype(np.float64)               # [heads, vdim, kv_lora]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    out = np.empty((T, nh, vd))
    for hh in range(nh):
        kn = c @ kb[hh]                                              # [T, nope]
        v = c @ vb[hh].T                                             # [T, vdim]
        s = (q[:, hh, :nope] @ kn.T + qr[:, hh, :] @ kr.T) * cfg["scale"] + mask
        s -= s.max(axis=1, keepdims=True)
        pr = np.exp(s)
        pr /= pr.sum(axis=1, keepdims=True)
        out[:, hh, :] = pr @ v
    return out.reshape(T, nh * vd) @ g.w(p + "attn_output.weight").T


def mla_absorbed(g: Gguf, p: str, h: np.ndarray, pos: np.ndarray, cfg: dict) -> np.ndarray:
    """--npu-like: the runtime's form (xing-runtime.patch xg_qnorm / xg_mla): the GEMV inputs rounded to bf16 (h,
    the normed q_a, the attention output), the GEMV outputs and the KV rows [latent | rope key] f32, q_nope absorbed
    through k_b into the latent, the output through v_b."""
    T = h.shape[0]
    nh, nope, rot, vd, kvl = cfg["heads"], cfg["nope"], cfg["rope"], cfg["vdim"], cfg["kv_lora"]
    eps = cfg["eps"]
    hb = bf(h)
    qkva = f32(hb @ np.concatenate([g.w(p + "attn_q_a.weight"), g.w(p + "attn_kv_a_mqa.weight")]).T)
    ql = qkva.shape[1] - kvl - rot
    qa = f32(rms(qkva[:, :ql], g.w(p + "attn_q_a_norm.weight").astype(np.float64), eps))
    c = f32(rms(qkva[:, ql:ql + kvl], g.w(p + "attn_kv_a_norm.weight").astype(np.float64), eps))
    kr = f32(rope_pairs(qkva[:, None, ql + kvl:], pos, cfg["inv"])[:, 0, :])
    q = f32(bf(qa) @ g.w(p + "attn_q_b.weight").T).reshape(T, nh, nope + rot)
    qr = f32(rope_pairs(q[:, :, nope:], pos, cfg["inv"]))
    kb = g.w(p + "attn_k_b.weight").astype(np.float64)               # [heads, kv_lora, nope]
    vb = g.w(p + "attn_v_b.weight").astype(np.float64)               # [heads, vdim, kv_lora]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    out = np.empty((T, nh, vd))
    for hh in range(nh):
        lat = q[:, hh, :nope] @ kb[hh].T                             # [T, kv_lora]
        s = (lat @ c.T + qr[:, hh, :] @ kr.T) * cfg["scale"] + mask
        s -= s.max(axis=1, keepdims=True)
        pr = np.exp(s)
        pr /= pr.sum(axis=1, keepdims=True)
        out[:, hh, :] = (pr @ c) @ vb[hh].T
    return f32(bf(f32(out.reshape(T, nh * vd))) @ g.w(p + "attn_output.weight").T)


def moe(g: Gguf, p: str, m: np.ndarray, cfg: dict, npu: bool = False) -> np.ndarray:
    T, hid = m.shape
    E, K, scale = cfg["experts"], cfg["topk"], cfg["router_scale"]
    s = sigmoid(m @ g.w(p + "ffn_gate_inp.weight").T.astype(np.float64))      # [T, E]
    choice = s + g.w(p + "exp_probs_b.bias").astype(np.float64)[None, :]
    idx = np.argsort(-choice, axis=1, kind="stable")[:, :K]
    wts = np.take_along_axis(s, idx, axis=1)
    wts = wts / (wts.sum(axis=1, keepdims=True) + 1e-20) * scale
    out = np.zeros((T, hid))
    if npu:                                         # xm bf16, weights f32, h f32 rounded to bf16 for down
        m, wts = bf(m), f32(wts)
    r = bf if npu else (lambda v: v)
    for e in np.unique(idx):
        rows, slot = np.nonzero(idx == e)
        x = m[rows]
        gt = x @ g.expert(p + "ffn_gate_exps.weight", int(e)).T
        up = x @ g.expert(p + "ffn_up_exps.weight", int(e)).T
        y = r(silu(gt) * up * (wts[rows, slot][:, None] if npu else 1.0)) @ g.expert(p + "ffn_down_exps.weight", int(e)).T
        out[rows] += y if npu else y * wts[rows, slot][:, None]
    gt = m @ g.w(p + "ffn_gate_shexp.weight").T
    up = m @ g.w(p + "ffn_up_shexp.weight").T
    return out + r(silu(gt) * up) @ g.w(p + "ffn_down_shexp.weight").T


def dense(g: Gguf, p: str, m: np.ndarray, npu: bool = False, ff: int = 1024) -> np.ndarray:
    if not npu:
        gt = m @ g.w(p + "ffn_gate.weight").T
        up = m @ g.w(p + "ffn_up.weight").T
        return (silu(gt) * up) @ g.w(p + "ffn_down.weight").T
    # --npu-like: the 1024-wide slices the runtime runs as slots, each h rounded to bf16 for its down part
    m = bf(m)
    G, U, D = g.w(p + "ffn_gate.weight"), g.w(p + "ffn_up.weight"), g.w(p + "ffn_down.weight")
    out = np.zeros((m.shape[0], D.shape[0]))
    for j in range(0, G.shape[0], ff):
        out += bf(silu(m @ G[j:j + ff].T) * (m @ U[j:j + ff].T)) @ D[:, j:j + ff].T
    return out


def config(g: Gguf) -> dict:
    a = ARCH
    arch = g.meta("general.architecture")
    if arch != a:
        raise SystemExit(f"architecture {arch!r}; this reference is the {a} forward")
    rot = int(g.meta(f"{a}.rope.dimension_count"))
    k_mla = int(g.meta(f"{a}.attention.key_length_mla"))
    factor = float(g.meta(f"{a}.rope.scaling.factor"))
    log_mul = float(g.meta(f"{a}.rope.scaling.yarn_log_multiplier", 0.0)) / 0.1     # converter writes 0.1 mscale_all_dim
    mscale = 0.1 * log_mul * math.log(factor) + 1.0 if factor > 1 else 1.0
    theta = float(g.meta(f"{a}.rope.freq_base"))
    orig = float(g.meta(f"{a}.rope.scaling.original_context_length"))
    bf = float(g.meta(f"{a}.rope.scaling.yarn_beta_fast", 32.0))
    bs = float(g.meta(f"{a}.rope.scaling.yarn_beta_slow", 1.0))
    return {
        "layers": int(g.meta(f"{a}.block_count")), "hidden": int(g.meta(f"{a}.embedding_length")),
        "heads": int(g.meta(f"{a}.attention.head_count")), "eps": float(g.meta(f"{a}.attention.layer_norm_rms_epsilon")),
        "rope": rot, "nope": k_mla - rot, "vdim": int(g.meta(f"{a}.attention.value_length_mla")),
        "kv_lora": int(g.meta(f"{a}.attention.kv_lora_rank")), "q_lora": int(g.meta(f"{a}.attention.q_lora_rank")),
        "dense_lead": int(g.meta(f"{a}.leading_dense_block_count")), "experts": int(g.meta(f"{a}.expert_count")),
        "topk": int(g.meta(f"{a}.expert_used_count")), "router_scale": float(g.meta(f"{a}.expert_weights_scale")),
        "hc": int(g.meta(f"{a}.hyper_connection.count", 4)),
        "hc_iters": int(g.meta(f"{a}.hyper_connection.sinkhorn_iterations", 20)),
        "hc_eps": float(g.meta(f"{a}.hyper_connection.epsilon", 1e-6)),
        "inv": yarn_inv_freq(rot, theta, factor, orig, bf, bs),
        "scale": k_mla ** -0.5 * mscale * mscale, "mscale": mscale, "theta": theta, "factor": factor,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="the first N layers (0: all)")
    ap.add_argument("--positions", default="all", help="all | last | comma-separated positions")
    ap.add_argument("--dump-res", action="store_true", help="write every layer's output streams")
    ap.add_argument("--gguf-py", default=None)
    ap.add_argument("--npu-like", action="store_true",
                    help="the runtime's arithmetic: absorbed MLA, bf16 GEMV inputs, f32 streams / KV rows / outputs")
    a = ap.parse_args()
    g = Gguf(a.gguf, a.gguf_py)
    cfg = config(g)
    N = a.layers or cfg["layers"]
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()]
    T = len(ids)
    pos = np.arange(T)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"xing4_0 GGUF: {cfg['layers']} layers ({N} run), hidden {cfg['hidden']}, {cfg['heads']} heads MLA "
          f"q {cfg['q_lora']} kv {cfg['kv_lora']} nope {cfg['nope']} rope {cfg['rope']} v {cfg['vdim']}, scale "
          f"{cfg['scale']:.6f} (mscale {cfg['mscale']:.5f}), hc {cfg['hc']} x {cfg['hc_iters']}; {T} tokens", flush=True)
    t0 = time.time()
    hc, eps = cfg["hc"], cfg["eps"]
    x = g.rows("token_embd.weight", ids)
    S = np.repeat(x[:, None, :], hc, axis=1)
    for l in range(N):
        p = f"blk.{l}."
        h, post, comb = hc_pre(S, g.w(p + "hc_attn_fn.weight"), g.w(p + "hc_attn_base.weight").astype(np.float64),
                               g.w(p + "hc_attn_scale.weight").astype(np.float64), eps, cfg["hc_iters"], cfg["hc_eps"])
        att = mla_absorbed if a.npu_like else mla
        y = att(g, p, rms(h, g.w(p + "attn_norm.weight").astype(np.float64), eps), pos, cfg)
        S = hc_post(y, S, post, comb)
        if a.npu_like:
            S = f32(S)
        h, post, comb = hc_pre(S, g.w(p + "hc_ffn_fn.weight"), g.w(p + "hc_ffn_base.weight").astype(np.float64),
                               g.w(p + "hc_ffn_scale.weight").astype(np.float64), eps, cfg["hc_iters"], cfg["hc_eps"])
        m = rms(h, g.w(p + "ffn_norm.weight").astype(np.float64), eps)
        y = (dense(g, p, m, a.npu_like, int(g.meta(f"{ARCH}.expert_feed_forward_length"))) if l < cfg["dense_lead"]
             else moe(g, p, m, cfg, a.npu_like))
        S = hc_post(f32(y) if a.npu_like else y, S, post, comb)
        if a.npu_like:
            S = f32(S)
        if a.dump_res:
            S.astype(np.float32).tofile(out / f"ref_res{l}.bin")
        print(f"  layer {l}: |S| {np.abs(S).max():.3f}  {time.time() - t0:.0f} s", flush=True)
    if a.positions == "all":
        ps = list(range(T))
    elif a.positions == "last":
        ps = [T - 1]
    else:
        ps = [int(s) for s in a.positions.split(",")]
    hn = rms(S[ps].mean(axis=1), g.w("output_norm.weight").astype(np.float64), eps)
    if a.npu_like:
        hn = bf(hn)
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
