"""Prepare all NPU buffers to run the interval-3 model (5Li3, L,L,F,L,L) on the
NPU decode kernels — the config FLM's closed engine NaN-collapses on.

Builds per-layer pools/packs/sides from model_5Li3.q4nx, computes prefill
states with our verified CPU forward, serializes them into the 3MB kernel state
format (conv bf16[3,8192] @0 + GDN fp32[32,128,128] @49152 for linear layers;
KV pack k@0 v@1073152 for full-attn), and writes an m3_chain script.

State format verified byte-exact against FLM's captured 3LiF states.
"""
import numpy as np, os, sys
from q4nx import Q4NX, MODEL_DIR, bf16_to_f32, f32_to_bf16
import build_pools as B
import full_forward as F

OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/code/FastFlowLM/npu-engine/m3out/5li3"
os.makedirs(OUT, exist_ok=True)
m = Q4NX(os.path.join(MODEL_DIR, "model_5Li3.q4nx"))
F.m = m   # retarget full_forward's dequant helpers at 5Li3

nlayers = 0
while f"model.layer.{nlayers}.input_layernorm.weight" in m.tensors:
    nlayers += 1
sched = ["full_attention" if f"model.layer.{l}.self_attn.q_proj.weight" in m.tensors
         else "linear_attention" for l in range(nlayers)]
print("5Li3 schedule:", sched)


def rms(x, eps=1e-6):
    return x / np.sqrt((np.asarray(x, np.float64) ** 2).mean(-1, keepdims=True) + eps)


def silu(x):
    return x / (1 + np.exp(-x))


def l2n(a):
    return a / np.sqrt((a ** 2).sum(-1, keepdims=True) + 1e-6)


def linear_prefill(layer, x_res, T):
    """advance x_res in place; return (conv3 [3,8192], S [32,128,128])."""
    ln = m.bf16(f"model.layer.{layer}.input_layernorm.weight")
    xn = (rms(x_res) * ln).astype(np.float32)
    Wqkv = F.dequant_std(f"model.layer.{layer}.linear_attn.qkv_proj.weight", 8192, 2048)
    Wz = F.dequant_std(f"model.layer.{layer}.self_attn.gate_proj.weight", 4096, 2048)
    Wout = F.dequant_std(f"model.layer.{layer}.linear_attn.ssm_out_proj.weight", 2048, 4096)
    convw = m.bf16(f"model.layer.{layer}.linear_attn.ssm_conv1d.weight")
    qkv = xn @ Wqkv.T
    z = silu(xn @ Wz.T)
    pad = np.zeros((3, 8192), dtype=np.float32)
    seq = np.vstack([pad, qkv])
    conv = np.zeros((T, 8192), dtype=np.float32)
    for k in range(4):
        conv += convw[k][None, :] * seq[k:k + T]
    conv = silu(conv)
    Wa = m.bf16(f"model.layer.{layer}.linear_attn.ssm_alpha_proj.weight")
    Wb = m.bf16(f"model.layer.{layer}.linear_attn.ssm_beta_proj.weight")
    A = m.f32(f"model.layer.{layer}.linear_attn.ssm_a")
    dtb = m.f32(f"model.layer.{layer}.linear_attn.ssm_dt.bias")
    decay = np.exp(A * np.log1p(np.exp(xn @ Wa + dtb)))
    beta = 1 / (1 + np.exp(-(xn @ Wb)))
    nw = m.bf16(f"model.layer.{layer}.linear_attn.ssm_norm.weight")
    S = np.zeros((32, 128, 128))
    o = np.zeros((T, 4096))
    for t in range(T):
        c = conv[t]
        q = l2n(c[:2048].reshape(16, 128)); k = l2n(c[2048:4096].reshape(16, 128))
        v = c[4096:].reshape(32, 128).astype(np.float64)
        for h in range(32):
            kk, qq = k[h // 2], q[h // 2]
            S[h] *= decay[t, h]
            delta = beta[t, h] * (v[h] - S[h].T @ kk)
            S[h] += np.outer(kk, delta)
            o[t, h * 128:(h + 1) * 128] = (S[h].T @ qq) / np.sqrt(128)
    og = (rms(o.reshape(T, 32, 128)) * nw).reshape(T, 4096) * z
    x_res += og.astype(np.float32) @ Wout.T
    conv3 = qkv[T - 3:T].copy()
    return conv3, S


def full_prefill(layer, x_res, T):
    ln = m.bf16(f"model.layer.{layer}.input_layernorm.weight")
    xn = (rms(x_res) * ln).astype(np.float32)
    Wqg = F.dequant_std(f"model.layer.{layer}.self_attn.q_proj.weight", 8192, 2048)
    Wq, Wg = Wqg[:4096], Wqg[4096:]
    Wk = F.dequant_std(f"model.layer.{layer}.self_attn.k_proj.weight", 512, 2048)
    Wv = F.dequant_std(f"model.layer.{layer}.self_attn.v_proj.weight", 512, 2048)
    Wo = F.dequant_std(f"model.layer.{layer}.self_attn.o_proj.weight", 2048, 4096)
    qnw = m.bf16(f"model.layer.{layer}.self_attn.q_norm.weight")
    knw = m.bf16(f"model.layer.{layer}.self_attn.k_norm.weight")

    def rope(t_, pos):
        h = 32
        freqs = (1e7) ** (-np.arange(h) / h)
        ang = pos * freqs
        c, s = np.cos(ang), np.sin(ang)
        y = t_.copy()
        x1, x2 = t_[..., :h], t_[..., h:64]
        y[..., :h] = x1 * c - x2 * s
        y[..., h:64] = x2 * c + x1 * s
        return y

    q = (rms((xn @ Wq.T).reshape(T, 16, 256)) * qnw)
    g = (xn @ Wg.T).reshape(T, 16, 256)
    k = (rms((xn @ Wk.T).reshape(T, 2, 256)) * knw)
    v = (xn @ Wv.T).reshape(T, 2, 256).astype(np.float64)
    K = np.stack([rope(k[t], t) for t in range(T)])
    Q = np.stack([rope(q[t], t) for t in range(T)])
    og = np.zeros((T, 16, 256))
    for t in range(T):
        for h in range(16):
            kvh = h // 8
            s = (K[:t + 1, kvh] @ Q[t, h]) / 16.0
            a = np.exp(s - s.max()); a /= a.sum()
            og[t, h] = (a @ v[:t + 1, kvh]) * (1 / (1 + np.exp(-g[t, h])))
    x_res += og.reshape(T, 4096).astype(np.float32) @ Wo.T
    return K.reshape(T, 512), v.reshape(T, 512)


def moe_prefill(layer, x_res, T):
    postln = m.bf16(f"model.layer.{layer}.post_attention_layernorm.weight")
    for t in range(T):
        xm = (rms(x_res[t]) * postln).astype(np.float32)
        router = m.bf16(f"model.layer.{layer}.moe_router.weight")
        lg = xm @ router
        p = np.exp(lg - lg.max()); p /= p.sum()
        top = np.argsort(-p)[:8]; w8 = p[top]; w8 /= w8.sum()
        out = np.zeros(2048)
        for e, ww in zip(top, w8):
            gt, up, D = F.expert_weights(layer, int(e))
            out += ww * (D @ (silu(gt @ xm) * (up @ xm)))
        Wsg, Wsu, Ds = F.shared_weights(layer)
        sh = Ds @ (silu(Wsg @ xm) * (Wsu @ xm))
        sg = 1 / (1 + np.exp(-(xm @ m.bf16(f"model.layer.{layer}.shared_expert_gate.weight"))))
        x_res[t] += out + sg * sh


def serialize_linear_state(conv3, S):
    buf = np.zeros(3145728, dtype=np.uint8)
    buf[:49152] = f32_to_bf16(conv3.reshape(-1).astype(np.float32)).view(np.uint8)
    buf[49152:49152 + 32 * 128 * 128 * 4] = S.reshape(-1).astype(np.float32).view(np.uint8)
    return buf


def serialize_kv_state(k, v, T):
    buf = np.zeros(3145728, dtype=np.uint8)
    buf[:T * 512 * 2] = f32_to_bf16(k.reshape(-1).astype(np.float32)).view(np.uint8)
    buf[1073152:1073152 + T * 512 * 2] = f32_to_bf16(v.reshape(-1).astype(np.float32)).view(np.uint8)
    return buf


if __name__ == "__main__":
    ids = np.load("prompt_token_ids.npy")
    T = len(ids)
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    x = np.stack([bf16_to_f32(np.frombuffer(m.mm[base + i * 4096:base + (i + 1) * 4096], dtype=np.uint16))
                  for i in ids]).astype(np.float64)

    states = []
    for l, lt in enumerate(sched):
        if lt == "linear_attention":
            c3, S = linear_prefill(l, x, T)
            states.append(serialize_linear_state(c3, S))
        else:
            k, v = full_prefill(l, x, T)
            states.append(serialize_kv_state(k, v, T))
        moe_prefill(l, x, T)
        print(f"  prefilled layer {l} ({lt})")

    hn = (rms(x[-1]) * m.bf16("model.norm.weight")).astype(np.float32)
    np.save(f"{OUT}/prefill_final_hidden.npy", hn)
    # Raw (pre-final-norm) residual hidden of the last prompt token: the input
    # the generate loop feeds to full_logits() (which applies rms + norm.weight)
    # to sample the FIRST generated token.
    np.save(f"{OUT}/prefill_final_residual.npy", x[-1].astype(np.float32))

    # Pools/packs/sides are prompt-INDEPENDENT (weights only); only the per-layer
    # states and the prefill hidden depend on the prompt. Skip the ~2.5GB pool
    # rebuild when the files already exist (set REBUILD_POOLS=1 to force).
    rebuild_pools = os.environ.get("REBUILD_POOLS") == "1"
    for l, lt in enumerate(sched):
        full = lt == "full_attention"
        if rebuild_pools or not os.path.exists(f"{OUT}/pool_L{l}.bin"):
            B.build_layer_pool(m, l, full).tofile(f"{OUT}/pool_L{l}.bin")
            B.build_pack(m, l).tofile(f"{OUT}/pack_L{l}.bin")
            B.build_side(m, l, full).tofile(f"{OUT}/side_L{l}.bin")
        states[l].tofile(f"{OUT}/state_L{l}.bin")
        print(f"  wrote state L{l}" + ("" if rebuild_pools or os.path.exists(f"{OUT}/pool_L{l}.bin") else " (+pool)"))
    if rebuild_pools or not os.path.exists(f"{OUT}/pool_lmhead.bin"):
        B.build_lmhead_pool(m).tofile(f"{OUT}/pool_lmhead.bin")

    # act buffer: [prefill_final_hidden | model.norm.weight] for block-1 lm_head,
    # and [embed(next_token) | model.norm.weight] for decode steps. Write the
    # block-1 act (lm_head on prefill hidden) to sample the first token.
    act = np.zeros(1048576, dtype=np.uint8)
    act[:4096] = f32_to_bf16(hn).view(np.uint8)  # already rms-normed*normw is applied by kernel? no: row0=hidden, row1=normw
    # kernel rms-norms row0 and multiplies row1(=normw). So row0 must be the RAW residual hidden (pre-final-norm).
    xr = x[-1].astype(np.float32)
    act[:4096] = f32_to_bf16(xr).view(np.uint8)
    act[4096:8192] = f32_to_bf16(m.bf16("model.norm.weight")).view(np.uint8)
    act.tofile(f"{OUT}/act_block1.bin")
    print("wrote act_block1.bin (raw residual hidden + norm weight)")
    print("DONE — buffers in", OUT)
