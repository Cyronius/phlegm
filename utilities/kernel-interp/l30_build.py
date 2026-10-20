"""Build ALL NPU buffers to run the full 30-layer interval-3 model (model_30L.q4nx)
on the NPU decode kernels — the config FLM's closed engine NaN-collapses on.

Generalizes run_5li3_npu.py to an N-layer schedule derived from the q4nx tensor
names. For each layer writes pool_L{l}.bin (512MB), pack_L{l}.bin (2MB),
side_L{l}.bin (6MB) and state_L{l}.bin (3MB kernel state, byte-format verified
vs FLM's captured 3LiF states). Also writes the lm_head pool, the CPU-computed
prefill final hidden, the block-1 lm_head act, the decode act for the first
sampled token, and first_token.npy.

Pools are 512MB each (30 -> 15GB on disk); they are built ONE AT A TIME (one
536MB numpy array live at a time) and streamed to disk. The NPU run
(l30_run_npu.py) reloads each layer's pool into a small set of resident pool BOs
via the driver's `load` directive.

Usage: python l30_build.py [model.q4nx] [out_dir]
"""
import numpy as np, os, sys, gc
from q4nx import Q4NX, MODEL_DIR, bf16_to_f32, f32_to_bf16
import build_pools as B
import full_forward as F

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(MODEL_DIR, "model_30L.q4nx")
OUT = sys.argv[2] if len(sys.argv) > 2 else "C:/code/FastFlowLM/npu-engine/m3out/l30"
os.makedirs(OUT, exist_ok=True)

m = Q4NX(MODEL)
F.m = m
ids = np.load("prompt_token_ids.npy")
T = len(ids)
F.T = T

nlayers = 0
while f"model.layer.{nlayers}.input_layernorm.weight" in m.tensors:
    nlayers += 1
sched = ["full_attention" if f"model.layer.{l}.self_attn.q_proj.weight" in m.tensors
         else "linear_attention" for l in range(nlayers)]
print(f"model {os.path.basename(MODEL)}  nlayers={nlayers}")
print("schedule:", "".join("F" if s == "full_attention" else "L" for s in sched))


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
    router = m.bf16(f"model.layer.{layer}.moe_router.weight")
    Wsg, Wsu, Ds = F.shared_weights(layer)
    sgate_w = m.bf16(f"model.layer.{layer}.shared_expert_gate.weight")
    for t in range(T):
        xm = (rms(x_res[t]) * postln).astype(np.float32)
        lg = xm @ router
        p = np.exp(lg - lg.max()); p /= p.sum()
        top = np.argsort(-p)[:8]; w8 = p[top]; w8 /= w8.sum()
        out = np.zeros(2048)
        for e, ww in zip(top, w8):
            gt, up, D = F.expert_weights(layer, int(e))
            out += ww * (D @ (silu(gt @ xm) * (up @ xm)))
        sh = Ds @ (silu(Wsg @ xm) * (Wsu @ xm))
        sg = 1 / (1 + np.exp(-(xm @ sgate_w)))
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
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    x = np.stack([bf16_to_f32(np.frombuffer(m.mm[base + i * 4096:base + (i + 1) * 4096], dtype=np.uint16))
                  for i in ids]).astype(np.float64)

    # ---- prefill: compute per-layer states + advance residual stream ----------
    states = []
    for l, lt in enumerate(sched):
        if lt == "linear_attention":
            c3, S = linear_prefill(l, x, T)
            states.append(serialize_linear_state(c3, S))
        else:
            k, v = full_prefill(l, x, T)
            states.append(serialize_kv_state(k, v, T))
        moe_prefill(l, x, T)
        print(f"  prefilled L{l:2d} ({'F' if lt=='full_attention' else 'L'})  absmax={np.abs(x).max():.4f} finite={bool(np.isfinite(x).all())}")

    hn_raw = x[-1].astype(np.float32)                         # raw residual (pre final-norm)
    hn = (rms(x[-1]) * m.bf16("model.norm.weight")).astype(np.float32)
    np.save(f"{OUT}/prefill_final_hidden.npy", hn)

    # first token from lm_head on the prefill hidden (block-1 path)
    lmb = np.frombuffer(m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, 8704)
    d = bf16_to_f32(np.ascontiguousarray(lmb[:, :512]).view(np.uint16))
    qq = np.ascontiguousarray(lmb[:, 512:]).view(np.int8)
    r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16); j = bc * 32 + r + 0 * i
    logits = np.zeros(248320, dtype=np.float32)
    for c0 in range(0, lmb.shape[0], 4096):
        ce = min(c0 + 4096, lmb.shape[0])
        vals = qq[c0:ce][:, p.reshape(-1)].reshape(ce - c0, 32, 8, 32).astype(np.float32)
        dd = d[c0:ce][:, j.reshape(-1)].reshape(ce - c0, 32, 8, 32)
        w = (vals * dd).reshape(ce - c0, 32, 256)
        for cc in range(c0, ce):
            logits[32 * (cc // 8):32 * (cc // 8) + 32] += w[cc - c0] @ hn[256 * (cc % 8):256 * (cc % 8) + 256]
    first = int(logits.argmax())
    np.save(f"{OUT}/first_token.npy", np.array(first))
    np.save(f"{OUT}/cpu_logits_build.npy", logits)
    print(f"prefill logits: finite={bool(np.isfinite(logits).all())} absmax={np.abs(logits).max():.3f} first_token={first}")

    NW = m.bf16("model.norm.weight")

    def embed(tok):
        return bf16_to_f32(np.frombuffer(m.mm[base + tok * 4096:base + (tok + 1) * 4096], dtype=np.uint16))

    # act_block1: [raw residual hidden | model.norm.weight] -> lm_head reproduces prefill logits
    act = np.zeros(1048576, dtype=np.uint8)
    act[:4096] = f32_to_bf16(hn_raw).view(np.uint8)
    act[4096:8192] = f32_to_bf16(NW).view(np.uint8)
    act.tofile(f"{OUT}/act_block1.bin")
    # act_decode: [embed(first_token) | model.norm.weight] -> the block-2 30-layer step
    act2 = np.zeros(1048576, dtype=np.uint8)
    act2[:4096] = f32_to_bf16(embed(first)).view(np.uint8)
    act2[4096:8192] = f32_to_bf16(NW).view(np.uint8)
    act2.tofile(f"{OUT}/act_decode.bin")

    # ---- build + stream all buffers to disk -----------------------------------
    for l, lt in enumerate(sched):
        full = lt == "full_attention"
        pool = B.build_layer_pool(m, l, full)
        pool.tofile(f"{OUT}/pool_L{l}.bin")
        del pool; gc.collect()
        B.build_pack(m, l).tofile(f"{OUT}/pack_L{l}.bin")
        B.build_side(m, l, full).tofile(f"{OUT}/side_L{l}.bin")
        states[l].tofile(f"{OUT}/state_L{l}.bin")
        print(f"  wrote buffers L{l:2d} ({'F' if full else 'L'})")
    lm = B.build_lmhead_pool(m)
    lm.tofile(f"{OUT}/pool_lmhead.bin")
    del lm; gc.collect()

    # persist the schedule for the run script
    np.save(f"{OUT}/schedule.npy", np.array([1 if s == "full_attention" else 0 for s in sched], dtype=np.int8))
    print(f"DONE — {nlayers} layers of buffers + lm_head pool in {OUT}")
