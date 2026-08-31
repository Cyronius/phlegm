"""Replicate one full NPU decode step on CPU and compare captured logits.

Decode block 2 (m0c): input = embed(token 248068) at position 11, states from
the prefill->decode boundary roundtrip:
  state buffer (3MB, per linear layer): [0:49152] conv state bf16 [3,8192]
  (last 3 tokens' post-qkv), [49152:] GDN state fp32 [32,128,128]
  L2 KV pack (3MB): k bf16 [T,512] @0, v @byte 1073152
Output: fp32 logits, odd vocab rows in the captured 1MB buffer (row 2b+1).
"""
import numpy as np
from q4nx import bf16_to_f32
import full_forward as F

m = F.m
CAP = "C:/caps/m0c"
POS = 11        # new token position (prompt was 0..10)

def silu(x):
    return x / (1 + np.exp(-x))

def load_linear_state(path):
    raw = np.fromfile(path, dtype=np.uint8)
    conv_state = bf16_to_f32(np.frombuffer(raw[:49152].tobytes(), dtype=np.uint16)).reshape(3, 8192).copy()
    S = np.frombuffer(raw[49152:49152+32*128*128*4].tobytes(), dtype=np.float32).reshape(32, 128, 128).astype(np.float64)
    return conv_state, S

def linear_decode(layer, x_res, conv_state, S):
    ln = m.bf16(f"model.layer.{layer}.input_layernorm.weight")
    x = (F.rms(x_res) * ln).astype(np.float32)
    Wqkv = F.dequant_std(f"model.layer.{layer}.linear_attn.qkv_proj.weight", 8192, 2048)
    Wz = F.dequant_std(f"model.layer.{layer}.self_attn.gate_proj.weight", 4096, 2048)
    Wout = F.dequant_std(f"model.layer.{layer}.linear_attn.ssm_out_proj.weight", 2048, 4096)
    convw = m.bf16(f"model.layer.{layer}.linear_attn.ssm_conv1d.weight")
    qkv = x @ Wqkv.T
    z = silu(x @ Wz.T)
    seq = np.vstack([conv_state, qkv])              # [4, 8192]
    c = silu((convw * seq).sum(0))                  # depthwise k=4, this token
    def l2n(a):
        return a / np.sqrt((a**2).sum(-1, keepdims=True) + 1e-6)
    q = l2n(c[:2048].reshape(16, 128))
    k = l2n(c[2048:4096].reshape(16, 128))
    v = c[4096:].reshape(32, 128).astype(np.float64)
    Wa = m.bf16(f"model.layer.{layer}.linear_attn.ssm_alpha_proj.weight")
    Wb = m.bf16(f"model.layer.{layer}.linear_attn.ssm_beta_proj.weight")
    A = m.f32(f"model.layer.{layer}.linear_attn.ssm_a")
    dtb = m.f32(f"model.layer.{layer}.linear_attn.ssm_dt.bias")
    decay = np.exp(A * np.log1p(np.exp(x @ Wa + dtb)))  # file ssm_a = -exp(A_log), pre-baked
    beta = 1/(1+np.exp(-(x @ Wb)))
    o = np.zeros((32, 128))
    for h in range(32):
        kk, qq = k[h//2], q[h//2]
        S[h] *= decay[h]
        delta = beta[h] * (v[h] - S[h].T @ kk)
        S[h] += np.outer(kk, delta)
        o[h] = (S[h].T @ qq) / np.sqrt(128)
    nw = m.bf16(f"model.layer.{layer}.linear_attn.ssm_norm.weight")
    og = (F.rms(o) * nw).reshape(4096) * z
    new_conv_state = np.vstack([conv_state[1:], qkv[None, :]]) if qkv.ndim == 1 else np.vstack([conv_state[1:], qkv])
    return x_res + og.astype(np.float32) @ Wout.T, new_conv_state, S

def moe_decode(layer, x_res):
    postln = m.bf16(f"model.layer.{layer}.post_attention_layernorm.weight")
    xm = (F.rms(x_res) * postln).astype(np.float32)
    router = m.bf16(f"model.layer.{layer}.moe_router.weight")
    lg = xm @ router
    p = np.exp(lg - lg.max()); p /= p.sum()
    top = np.argsort(-p)[:8]
    w8 = p[top]; w8 /= w8.sum()
    out = np.zeros(2048)
    for e, ww in zip(top, w8):
        gt, up, D = F.expert_weights(layer, int(e))
        h = silu(gt @ xm) * (up @ xm)
        out += ww * (D @ h)
    Wsg, Wsu, Ds = F.shared_weights(layer)
    sh = Ds @ (silu(Wsg @ xm) * (Wsu @ xm))
    sg = 1/(1+np.exp(-(xm @ m.bf16(f"model.layer.{layer}.shared_expert_gate.weight"))))
    return x_res + out + sg*sh

def attn_decode(layer, x_res, kcache, vcache, pos):
    ln = m.bf16(f"model.layer.{layer}.input_layernorm.weight")
    x = (F.rms(x_res) * ln).astype(np.float32)
    Wqg = F.dequant_std(f"model.layer.{layer}.self_attn.q_proj.weight", 8192, 2048)
    Wq, Wg = Wqg[:4096], Wqg[4096:]
    Wk = F.dequant_std(f"model.layer.{layer}.self_attn.k_proj.weight", 512, 2048)
    Wv = F.dequant_std(f"model.layer.{layer}.self_attn.v_proj.weight", 512, 2048)
    Wo = F.dequant_std(f"model.layer.{layer}.self_attn.o_proj.weight", 2048, 4096)
    qn = m.bf16(f"model.layer.{layer}.self_attn.q_norm.weight")
    kn = m.bf16(f"model.layer.{layer}.self_attn.k_norm.weight")
    q = (F.rms((x @ Wq.T).reshape(16, 256)) * qn).astype(np.float64)
    g = (x @ Wg.T).reshape(16, 256)
    k = (F.rms((x @ Wk.T).reshape(2, 256)) * kn).astype(np.float64)
    v = (x @ Wv.T).reshape(2, 256).astype(np.float64)
    def rope1(t_, p):
        h = 32
        freqs = (1e7) ** (-np.arange(h)/h)
        ang = p * freqs
        C, Sn = np.cos(ang), np.sin(ang)
        y = t_.copy()
        x1, x2 = t_[..., :h], t_[..., h:64]
        y[..., :h] = x1*C - x2*Sn
        y[..., h:64] = x2*C + x1*Sn
        return y
    q = rope1(q, POS); k = rope1(k, POS)
    K = np.concatenate([kcache, k[None]], 0)     # [pos+1, 2, 256]
    V = np.concatenate([vcache, v[None]], 0)
    o = np.zeros((16, 256))
    for h in range(16):
        s = (K[:, h//8] @ q[h]) / 16.0
        a = np.exp(s - s.max()); a /= a.sum()
        o[h] = a @ V[:, h//8]
    og = o * (1/(1+np.exp(-g)))
    return x_res + og.reshape(4096).astype(np.float32) @ Wo.T, K, V

def lm_head_odd(hn):
    lm = m.raw("lm_head.weight")
    lmb = np.frombuffer(lm, dtype=np.uint8).reshape(-1, 8704)
    d = bf16_to_f32(np.ascontiguousarray(lmb[:, :512]).view(np.uint16))
    qq = np.ascontiguousarray(lmb[:, 512:]).view(np.int8)
    r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
    p = (r//16)*4096 + bc*512 + i*16 + (r % 16)
    j = bc*32 + r + 0*i
    out = np.zeros(248320, dtype=np.float32)
    CH = 8192
    for c0 in range(0, lmb.shape[0], CH):
        ce = min(c0+CH, lmb.shape[0])
        vals = qq[c0:ce][:, p.reshape(-1)].reshape(ce-c0, 32, 8, 32).astype(np.float32)
        dd = d[c0:ce][:, j.reshape(-1)].reshape(ce-c0, 32, 8, 32)
        w = (vals*dd).reshape(ce-c0, 32, 256)
        for cc in range(c0, ce):
            out[32*(cc//8):32*(cc//8)+32] += w[cc-c0] @ hn[256*(cc % 8):256*(cc % 8)+256]
    return out[1::2][:124160]     # odd rows, buffer order

if __name__ == "__main__":
    TOK = 248068
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    x = bf16_to_f32(np.frombuffer(m.mm[base+TOK*4096: base+(TOK+1)*4096], dtype=np.uint16)).astype(np.float64)
    cs0, S0 = load_linear_state(f"{CAP}/000898.bo")
    cs1, S1 = load_linear_state(f"{CAP}/000900.bo")
    kvb = bf16_to_f32(np.fromfile(f"{CAP}/000902.bo", dtype=np.uint16))
    kcache = kvb[:11*512].reshape(11, 2, 256).astype(np.float64)
    vcache = kvb[536576:536576+11*512].reshape(11, 2, 256).astype(np.float64)

    x, cs0, S0 = linear_decode(0, x, cs0, S0)
    x = moe_decode(0, x)
    x, cs1, S1 = linear_decode(1, x, cs1, S1)
    x = moe_decode(1, x)
    x, K, V = attn_decode(2, x, kcache, vcache, POS)
    x = moe_decode(2, x)
    hn = (F.rms(x) * m.bf16("model.norm.weight")).astype(np.float32)
    mine = lm_head_odd(hn)
    ref = np.fromfile(f"{CAP}/000905.bo", dtype=np.float32)[:124160]
    c = np.corrcoef(mine, ref)[0, 1]
    print("decode block-2 logits corr:", round(float(c), 5))
    print("my top5 (buffer idx):", np.argsort(mine)[-5:][::-1], "ref top5:", np.argsort(ref)[-5:][::-1])
    print("my argmax vocab:", 2*int(mine.argmax())+1, "ref argmax vocab:", 2*int(ref.argmax())+1)
