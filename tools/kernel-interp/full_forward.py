"""Full 3LiF (L,L,F) model forward on CPU from model_3LiF.q4nx ONLY.

Every layout/math element here was verified against captured NPU buffers:
- q4 chunks (block32 q4_1, bf16 d/m planar, 16-lane nibble interleave)
- standard matmul tiling + MoE expert band tiling
- linear-attn block: qkv -> conv(silu)+qk-l2norm -> gated delta rule (1/sqrt(128))
  -> RMSNormGated*silu(z) -> out_proj -> residual
- MoE: router softmax top-8 renorm, experts silu(g)*u @ down, shared expert
  * sigmoid(shared_gate)
- full-attn: fused q_proj per-head [q256|gate256], q/k norm (weights stored
  effective), partial RoPE (dim 64 half-split, theta 1e7), sigmoid gate, o_proj
- final model.norm + q8 lm_head
Verification target: captured first-decode logits (m0c 000897.bo).
"""
import numpy as np, os, json
from q4nx import Q4NX, MODEL_DIR, bf16_to_f32
from moe_forward import dq_chunks, silu

m = Q4NX(os.environ.get("MODEL_Q4NX", os.path.join(MODEL_DIR, "model_3LiF.q4nx")))
print("model format:", m.fmt)
ids = np.load("prompt_token_ids.npy")
T = len(ids)

def rms(x, eps=1e-6):
    return x / np.sqrt((np.asarray(x, dtype=np.float64)**2).mean(-1, keepdims=True) + eps)

def dequant_std(name, out_dim, in_dim):
    """Format-aware dequant of a quantized matmul tensor to logical [out,in].
    q4_1 (1.0.2) is byte-identical to the old path; q4k (1.0.3) dequants Q4_K
    and undoes the linear-attention head-pairing reorders."""
    return m.matmul_w(name, out_dim, in_dim)

def _raster(chunks_bytes, out_dim, in_dim):
    return m.dq_tile(chunks_bytes, out_dim, in_dim)

def expert_weights(layer, e):
    """file layout: each expert contiguous, raster chunks (no reorder).
    Per-expert byte stride = (out//32)*(in//256)*chunk_bytes = 128*chunk_bytes."""
    stride = 128 * m.chunk_bytes
    upb = np.frombuffer(m.raw(f"model.layer.{layer}.mlp.up_exps_proj.weight"), dtype=np.uint8)
    gtb = np.frombuffer(m.raw(f"model.layer.{layer}.mlp.gate_exps_proj.weight"), dtype=np.uint8)
    dnb = np.frombuffer(m.raw(f"model.layer.{layer}.mlp.down_exps_proj.weight"), dtype=np.uint8)
    up = _raster(upb[e*stride:(e+1)*stride], 512, 2048)
    gt = _raster(gtb[e*stride:(e+1)*stride], 512, 2048)
    D = _raster(dnb[e*stride:(e+1)*stride], 2048, 512)
    return gt, up, D

def shared_weights(layer):
    Wsg = _raster(np.frombuffer(m.raw(f"model.layer.{layer}.mlp.share_gate_exps_proj.weight"), dtype=np.uint8), 512, 2048)
    Wsu = _raster(np.frombuffer(m.raw(f"model.layer.{layer}.mlp.share_up_exps_proj.weight"), dtype=np.uint8), 512, 2048)
    D = _raster(np.frombuffer(m.raw(f"model.layer.{layer}.mlp.share_down_exps_proj.weight"), dtype=np.uint8), 2048, 512)
    return Wsg, Wsu, D

def moe_block(layer, x_res):
    """x_res [T,2048] raw residual -> x_res + moe_out"""
    postln = m.bf16(f"model.layer.{layer}.post_attention_layernorm.weight")
    xm = (rms(x_res) * postln).astype(np.float32)
    router = m.bf16(f"model.layer.{layer}.moe_router.weight")     # [2048,256]
    lg = xm @ router
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    top = np.argsort(-p, 1)[:, :8]
    w8 = np.take_along_axis(p, top, 1); w8 /= w8.sum(1, keepdims=True)
    used = {}
    for t in range(T):
        for e, ww in zip(top[t], w8[t]):
            used.setdefault(int(e), []).append((t, float(ww)))
    out = np.zeros((T, 2048))
    for e, lst in used.items():
        gt, up, D = expert_weights(layer, e)
        for t, ww in lst:
            h = silu(gt @ xm[t]) * (up @ xm[t])
            out[t] += ww * (D @ h)
    Wsg, Wsu, Ds = shared_weights(layer)
    sh = silu(xm @ Wsg.T) * (xm @ Wsu.T) @ Ds.T
    sgate = 1/(1+np.exp(-(xm @ m.bf16(f"model.layer.{layer}.shared_expert_gate.weight"))))
    return x_res + out + sgate[:, None]*sh

def linear_attn_layer(layer, x_res):
    ln = m.bf16(f"model.layer.{layer}.input_layernorm.weight")
    x = (rms(x_res) * ln).astype(np.float32)
    Wqkv = dequant_std(f"model.layer.{layer}.linear_attn.qkv_proj.weight", 8192, 2048)
    Wz = dequant_std(f"model.layer.{layer}.self_attn.gate_proj.weight", 4096, 2048)
    Wout = dequant_std(f"model.layer.{layer}.linear_attn.ssm_out_proj.weight", 2048, 4096)
    convw = m.bf16(f"model.layer.{layer}.linear_attn.ssm_conv1d.weight")   # [4,8192]
    qkv = x @ Wqkv.T                                    # [T,8192]
    z = silu(x @ Wz.T)                                  # [T,4096]
    pad = np.zeros((3, 8192), dtype=np.float32)
    seq = np.concatenate([pad, qkv])
    conv = np.zeros((T, 8192), dtype=np.float32)
    for k in range(4):
        conv += convw[k][None, :] * seq[k:k+T]
    conv = silu(conv)
    q = rms(conv[:, :2048].reshape(T, 16, 128), 1e-6)   # l2norm via rms*sqrt? no: true l2
    # careful: kernel uses true L2 norm (verified): x/||x||
    def l2n(a):
        return a / np.sqrt((a**2).sum(-1, keepdims=True) + 1e-6)
    q = l2n(conv[:, :2048].reshape(T, 16, 128))
    k_ = l2n(conv[:, 2048:4096].reshape(T, 16, 128))
    v = conv[:, 4096:].reshape(T, 32, 128).astype(np.float64)
    Wa = m.bf16(f"model.layer.{layer}.linear_attn.ssm_alpha_proj.weight")  # [2048,32]
    Wb = m.bf16(f"model.layer.{layer}.linear_attn.ssm_beta_proj.weight")
    A = m.f32(f"model.layer.{layer}.linear_attn.ssm_a")
    dtb = m.f32(f"model.layer.{layer}.linear_attn.ssm_dt.bias")
    a = x @ Wa; b = x @ Wb
    decay = np.exp(A * np.log1p(np.exp(a + dtb)))  # file ssm_a = -exp(A_log), pre-baked
    beta = 1/(1+np.exp(-b))
    o = np.zeros((T, 32, 128))
    S = np.zeros((32, 128, 128))
    for t in range(T):
        for h in range(32):
            kk, qq = k_[t, h//2], q[t, h//2]
            S[h] *= decay[t, h]
            delta = beta[t, h] * (v[t, h] - S[h].T @ kk)
            S[h] += np.outer(kk, delta)
            o[t, h] = (S[h].T @ qq) / np.sqrt(128)
    nw = m.bf16(f"model.layer.{layer}.linear_attn.ssm_norm.weight")        # [128]
    og = (rms(o) * nw).reshape(T, 4096) * z
    return x_res + og.astype(np.float32) @ Wout.T

def full_attn_layer(layer, x_res, pos):
    ln = m.bf16(f"model.layer.{layer}.input_layernorm.weight")
    x = (rms(x_res) * ln).astype(np.float32)
    Wqg = dequant_std(f"model.layer.{layer}.self_attn.q_proj.weight", 8192, 2048)
    Wq, Wg = Wqg[:4096], Wqg[4096:]     # file stores planar [q | gate] (HF interleaves per head)
    Wk = dequant_std(f"model.layer.{layer}.self_attn.k_proj.weight", 512, 2048)
    Wv = dequant_std(f"model.layer.{layer}.self_attn.v_proj.weight", 512, 2048)
    Wo = dequant_std(f"model.layer.{layer}.self_attn.o_proj.weight", 2048, 4096)
    qn = m.bf16(f"model.layer.{layer}.self_attn.q_norm.weight")
    kn = m.bf16(f"model.layer.{layer}.self_attn.k_norm.weight")
    q = (x @ Wq.T).reshape(T, 16, 256)
    g = (x @ Wg.T).reshape(T, 16, 256)
    k_ = (x @ Wk.T).reshape(T, 2, 256)
    v = (x @ Wv.T).reshape(T, 2, 256)
    q = rms(q) * qn; k_ = rms(k_) * kn
    def rope(t_, p):
        h = 32
        freqs = (1e7) ** (-np.arange(h)/h)
        ang = p[:, None] * freqs[None, :]
        C, Sn = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        y = t_.copy()
        x1, x2 = t_[..., :h], t_[..., h:64]
        y[..., :h] = x1*C - x2*Sn
        y[..., h:64] = x2*C + x1*Sn
        return y
    q, k_ = rope(q, pos), rope(k_, pos)
    o = np.zeros((T, 16, 256))
    for h in range(16):
        s = (q[:, h] @ k_[:, h//8].T)/16.0 + np.triu(np.full((T, T), -np.inf), 1)
        aw = np.exp(s - s.max(1, keepdims=True)); aw /= aw.sum(1, keepdims=True)
        o[:, h] = aw @ v[:, h//8]
    og = o * (1/(1+np.exp(-g)))
    return x_res + og.reshape(T, 4096).astype(np.float32) @ Wo.T

if __name__ == "__main__":
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    E = np.stack([bf16_to_f32(np.frombuffer(m.mm[base+i*4096: base+(i+1)*4096], dtype=np.uint16)) for i in ids])
    x = E.astype(np.float64)
    x = moe_block(0, linear_attn_layer(0, x))
    print("L0 done, absmax", np.abs(x).max())
    x = moe_block(1, linear_attn_layer(1, x))
    print("L1 done, absmax", np.abs(x).max())
    x = moe_block(2, full_attn_layer(2, x, np.arange(T).astype(np.float64)))
    print("L2 done, absmax", np.abs(x).max())
    hn = (rms(x[-1]) * m.bf16("model.norm.weight")).astype(np.float32)
    np.save("final_hidden.npy", hn)
    # q8 lm_head logits (format-aware: 16-lane for q4_1, column-major for q4k)
    logits = m.lmhead_logits(hn)
    np.save("my_logits.npy", logits)
    refpath = "C:/caps/m0c/000897.bo"
    if m.fmt == "q4_1" and os.path.exists(refpath):
        ref = np.fromfile(refpath, dtype=np.float32)
        nz = np.nonzero(ref)[0]
        c = np.corrcoef(logits[nz], ref[nz])[0, 1]
        print("logits vs captured (nonzero half): corr", round(float(c), 6))
    print("logits: finite", bool(np.isfinite(logits).all()), "absmax", round(float(np.abs(logits).max()), 3),
          "argmax", int(logits.argmax()))
    print("my top5", np.argsort(logits)[-5:][::-1])
