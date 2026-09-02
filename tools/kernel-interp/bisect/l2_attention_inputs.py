import os, sys, numpy as np
sys.path.insert(0, "/mnt/c/code/phlegm/tools/kernel-interp")
os.environ.setdefault("MODEL_Q4NX", "/mnt/c/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model_3LiF.q4nx")
os.chdir("/mnt/c/code/phlegm/tools/kernel-interp")
import full_forward as F
from q4nx import bf16_to_f32
m, ids, T = F.m, F.ids, F.T
CAP = "/mnt/c/caps/m0c"
tr = [l.rstrip("\n").split("\t") for l in open(f"{CAP}/bo_trace.tsv")]
def bofile(h):
    for a in tr:
        if len(a) >= 5 and a[4].startswith(h): return f"{CAP}/{a[0]}.bo"
    raise KeyError(h)
def act(f, n=T, dim=2048, off=0):
    return bf16_to_f32(np.fromfile(f, dtype=np.uint16)[off:off + n*dim]).reshape(n, dim).astype(np.float64)
def corr_rows(a, b): return [round(float(np.corrcoef(a[t], b[t])[0, 1]), 4) for t in range(a.shape[0])]
L = 2
xn2 = act(f"{CAP}/run_3499_a4.bin")                       # FLM's normed L2 input (teacher-forced)
Wqg = F.dequant_std(f"model.layer.{L}.self_attn.q_proj.weight", 8192, 2048).astype(np.float64)
Wq, Wg = Wqg[:4096], Wqg[4096:]
Wk = F.dequant_std(f"model.layer.{L}.self_attn.k_proj.weight", 512, 2048).astype(np.float64)
Wv = F.dequant_std(f"model.layer.{L}.self_attn.v_proj.weight", 512, 2048).astype(np.float64)
Wo = F.dequant_std(f"model.layer.{L}.self_attn.o_proj.weight", 2048, 4096).astype(np.float64)
qn = m.bf16(f"model.layer.{L}.self_attn.q_norm.weight").astype(np.float64); kn = m.bf16(f"model.layer.{L}.self_attn.k_norm.weight").astype(np.float64)
q = xn2 @ Wq.T; g = xn2 @ Wg.T; k = xn2 @ Wk.T; v = xn2 @ Wv.T
# captured q / gate op outputs (op@3499 a3 post c79ea008, op@3511 b696c83b): [T,4096] bf16?
for name, h, mine in (("q proj", "c79ea008", q), ("gate proj", "b696c83b", g)):
    try:
        capq = act(bofile(h), T, 4096); print(f"{name} vs cap: corr {corr_rows(mine, capq)}")
    except KeyError as e: print(name, "no file for", e)
kvb = np.fromfile(f"{CAP}/000902.bo", dtype=np.uint16)
kcap = bf16_to_f32(kvb[:T*512]).reshape(T, 2, 256).astype(np.float64)
vcap = bf16_to_f32(kvb[536576:536576 + T*512]).reshape(T, 2, 256).astype(np.float64)
print("v (raw proj) vs cap V:", corr_rows(v.reshape(T, 512), vcap.reshape(T, 512)))
def rope(t_, p, half=32, theta=1e7, interleave=False):
    y = t_.copy()
    freqs = theta ** (-np.arange(half) / half); ang = p[:, None] * freqs[None, :]
    C, S = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    if not interleave:
        x1, x2 = t_[..., :half], t_[..., half:2*half]
        y[..., :half] = x1 * C - x2 * S; y[..., half:2*half] = x2 * C + x1 * S
    else:
        x1, x2 = t_[..., 0:2*half:2], t_[..., 1:2*half:2]
        y[..., 0:2*half:2] = x1 * C - x2 * S; y[..., 1:2*half:2] = x2 * C + x1 * S
    return y
pos = np.arange(T).astype(np.float64)
kh = k.reshape(T, 2, 256)
for label, kk in (("rms*kn, rope half32 theta1e7", rope(F.rms(kh) * kn, pos)),
                  ("rms*kn no rope", F.rms(kh) * kn),
                  ("rms*kn rope interleaved", rope(F.rms(kh) * kn, pos, interleave=True)),
                  ("rms*kn rope half64", rope(F.rms(kh) * kn, pos, half=64)),
                  ("rms*kn rope half128 (full)", rope(F.rms(kh) * kn, pos, half=128)),
                  ("rms*kn rope theta1e6", rope(F.rms(kh) * kn, pos, theta=1e6)),
                  ("rms*(1+kn) rope", rope(F.rms(kh) * (1 + kn), pos)),
                  ("no norm, rope", rope(kh, pos))):
    print(f"k' [{label:32}] vs cap K: {corr_rows(kk.reshape(T, 512), kcap.reshape(T, 512))}")
# attention -> xm2 vs captured MoE input (shared-expert op a4 in, hash 173bef71)
xm2cap = act(bofile("173bef71"))
x1 = None
# residual: recover x_res before L2 is not captured; use replica chain
E = np.stack([bf16_to_f32(np.frombuffer(m.mm[m.data_base + m.tensors['model.embed_tokens.weight']['data_offsets'][0] + i*4096:][:4096], dtype=np.uint16)) for i in ids]).astype(np.float64)
x0 = F.moe_block(0, F.linear_attn_layer(0, E)); x1 = F.moe_block(1, F.linear_attn_layer(1, x0))
postln = m.bf16(f"model.layer.{L}.post_attention_layernorm.weight")
def attn_out(qq, kk, vv, gg, scale=1/16.0, gate=True):
    o = np.zeros((T, 16, 256))
    for h in range(16):
        s = (qq[:, h] @ kk[:, h//8].T) * scale
        s = s + np.triu(np.full((T, T), -1e30), 1)
        a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        o[:, h] = a @ vv[:, h//8]
    og = o * (1/(1+np.exp(-gg))) if gate else o
    return og.reshape(T, 4096)
qh = rope(F.rms(q.reshape(T, 16, 256)) * qn, pos); kk = rope(F.rms(kh) * kn, pos); vv = v.reshape(T, 2, 256); gg = g.reshape(T, 16, 256)
for label, og in (("standard", attn_out(qh, kk, vv, gg)),
                  ("cap K/V", attn_out(qh, kcap, vcap, gg)),
                  ("no gate", attn_out(qh, kk, vv, gg, gate=False)),
                  ("scale 1/sqrt(128)", attn_out(qh, kk, vv, gg, scale=1/np.sqrt(128))),
                  ("no q norm", attn_out(rope(q.reshape(T,16,256), pos), kk, vv, gg))):
    xr = x1 + og @ Wo.T
    xm = F.rms(xr) * postln
    print(f"xm2 [{label:18}] vs cap MoE input: {corr_rows(xm, xm2cap)}")
