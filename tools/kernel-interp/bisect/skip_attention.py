import os, sys, numpy as np
sys.path.insert(0, "/mnt/c/code/phlegm/tools/kernel-interp")
os.environ.setdefault("MODEL_Q4NX", "/mnt/c/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model_3LiF.q4nx")
os.chdir("/mnt/c/code/phlegm/tools/kernel-interp")
import full_forward as F
import decode_step as DS
from q4nx import bf16_to_f32
m, ids, T = F.m, F.ids, F.T
CAP = "/mnt/c/caps/m0c"
# prefill: final hidden with L2 attention skipped
E = np.stack([bf16_to_f32(np.frombuffer(m.mm[m.data_base + m.tensors['model.embed_tokens.weight']['data_offsets'][0] + i*4096:][:4096], dtype=np.uint16)) for i in ids]).astype(np.float64)
x0 = F.moe_block(0, F.linear_attn_layer(0, E)); x1 = F.moe_block(1, F.linear_attn_layer(1, x0))
fin = bf16_to_f32(np.fromfile(f"{CAP}/000896.bo", dtype=np.uint16)[:2048]).astype(np.float64)
x2_skip = F.moe_block(2, x1)
x2_full = F.moe_block(2, F.full_attn_layer(2, x1, np.arange(T).astype(np.float64)))
print(f"prefill final hidden vs cap 000896: with attention {np.corrcoef(x2_full[-1], fin)[0,1]:.5f}   attention skipped {np.corrcoef(x2_skip[-1], fin)[0,1]:.5f}", flush=True)
hn = (F.rms(x2_skip[-1]) * m.bf16("model.norm.weight")).astype(np.float32)
lg = m.lmhead_logits(hn); ref = np.fromfile(f"{CAP}/000897.bo", dtype=np.float32); nz = np.nonzero(ref)[0]
print(f"prefill logits (attention skipped) vs cap 000897: corr {np.corrcoef(lg[nz], ref[nz])[0,1]:.5f}", flush=True)
# decode step: token 248068 at pos 11, with L2 attention skipped vs full
ref = np.fromfile(f"{CAP}/000905.bo", dtype=np.float32)[:124160]
TOK, POS = 248068, 11
t0 = m.tensors["model.embed_tokens.weight"]; base = m.data_base + t0["data_offsets"][0]
x = bf16_to_f32(np.frombuffer(m.mm[base+TOK*4096: base+(TOK+1)*4096], dtype=np.uint16)).astype(np.float64)
cs0, S0 = DS.load_linear_state(f"{CAP}/000898.bo"); cs1, S1 = DS.load_linear_state(f"{CAP}/000900.bo")
kvb = bf16_to_f32(np.fromfile(f"{CAP}/000902.bo", dtype=np.uint16))
kc = kvb[:POS*512].reshape(POS, 2, 256).astype(np.float64); vc = kvb[536576:536576+POS*512].reshape(POS, 2, 256).astype(np.float64)
x, _, _ = DS.linear_decode(0, x, cs0, S0); x = DS.moe_decode(0, x); x, _, _ = DS.linear_decode(1, x, cs1, S1); x = DS.moe_decode(1, x)
for label, xa in (("with attention", DS.attn_decode(2, x.copy(), kc, vc, POS)[0]), ("attention skipped", x.copy())):
    y = DS.moe_decode(2, xa)
    hn = (DS.F.rms(y) * m.bf16("model.norm.weight")).astype(np.float32)
    mine = DS.lm_head_odd(hn)
    print(f"decode logits [{label}] vs cap 000905: corr {np.corrcoef(mine, ref)[0,1]:.5f}  argmax mine {2*int(mine.argmax())+1} ref {2*int(ref.argmax())+1}", flush=True)
