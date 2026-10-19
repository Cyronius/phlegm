import os, sys, numpy as np
sys.path.insert(0, "/mnt/c/code/phlegm/tools/kernel-interp")
os.environ.setdefault("MODEL_Q4NX", "/mnt/c/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model_3LiF.q4nx")
os.chdir("/mnt/c/code/phlegm/tools/kernel-interp")
import full_forward as F
from q4nx import bf16_to_f32
m, ids, T = F.m, F.ids, F.T
CAP = "/mnt/c/caps/m0c"
def act(f, n=T, dim=2048):
    return bf16_to_f32(np.fromfile(f, dtype=np.uint16)[:n*dim]).reshape(n, dim).astype(np.float64)
def report(name, mine, cap):
    cs = [float(np.corrcoef(mine[t], cap[t])[0, 1]) for t in range(cap.shape[0])]
    rel = np.abs(mine - cap).max() / np.abs(cap).max()
    print(f"{name:34} corr/token min {min(cs):.5f} max {max(cs):.5f} last {cs[-1]:.5f}  maxrel {rel:.2e}", flush=True)
def normed(x, l):
    return F.rms(x) * m.bf16(f"model.layer.{l}.input_layernorm.weight")
t0 = m.tensors["model.embed_tokens.weight"]; base = m.data_base + t0["data_offsets"][0]
E = np.stack([bf16_to_f32(np.frombuffer(m.mm[base+i*4096: base+(i+1)*4096], dtype=np.uint16)) for i in ids]).astype(np.float64)
report("xn0 = rms(E)*ln0 vs cap op255 in", normed(E, 0), act(f"{CAP}/run_255_a4.bin"))
x0 = F.moe_block(0, F.linear_attn_layer(0, E))
report("xn1 (chained) vs cap op1990 in", normed(x0, 1), act(f"{CAP}/run_1990_a4.bin"))
x1 = F.moe_block(1, F.linear_attn_layer(1, x0))
report("xn2 (chained) vs cap op3499 in", normed(x1, 2), act(f"{CAP}/run_3499_a4.bin"))
x2 = F.moe_block(2, F.full_attn_layer(2, x1, np.arange(T).astype(np.float64)))
fin = act(f"{CAP}/000896.bo", 1)[0]
print(f"final residual (chained, last token) vs cap 000896: corr {np.corrcoef(x2[-1], fin)[0,1]:.5f}")
hn = (F.rms(x2[-1]) * m.bf16("model.norm.weight")); hc = F.rms(fin) * m.bf16("model.norm.weight")
print(f"final normed: corr {np.corrcoef(hn, hc)[0,1]:.5f}")
