import os, sys, time, numpy as np
sys.path.insert(0, "/mnt/c/code/phlegm/tools/kernel-interp")
os.environ["MODEL_Q4NX"] = "/mnt/c/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx"
os.chdir("/mnt/c/code/phlegm/tools/kernel-interp")
import full_forward as F
from q4nx import bf16_to_f32
m, ids, T = F.m, F.ids, F.T
nl = max(int(n.split(".layer.")[1].split(".")[0]) for n in m.tensors if ".layer." in n) + 1
print("layers", nl, "fmt", m.fmt, flush=True)
E = np.stack([bf16_to_f32(np.frombuffer(m.mm[m.data_base + m.tensors['model.embed_tokens.weight']['data_offsets'][0] + i*4096:][:4096], dtype=np.uint16)) for i in ids]).astype(np.float64)
x = E; t0 = time.time()
for l in range(nl):
    full = f"model.layer.{l}.self_attn.q_proj.weight" in m.tensors
    x = F.moe_block(l, F.full_attn_layer(l, x, np.arange(T).astype(np.float64)) if full else F.linear_attn_layer(l, x))
    print(f"layer {l} {'FULL' if full else 'lin '} done  absmax {np.abs(x).max():.3f}  t={time.time()-t0:.0f}s", flush=True)
hn = (F.rms(x[-1]) * m.bf16("model.norm.weight")).astype(np.float32)
lg = m.lmhead_logits(hn).astype(np.float64)
ref = np.fromfile("/mnt/c/caps/pf_t11_full/008566.bo", dtype=np.float32).astype(np.float64)
odd = lg[1::2][:124160]
print(f"prefill logits vs pf_t11_full 008566 (odd-row convention): corr {np.corrcoef(odd, ref[:124160])[0,1]:.5f}  argmax vocab mine {2*int(odd.argmax())+1} ref {2*int(ref[:124160].argmax())+1}", flush=True)
nz = np.nonzero(ref)[0]
print(f"(nonzero-index convention): corr {np.corrcoef(lg[nz], ref[nz])[0,1]:.5f}", flush=True)
