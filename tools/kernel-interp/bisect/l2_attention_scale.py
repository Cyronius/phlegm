import os, sys, re, numpy as np
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
ops = open("/mnt/c/code/phlegm/tools/kernel-interp/bisect/m0c_ops.txt").read().splitlines()
rows = []
for ln in ops:
    mm = re.match(r"op@(\d+)\s+elf=\s*(\d+) \(\s*(\d+)B\).*a4 in \d+ pre=([0-9a-f]+)", ln)
    if mm and int(mm.group(1)) > 3586 and int(mm.group(3)) == 5200 and len(rows) < 40:
        b = bf16_to_f32(np.fromfile(bofile(mm.group(4)), dtype=np.uint16)[:64*2048]).reshape(64, 2048).astype(np.float64)
        rows += [b[r] for r in range(64) if np.abs(b[r]).max() > 0]
rows = np.stack(rows)
E = np.stack([bf16_to_f32(np.frombuffer(m.mm[m.data_base + m.tensors['model.embed_tokens.weight']['data_offsets'][0] + i*4096:][:4096], dtype=np.uint16)) for i in ids]).astype(np.float64)
x0 = F.moe_block(0, F.linear_attn_layer(0, E)); x1 = F.moe_block(1, F.linear_attn_layer(1, x0))
xr2 = F.full_attn_layer(2, x1, np.arange(T).astype(np.float64))
og_wo = xr2 - x1                      # the attention contribution (og @ Wo^T)
postln = m.bf16("model.layer.2.post_attention_layernorm.weight")
fin = bf16_to_f32(np.fromfile(f"{CAP}/000896.bo", dtype=np.uint16)[:2048]).astype(np.float64)
print("norms: |x1| last", np.linalg.norm(x1[-1]).round(3), "|attn contrib| last", np.linalg.norm(og_wo[-1]).round(3), "|cap final|", np.linalg.norm(fin).round(3))
for s in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    xr = x1 + s * og_wo
    xm = F.rms(xr) * postln
    best = [max(float(np.corrcoef(r, xm[t])[0, 1]) for t in range(T)) for r in rows]
    print(f"attn scale {s:5.2f}: batch rows mean {np.mean(best):.4f} max {max(best):.4f}", flush=True)
# what does FLM's xm2 look like vs our residual-only / attn-only directions (last token)?
for r in rows[:3]:
    print("cap row vs rms(x1)*postln per token:", [round(float(np.corrcoef(r, (F.rms(x1)*postln)[t])[0,1]),3) for t in range(T)])
