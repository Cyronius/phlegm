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
    return None
# router weights: pack L2 vs model file
pack2 = np.fromfile("/mnt/c/caps/m0d/000124.bo", dtype=np.uint8)
rw_pack = pack2[12288:12288 + 1048576].view(np.uint16)
rw_file = np.frombuffer(m.raw("model.layer.2.moe_router.weight"), dtype=np.uint16)[:524288]
print("L2 router: pack == file ?", np.array_equal(rw_pack, rw_file), "| postln equal?",
      np.array_equal(pack2[4096:8192].view(np.uint16), np.frombuffer(m.raw("model.layer.2.post_attention_layernorm.weight"), dtype=np.uint16)[:2048]))
for l in (0, 1):
    pk = np.fromfile(f"/mnt/c/caps/m0d/00011{8 if l == 0 else 21}.bo" if l == 0 else "/mnt/c/caps/m0d/000121.bo", dtype=np.uint8)
    print(f"L{l} router: pack == file ?", np.array_equal(pk[12288:12288+1048576].view(np.uint16), np.frombuffer(m.raw(f"model.layer.{l}.moe_router.weight"), dtype=np.uint16)[:524288]))
# chained residual into L2 and our attention output
E = np.stack([bf16_to_f32(np.frombuffer(m.mm[m.data_base + m.tensors['model.embed_tokens.weight']['data_offsets'][0] + i*4096:][:4096], dtype=np.uint16)) for i in ids]).astype(np.float64)
x0 = F.moe_block(0, F.linear_attn_layer(0, E)); x1 = F.moe_block(1, F.linear_attn_layer(1, x0))
xr2 = F.full_attn_layer(2, x1, np.arange(T).astype(np.float64))
postln = m.bf16("model.layer.2.post_attention_layernorm.weight")
xm2 = F.rms(xr2) * postln
xm1 = F.rms(F.linear_attn_layer(1, x0)) * m.bf16("model.layer.1.post_attention_layernorm.weight")
# expert ops of L2: op table lines after 3586 with 5200B elfs -> a4 hashes
ops = open("/mnt/c/code/phlegm/tools/kernel-interp/bisect/m0c_ops.txt").read().splitlines()
def batches(start_after, count):
    out = []
    for ln in ops:
        mm = re.match(r"op@(\d+)\s+elf=\s*(\d+) \(\s*(\d+)B\).*a4 in \d+ pre=([0-9a-f]+)", ln)
        if not mm: continue
        ev, elf, sz, h = int(mm.group(1)), int(mm.group(2)), int(mm.group(3)), mm.group(4)
        if ev > start_after and sz == 5200:
            out.append((ev, h))
        if len(out) >= count: break
    return out
def analyse(label, bl, xm):
    for ev, h in bl:
        f = bofile(h)
        if f is None: print(ev, "no file"); continue
        b = bf16_to_f32(np.fromfile(f, dtype=np.uint16)[:64*2048]).reshape(64, 2048).astype(np.float64)
        rows = [r for r in range(64) if np.abs(b[r]).max() > 0]
        res = []
        for r in rows:
            c = [float(np.corrcoef(b[r], xm[t])[0, 1]) for t in range(T)]
            res.append((int(np.argmax(c)), round(max(c), 4)))
        print(f"{label} op@{ev}: batch rows {len(rows)} -> (token, corr): {res}")
analyse("L1 expert", batches(2055, 3), xm1)
analyse("L2 expert", batches(3586, 6), xm2)
