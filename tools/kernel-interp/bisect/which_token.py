import os, sys, numpy as np
sys.path.insert(0, "/mnt/c/code/phlegm/tools/kernel-interp")
os.environ.setdefault("MODEL_Q4NX", "/mnt/c/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model_3LiF.q4nx")
os.chdir("/mnt/c/code/phlegm/tools/kernel-interp")
from q4nx import Q4NX, bf16_to_f32
m = Q4NX(os.environ["MODEL_Q4NX"])
t0 = m.tensors["model.embed_tokens.weight"]; base = m.data_base + t0["data_offsets"][0]
nrow = 248320
def emb_u16(tok): return np.frombuffer(m.mm[base+tok*4096: base+(tok+1)*4096], dtype=np.uint16)
for f in ("000904", "000906", "000896"):
    act = np.fromfile(f"/mnt/c/caps/m0c/{f}.bo", dtype=np.uint16)
    a = act[:2048]
    print(f, "nonzero bytes beyond 8192:", int((act[4096:] != 0).sum()), "| norm slot == model.norm?",
          bool(np.array_equal(act[2048:4096], np.frombuffer(m.raw("model.norm.weight"), dtype=np.uint16)[:2048])))
    for tok in (248068, 198, 248046, 248045):
        print(f"   == embed({tok}) exact: {np.array_equal(a, emb_u16(tok))}")
    # nearest row by cosine over the whole table (bf16 -> f32), in chunks
    af = bf16_to_f32(a).astype(np.float32); best = (-2, -1)
    tab = np.frombuffer(m.mm[base: base + nrow*4096], dtype=np.uint16).reshape(nrow, 2048)
    for lo in range(0, nrow, 32768):
        w = bf16_to_f32(tab[lo:lo+32768].reshape(-1)).reshape(-1, 2048)
        c = (w @ af) / (np.linalg.norm(w, axis=1) * np.linalg.norm(af) + 1e-30)
        i = int(c.argmax());
        if c[i] > best[0]: best = (float(c[i]), lo + i)
    print(f"   nearest embedding row: token {best[1]} cos {best[0]:.6f}")
