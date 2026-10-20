"""Per-tensor logical-order read-back diff between a 1.0.2 file and its 1.0.3
transcode.  A wrong reorder-undo shows as O(weight) error on ONE tensor; the
q4_1->Q4_K requant shows as uniform small error everywhere."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernel-interp"))
from q4nx import Q4NX

MD = "C:/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2"
m2 = Q4NX(os.path.join(MD, "model_3LiF.q4nx"))
m3 = Q4NX(os.path.join(MD, "model_3LiF_v103.q4nx"))
print("fmts", m2.fmt, m3.fmt)


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max()), float(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-9))


worst = []
for name, meta in m2.tensors.items():
    dt, sh = meta["dtype"], meta["shape"]
    if dt == "I8":
        if name == "lm_head.weight":
            continue
        out, inn = sh[0] * 32, sh[1] * 256
        if "exps_proj" in name and "share" not in name:
            oe = out // 256
            for e in (0, 200):
                r2 = np.frombuffer(m2.raw(name), np.uint8)[e * 128 * m2.chunk_bytes:(e + 1) * 128 * m2.chunk_bytes]
                r3 = np.frombuffer(m3.raw(name), np.uint8)[e * 128 * m3.chunk_bytes:(e + 1) * 128 * m3.chunk_bytes]
                worst.append((*rel(m2.dq_tile(r2, oe, inn), m3.dq_tile(r3, oe, inn))[::-1], f"{name}#e{e}"))
        else:
            worst.append((*rel(m2.matmul_w(name, out, inn), m3.matmul_w(name, out, inn))[::-1], name))
    elif dt in ("BF16", "F32"):
        a = m2.bf16(name) if dt == "BF16" else m2.f32(name)
        b = m3.bf16(name) if dt == "BF16" else m3.f32(name)
        worst.append((*rel(a, b)[::-1], name))

worst.sort(reverse=True)
print("WORST 14 (rel, maxabs, name):")
for rr, mx, nm in worst[:14]:
    print(f"  rel {rr:.4e}  maxabs {mx:.4e}  {nm}")
print("BEST 3:")
for rr, mx, nm in worst[-3:]:
    print(f"  rel {rr:.4e}  maxabs {mx:.4e}  {nm}")
