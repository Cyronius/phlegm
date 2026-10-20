"""Validate our Q4_K (1.0.3) reader against a REAL converter-produced file, using
INDEPENDENT ground truth (gguf.dequantize) -- not our own packer.

Flow: Qwen3.5-0.8B-q4_k.gguf --(reference convert.py)--> model.q4nx (1.0.3, Q4_K).
For each non-reordered Q4_K matmul tensor, our dequant_q4k_file(...) must match
gguf.dequantize(source) at the quant bound (~0.01-0.05). q_proj is skipped (it
carries the (g p h)->(p g h) reorder); the 0.8B has NO linear reorders
(reorder_linear_required=False, ffn<=6144), so every other matmul is identity.

  python validate_real_08b.py <converted_model.q4nx> <source.gguf>
"""
import os, sys, json, struct
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernel-interp"))
import q4nx_v103 as V
from gguf import GGUFReader, dequantize

Q4NX_PATH, GGUF_PATH = sys.argv[1], sys.argv[2]
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "configs", "qwen3.5_0.8b.json")


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def st_raw(path, hdr, base, name):
    o0, o1 = hdr[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + o0)
        return f.read(o1 - o0)


# gguf source tensors by name
gg = {t.name: t for t in GGUFReader(GGUF_PATH).tensors}

# name map (q4nx_name -> gguf_name), expanding {bid} against actual layers
cfg = json.load(open(CFG))["name_map"]
hdr, base = st_header(Q4NX_PATH)
q2g = {}
for info in cfg.values():
    gt, qt = info["gguf_name"], info["q4nx_name"]
    if "{bid}" in gt:
        for L in range(64):
            q = qt.format(bid=L); g = gt.format(bid=L)
            if q in hdr and g in gg:
                q2g[q] = g
    elif qt in hdr and gt in gg:
        q2g[qt] = gt

# validate the non-reordered Q4_K matmuls
SKIP = ("q_proj",)  # carries the (g p h)->(p g h) reorder
KEEP = ("k_proj", "v_proj", "o_proj", "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj",
        "attn_gate_proj", "linear_attn.qkv_proj", "ssm_out_proj")
rows = []
for name, meta in hdr.items():
    if name == "__metadata__" or meta["dtype"] != "I8":
        continue
    cb = meta["shape"][-1]
    if cb != V.Q4K_CHUNK:          # only Q4_K tensors (skip Q4_1 ssm_out, Q8 alpha/beta/lmhead)
        continue
    if any(s in name for s in SKIP) or not any(k in name for k in KEEP):
        continue
    if name not in q2g:
        continue
    out, inn = meta["shape"][0] * 32, meta["shape"][1] * 256
    ours = V.dequant_q4k_file(np.frombuffer(st_raw(Q4NX_PATH, hdr, base, name), np.uint8), out, inn)
    ref = np.asarray(dequantize(gg[q2g[name]].data, gg[q2g[name]].tensor_type), np.float32)
    if ref.shape != (out, inn):
        rows.append((9.99, 0, f"SHAPE {ref.shape} vs {(out, inn)}  {name}"))
        continue
    d = np.abs(ours - ref)
    rel = float(d.mean() / (np.abs(ref).mean() + 1e-9))
    rows.append((rel, float(d.max()), name))

rows.sort(reverse=True)
print(f"validated {len(rows)} Q4_K matmul tensors (ours vs gguf.dequantize):")
for rel, mx, nm in rows[:12]:
    print(f"  rel {rel:.4e}  maxabs {mx:.4e}  {nm}")
print("  ...")
for rel, mx, nm in rows[-3:]:
    print(f"  rel {rel:.4e}  maxabs {mx:.4e}  {nm}")
worst = rows[0][0] if rows else 1.0
print("\nRESULT:", "PASS (Q4_K byte layout correct on a real file)" if worst < 0.08 else "FAIL")
