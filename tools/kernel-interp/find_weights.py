"""Locate model tensors inside captured NPU weight-pool blobs.

For every tensor in the q4nx slice, take distinctive snippets (start / middle)
and search each large blob. Reports tensor -> (blob, offset) hits.
Usage: python find_weights.py C:/caps/m0d [min_blob_mb]
"""
import sys, os, glob
import numpy as np
from q4nx import Q4NX, MODEL_DIR

capdir = sys.argv[1]
min_mb = float(sys.argv[2]) if len(sys.argv) > 2 else 100
m = Q4NX(os.path.join(MODEL_DIR, "model_3LiF.q4nx"))

blobs = []
for p in glob.glob(os.path.join(capdir, "blob_*.bin")):
    sz = os.path.getsize(p)
    if sz >= min_mb * 1024 * 1024:
        blobs.append((p, open(p, "rb").read()))
print(f"searching {len(blobs)} blobs >= {min_mb} MB")

def snippets(name):
    """distinctive snippets: (label, offset_in_tensor, bytes)"""
    r = m.raw(name)
    n = len(r)
    out = [("start", 0, r[:64])]
    if n > 8192:
        mid = (n // 2) & ~63
        out.append(("mid", mid, r[mid : mid + 64]))
    return out

for name in m.tensors:
    hits = []
    for label, toff, snip in snippets(name):
        if len(set(snip)) < 8:  # too low-entropy to be distinctive
            continue
        for p, data in blobs:
            pos = data.find(snip)
            while pos != -1:
                hits.append((label, toff, os.path.basename(p), pos))
                if len(hits) > 6:
                    break
                pos = data.find(snip, pos + 1)
    if hits:
        for label, toff, bp, pos in hits:
            print(f"{name}  {label}@{toff}  ->  {bp} @ {pos}  (pool_off-t_off={pos-toff})")
    else:
        print(f"{name}  NO HIT")
