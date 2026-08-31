"""Scan dumped bo tensors in submission order for the FIRST appearance of NaN.

The interval-3 collapse is a NaN blowup in the logits. NaN doesn't arise from a
finite matmul unless an input is already NaN/Inf -- so the earliest sync whose
active payload contains NaN is at/just-after the faulty computation. Comparing
the broken run's first-NaN index against the healthy run (which should have none)
localizes where interval-3 diverges numerically.

We interpret each 1 MB tile as bf16 (the engine's activation dtype) and measure
NaN fraction over the *nonzero* region only, so stale padding doesn't dominate.

Usage: python scan_nan.py C:/caps/bo_i3_dump [C:/caps/bo_i4_dump]
"""
import sys, os, glob
import numpy as np

def load_trace(D):
    t = {}
    with open(os.path.join(D, "bo_trace.tsv")) as f:
        for ln in f:
            a = ln.rstrip("\n").split("\t")
            if len(a) >= 6:
                t[int(a[0])] = (a[1], int(a[2]), a[4], int(a[5]))  # dir,size,hash,dumped
    return t

def bf16(b):
    u = b[:(len(b)//2)*2].view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)

def scan(D):
    trace = load_trace(D)
    files = sorted(glob.glob(os.path.join(D, "*.bo")))
    rows = []
    for fp in files:
        idx = int(os.path.basename(fp)[:6])
        b = np.fromfile(fp, dtype=np.uint8)
        # active region = up to last nonzero byte
        nz = np.nonzero(b)[0]
        active = b[:nz[-1]+1] if len(nz) else b[:0]
        f = bf16(active)
        nan = int(np.isnan(f).sum())
        inf = int(np.isinf(f).sum())
        frac = nan / max(1, len(f))
        d = trace.get(idx, ("?", len(b), "", 1))
        rows.append((idx, d[0], d[1], len(active), nan, inf, frac))
    return rows

def report(D):
    rows = scan(D)
    print(f"\n===== {D}  ({len(rows)} tensors) =====")
    nanrows = [r for r in rows if r[4] > 0]
    if not nanrows:
        print("  NO NaN in any dumped tensor.")
        return
    first = nanrows[0]
    print(f"  FIRST NaN at sync idx={first[0]} dir={first[1]} size={first[2]} "
          f"active={first[3]}B nan={first[4]} frac={first[6]:.3f}")
    print(f"  total tensors with NaN: {len(nanrows)}  "
          f"(first {first[0]}, last {nanrows[-1][0]})")
    # context: show a window of indices around the first NaN
    print("  --- window around first NaN (idx dir size activeB nan nanfrac) ---")
    lo = first[0] - 8
    for r in rows:
        if lo <= r[0] <= first[0] + 6:
            flag = " <<< first NaN" if r[0] == first[0] else ""
            print(f"    {r[0]:6d} {r[1]} {r[2]:>9d} {r[3]:>8d} {r[4]:>7d} {r[6]:.3f}{flag}")

if __name__ == "__main__":
    for D in sys.argv[1:]:
        report(D)
