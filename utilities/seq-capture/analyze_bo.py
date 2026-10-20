"""Localize the interval-3 collapse from xrt::bo sync metadata.

The closed engine moves every tensor through fixed-size DMA staging buffers, so
each sync() logs (dir, size, offset, fnv1a-hash). The degenerate "////////"
output means every decode step emits the same token -- so the activation buffers
feeding the final logits carry identical bytes each step, i.e. their hash LOCKS
to a constant across decode steps. A healthy run's hashes keep changing.

Method (per run, no byte dumps needed):
  1. Isolate the decode region (the long 1 MB-dominated tail).
  2. Split it into per-token decode steps using a recurring boundary signature.
  3. For each op-slot within a decode step, test whether its hash is constant
     across the last K steps.
  4. The EARLIEST constant slot in the broken run that is NOT constant in the
     healthy run is where the collapse originates.

Usage: python analyze_bo.py C:/caps/bo_i4_meta C:/caps/bo_i3_meta
"""
import sys, os
from collections import Counter

def load(p):
    rows = []
    with open(os.path.join(p, "bo_trace.tsv")) as f:
        for ln in f:
            a = ln.rstrip("\n").split("\t")
            if len(a) >= 5:
                rows.append((int(a[0]), a[1], int(a[2]), int(a[3]), a[4]))
    return rows  # (idx, dir, size, offset, hash)

def decode_region(rows):
    """Return the tail sub-list once we've entered the steady 1 MB decode loop.
    Weights load as huge (>=8 MB) syncs up front; prefill uses larger tiles.
    Heuristic: start at the last position where a >=4 MB sync occurs, so the
    remaining tail is the pure small-tile decode loop."""
    last_big = 0
    for i, x in enumerate(rows):
        if x[2] >= 4 * 1024 * 1024:
            last_big = i
    return rows[last_big + 1:]

def find_period(region):
    """Find the decode-step period by autocorrelation on the (dir,size) signature
    stream. Returns the smallest lag (in [4, len/3]) that maximizes signature
    self-match."""
    sig = [(x[1], x[2]) for x in region]
    n = len(sig)
    best, best_lag = -1.0, None
    for lag in range(4, max(5, n // 3)):
        m = sum(1 for i in range(n - lag) if sig[i] == sig[i + lag])
        score = m / (n - lag)
        if score > best:
            best, best_lag = score, lag
    return best_lag, best

def segment_by_anchor(region):
    """Split the decode region into steps using the most common (dir,size,hash)
    triple as a step boundary anchor -- robust when a fixed op recurs once/step."""
    keys = [(x[1], x[2], x[4]) for x in region]
    # candidate anchors: recurring (dir,size) whose hash is constant -> once/step
    from collections import defaultdict
    by_ds = defaultdict(list)
    for i, x in enumerate(region):
        by_ds[(x[1], x[2])].append(i)
    # pick anchor: a (dir,size) that appears many times with a SINGLE constant hash
    anchor = None
    best_count = 0
    for ds, idxs in by_ds.items():
        hs = set(region[i][4] for i in idxs)
        if len(hs) == 1 and len(idxs) > best_count and len(idxs) >= 4:
            best_count, anchor = len(idxs), ds
    if anchor is None:
        return None, None
    starts = [i for i, x in enumerate(region) if (x[1], x[2]) == anchor]
    steps = [region[starts[j]:starts[j + 1]] for j in range(len(starts) - 1)]
    return steps, anchor

def slot_constancy(steps, k=6):
    """Across the last k complete, equal-length steps, report for each slot
    whether its hash is constant. Returns list of (slot, dir, size, constant, nuniq)."""
    if not steps:
        return []
    # use steps that share the modal length
    lens = Counter(len(s) for s in steps)
    L = lens.most_common(1)[0][0]
    uniform = [s for s in steps if len(s) == L][-k:]
    out = []
    for slot in range(L):
        hashes = [s[slot][4] for s in uniform]
        d, sz = uniform[0][slot][1], uniform[0][slot][2]
        nun = len(set(hashes))
        out.append((slot, d, sz, nun == 1, nun))
    return out, L, len(uniform)

def analyze(path, label):
    rows = load(path)
    reg = decode_region(rows)
    steps, anchor = segment_by_anchor(reg)
    period, score = find_period(reg)
    print(f"\n===== {label} =====")
    print(f"  total syncs={len(rows)}  decode-region={len(reg)}  "
          f"autocorr-period={period} (score={score:.2f})")
    if not steps:
        print("  could not segment into steps")
        return None
    sc, L, nused = slot_constancy(steps)
    print(f"  decode steps found={len(steps)}  step-len(modal)={L}  "
          f"steps-used={nused}  anchor(dir,size)={anchor}")
    const_slots = [s for s in sc if s[3]]
    print(f"  constant-across-steps slots: {len(const_slots)}/{L}")
    return sc, L

def main():
    a4 = analyze(sys.argv[1], "8Li4 (healthy)")
    a3 = analyze(sys.argv[2], "6Li3 (broken)")
    if not a4 or not a3:
        return
    sc4, L4 = a4
    sc3, L3 = a3
    print("\n===== slot-by-slot (broken vs healthy) =====")
    print("  slot dir     size   broken_const healthy_const  <- collapse where broken locks but healthy varies")
    L = min(L4, L3)
    first = None
    for i in range(L):
        b = sc3[i]; h = sc4[i]
        mark = ""
        if b[3] and not h[3]:
            mark = "  <<< COLLAPSE"
            if first is None: first = i
        print(f"  {i:3d}  {b[1]:3s}  {b[2]:>8d}   {str(b[3]):5s}({b[4]})     {str(h[3]):5s}({h[4]}){mark}")
    print(f"\n  EARLIEST slot that locks in broken but varies in healthy: {first}")

if __name__ == "__main__":
    main()
