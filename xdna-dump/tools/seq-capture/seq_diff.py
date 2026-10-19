"""Diff two capture corpora produced by the xrt_coreutil shim.

Each corpus is a directory of NNNNNN.seq blobs plus trace.tsv (one line per
captured elf: idx, kind, size, fnv1a). Two runs that differ only in
full_attention_interval should emit the *same op sequence in a regular
pattern*; the interval-3 bug shows up as a divergence in the ordered stream
(a missing / extra / reordered op). This aligns the two ordered hash streams
and prints the first divergence -- the smoking gun.

Usage:
    python seq_diff.py <dir_interval4> <dir_interval3>
"""
import sys, os, difflib
from collections import Counter


def load(dirpath):
    """Return ordered list of (idx, kind, size, hash) for 'elf' captures."""
    tp = os.path.join(dirpath, "trace.tsv")
    rows = []
    with open(tp, "r", encoding="latin1") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            idx, kind, size, h = parts[0], parts[1], parts[2], parts[3]
            if kind == "elf":
                rows.append((idx, kind, int(size), h))
    return rows


def signature(rows):
    """Map each distinct (size,hash) to a short stable label A,B,C,... so the
    op stream is human-readable. Same shape+content -> same label across runs."""
    return [(r[2], r[3]) for r in rows]


def label_map(*streams):
    seen = {}
    def lab(n):
        s = ""
        n += 1
        while n:
            n, r = divmod(n - 1, 26)
            s = chr(ord('A') + r) + s
        return s
    order = []
    for st in streams:
        for key in st:
            if key not in seen:
                seen[key] = lab(len(order))
                order.append(key)
    return seen


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    a, b = load(a_dir), load(b_dir)
    sa, sb = signature(a), signature(b)
    lm = label_map(sa, sb)
    la = [lm[k] for k in sa]
    lb = [lm[k] for k in sb]

    print(f"# {a_dir}: {len(a)} ops")
    print(f"# {b_dir}: {len(b)} ops")
    print(f"# distinct op signatures: {len(lm)}")
    print()

    # Per-op-type counts -- a missing op type is the coarsest signal.
    ca, cb = Counter(la), Counter(lb)
    diff_types = sorted(set(ca) | set(cb), key=lambda t: (len(t), t))
    print("op   countA countB")
    for t in diff_types:
        flag = "" if ca[t] == cb[t] else "  <-- differs"
        print(f"{t:<4} {ca[t]:>6} {cb[t]:>6}{flag}")
    print()

    # Ordered alignment: first structural divergence in the submission stream.
    sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
    print("## ordered alignment (op stream) ##")
    printed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        print(f"[{tag}] A[{i1}:{i2}]={''.join(la[i1:i2]) or '-'}  "
              f"B[{j1}:{j2}]={''.join(lb[j1:j2]) or '-'}")
        # show the surrounding window once, for the first divergence
        if printed == 0:
            lo = max(0, i1 - 4)
            print(f"    A context [{lo}:{i2+4}]: {''.join(la[lo:i2+4])}")
            lo = max(0, j1 - 4)
            print(f"    B context [{lo}:{j2+4}]: {''.join(lb[lo:j2+4])}")
        printed += 1
    if printed == 0:
        print("streams identical in structure (no scheduling divergence at op granularity)")
    print(f"\n# {printed} divergent region(s)")


if __name__ == "__main__":
    main()
