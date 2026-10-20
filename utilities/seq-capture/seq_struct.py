"""Structural analysis of captured NPU control sequences.

Raw-byte diffing is defeated by per-layer weight-offset patching (nearly every
op is byte-unique). This parses each .seq into its AIE ctrlcode command stream
(format per src/include/npu_utils/instr_utils/*.hpp) and builds two views:

  * a STRUCTURAL fingerprint per op -- command types + shapes, with the patched
    weight pointers (DDR-patch arg_offset) MASKED OUT. Ops that differ only in
    which weights they load collapse to one fingerprint, so a genuine op-type
    difference between two runs becomes visible.
  * the weight-offset progression -- the DDR-patch (arg_idx, arg_offset) values
    in order, so a wrong/duplicated/skipped weight load (the likely interval-3
    defect) shows up even when the op structure is identical.

Usage: python seq_struct.py <dirA> <dirB>
"""
import sys, os, struct, difflib
from collections import Counter

# op_headers (src/include/npu_utils/instr_utils/npu_cmd.hpp) -> (name, op_lines)
OPS = {0: ("write", 6), 1: ("blockwrite", 12), 3: ("maskwrite", 7),
       0x80: ("tct", 4), 0x81: ("ddr", 12)}


def parse(path):
    """Return (struct_sig, ddr_list) for one .seq.
    struct_sig: tuple describing op structure with weight offsets masked.
    ddr_list: list of (arg_idx, arg_offset) from DDR-patch commands, in order."""
    with open(path, "rb") as f:
        d = f.read()
    w = list(struct.unpack("<%dI" % (len(d) // 4), d[: (len(d) // 4) * 4]))
    sig = []
    ddr = []
    i = 4  # skip 4-word header
    n = len(w)
    while i < n:
        op = w[i]
        if op in OPS:
            name, lines = OPS[op]
            if name == "ddr":
                arg_idx = w[i + 8] if i + 8 < n else -1
                arg_off = w[i + 10] if i + 10 < n else -1
                ddr.append((arg_idx, arg_off))
                sig.append(("ddr", arg_idx))               # keep arg_idx, MASK arg_offset
            elif name == "blockwrite":
                blen = w[i + 4] if i + 4 < n else -1
                d0 = w[i + 7] if i + 7 < n else -1
                d1 = w[i + 8] if i + 8 < n else -1
                sig.append(("bw", blen, d0, d1))           # shape, not buffer_offset
            elif name == "maskwrite":
                sig.append(("mw", w[i + 4] if i + 4 < n else -1, w[i + 5] if i + 5 < n else -1))
            elif name == "write":
                sig.append(("wr", w[i + 2] if i + 2 < n else -1))
            else:
                sig.append((name,))
            i += lines
        else:
            i += 1
    return tuple(sig), ddr


def load(dirpath):
    files = sorted(f for f in os.listdir(dirpath) if f.endswith(".seq"))
    ops = []
    for fn in files:
        sig, ddr = parse(os.path.join(dirpath, fn))
        ops.append((fn, sig, ddr))
    return ops


def label_map(*streams):
    seen, order = {}, []
    def lab(n):
        s = ""; n += 1
        while n:
            n, r = divmod(n - 1, 26); s = chr(65 + r) + s
        return s
    for st in streams:
        for k in st:
            if k not in seen:
                seen[k] = lab(len(order)); order.append(k)
    return seen


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    A, B = load(a_dir), load(b_dir)
    sa = [hash(o[1]) for o in A]
    sb = [hash(o[1]) for o in B]
    lm = label_map(sa, sb)
    la = [lm[x] for x in sa]; lb = [lm[x] for x in sb]

    print(f"# {a_dir}: {len(A)} ops")
    print(f"# {b_dir}: {len(B)} ops")
    print(f"# distinct STRUCTURAL op types: {len(lm)}  (raw-byte gave ~hundreds)")
    print()
    ca, cb = Counter(la), Counter(lb)
    print("op   countA countB")
    for t in sorted(set(ca) | set(cb), key=lambda t: (len(t), t)):
        flag = "" if ca[t] == cb[t] else "  <-- differs"
        print(f"{t:<4} {ca[t]:>6} {cb[t]:>6}{flag}")

    print("\n## ordered STRUCTURAL alignment (first divergences) ##")
    sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
    printed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        seg_a = "".join(la[i1:i2]) or "-"
        seg_b = "".join(lb[j1:j2]) or "-"
        print(f"[{tag}] A[{i1}:{i2}]={seg_a[:40]}  B[{j1}:{j2}]={seg_b[:40]}")
        printed += 1
        if printed >= 10:
            break
    if printed == 0:
        print("NO structural divergence: both runs execute the same op TYPES in the same order.")
        print("=> interval-3 defect is NOT a wrong-op / wrong-shape; look at weight offsets below.")

    # weight-offset progression: total DDR patches + a per-op offset fingerprint
    def offs(ops):
        return [tuple(o[2]) for o in ops]  # per-op list of (arg_idx,arg_offset)
    oa, ob = offs(A), offs(B)
    print(f"\n## weight-offset (DDR-patch) view ##")
    print(f"# total DDR patches: A={sum(len(x) for x in oa)}  B={sum(len(x) for x in ob)}")
    # show the distinct arg_offset SET per run (are the same weight regions touched?)
    sa_off = set(off for x in oa for (_, off) in x)
    sb_off = set(off for x in ob for (_, off) in x)
    print(f"# distinct weight arg_offsets: A={len(sa_off)}  B={len(sb_off)}")
    only_b = sorted(sb_off - sa_off)[:20]
    only_a = sorted(sa_off - sb_off)[:20]
    print(f"# offsets in B(interval-3) but not A(interval-4): {len(sb_off - sa_off)} e.g. {only_b}")
    print(f"# offsets in A(interval-4) but not B(interval-3): {len(sa_off - sb_off)} e.g. {only_a}")


if __name__ == "__main__":
    main()
