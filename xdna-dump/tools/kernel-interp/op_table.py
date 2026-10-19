"""Dump the complete op table from a capture's events.tsv (m0d format, RUNARG blobs).

For each run: start_ev, elf idx, per-arg (bo, size, pre-hash) and the first D2H
after start per bo (post-hash) -> classify in/out/in-place.
Usage: python op_table.py C:/caps/m0d [--csv out.csv]
"""
import sys, os
from collections import defaultdict

D = sys.argv[1]
ev = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(D, "events.tsv"))]

elf_idx, mod_elf, ker_mod, run_ker = {}, {}, {}, {}
d2h = defaultdict(list)
h2d = defaultdict(list)
bind_evs = defaultdict(list)   # bo -> [ev of every RUNARG bind]
ops = {}
elf_size = {}

for a in ev:
    k = a[1]
    if k == "ELF":
        elf_idx[a[5]] = int(a[2]); elf_size[int(a[2])] = int(a[3])
    elif k == "MODULE": mod_elf[a[2]] = a[3]
    elif k == "KERNEL": ker_mod[a[2]] = a[3]
    elif k == "RUN":    run_ker[a[2]] = a[3]
    elif k == "D2H":    d2h[a[2]].append((int(a[0]), a[3], int(a[4]), a[5]))
    elif k == "H2D":    h2d[a[2]].append((int(a[0]), a[3], int(a[4]), a[5]))
    elif k == "START":
        run = a[2]
        ker = run_ker.get(run); mod = ker_mod.get(ker) if ker else None
        elf = mod_elf.get(mod) if mod else None
        ops[int(a[0])] = {"start": int(a[0]), "elf": elf_idx.get(elf), "args": []}
    elif k == "RUNARG":
        sev = int(a[2])
        if sev in ops:
            ops[sev]["args"].append(
                {"arg": int(a[3]), "bo": a[4], "size": int(a[5]), "hash": a[6]})
            bind_evs[a[4]].append((int(a[0]), sev))

def next_bind(bo, after):
    """first bind of this bo by a LATER op (its own RUNARGs don't count)"""
    c = [e for e, s in bind_evs[bo] if s != after and e > after]
    return min(c) if c else 10**18

rows = []
for sev in sorted(ops):
    op = ops[sev]
    parts = []
    for a in op["args"]:
        # post state: first D2H of this bo after start but before its next bind
        nb = next_bind(a["bo"], sev)
        post = [x for x in d2h.get(a["bo"], []) if sev < x[0] < nb]
        post = min(post, key=lambda x: x[0]) if post else None
        if post and post[3] != a["hash"]:
            role = "OUT*" if a["hash"].strip("0") else "OUT"   # OUT* = in-place (nonzero pre)
        elif post:
            role = "unch"
        else:
            role = "in"
        ph = post[3] if post else ""
        parts.append((a["arg"], role, a["size"], a["hash"], ph))
    rows.append((sev, op["elf"], parts))

print(f"{len(rows)} ops")
for sev, elf, parts in rows:
    s = " | ".join(f"a{arg} {role} {sz} pre={h[:8]}" + (f" post={ph[:8]}" if ph and ph != h else "")
                   for arg, role, sz, h, ph in parts)
    es = elf_size.get(elf, 0)
    print(f"op@{sev:<6} elf={elf if elf is not None else '?':>4} ({es:>7}B)  {s}")
