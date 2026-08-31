"""Reconstruct discrete kernel ops from the unified events.tsv, so one op can be
replayed. Groups SETARG by run pointer, links each bound bo to its nearest sync
(H2D=input / D2H=output) and dumped bytes, and picks the ELF that precedes the run.

events.tsv columns (tab-sep, first col = global event #):
  <ev> ELF    <elfidx> <size> <hash>
  <ev> H2D|D2H <bo_ptr> <boidx> <size> <hash>
  <ev> SETARG <run_ptr> <argidx> <bo_ptr>
  <ev> START  <run_ptr>
Usage: python analyze_events.py C:/caps/m0
"""
import sys, os
from collections import defaultdict

D = sys.argv[1]
rows = []
for ln in open(os.path.join(D, "events.tsv")):
    a = ln.rstrip("\n").split("\t")
    if len(a) >= 2:
        rows.append(a)

# index syncs by bo pointer -> list of (ev, dir, boidx, size, hash)
syncs = defaultdict(list)
elfs = []          # (ev, elfidx, size, hash)
for a in rows:
    ev = int(a[0]); kind = a[1]
    if kind in ("H2D", "D2H"):
        syncs[a[2]].append((ev, kind, a[3], int(a[4]), a[5]))
    elif kind == "ELF":
        elfs.append((ev, a[2], int(a[3]), a[4]))

# group SETARG into runs: a run is a maximal run of SETARGs for the same run_ptr
# whose argidx is non-decreasing restart (3,4,5,6,7). A new "3" starts a new op.
ops = []
cur = None
for a in rows:
    ev = int(a[0])
    if a[1] == "SETARG":
        run, argidx, bo = a[2], int(a[3]), a[4]
        if argidx == 3 or cur is None or cur["run"] != run:
            if cur: ops.append(cur)
            cur = {"run": run, "ev0": ev, "args": {}}
        cur["args"][argidx] = bo
    elif a[1] == "START" and cur is not None:
        cur["start_ev"] = ev
if cur: ops.append(cur)

def nearest_sync(bo, ev0):
    """the sync for bo closest in time to the op's binding."""
    best = None
    for s in syncs.get(bo, []):
        d = abs(s[0] - ev0)
        if best is None or d < best[0]:
            best = (d, s)
    return best[1] if best else None

# pick ELF preceding each op
def elf_before(ev0):
    cand = [e for e in elfs if e[0] < ev0]
    return cand[-1] if cand else (elfs[0] if elfs else None)

print(f"reconstructed {len(ops)} ops from {len(rows)} events, {len(elfs)} elfs\n")
for i, op in enumerate(ops[:8]):
    elf = elf_before(op["ev0"])
    print(f"op#{i} run={op['run']} ev0={op['ev0']} elf=idx{elf[1] if elf else '?'}({elf[2] if elf else 0}B)")
    for arg in sorted(op["args"]):
        bo = op["args"][arg]
        s = nearest_sync(bo, op["ev0"])
        if s:
            role = "IN " if s[1] == "H2D" else "OUT"
            dumped = os.path.exists(os.path.join(D, f"{s[2]}.bo"))
            print(f"   arg{arg} bo={bo} {role} size={s[3]} boidx={s[2]} dumped={dumped}")
        else:
            print(f"   arg{arg} bo={bo} (no sync seen)")

# find the smallest fully-dumped op (all args have a dumped sync) as replay target
def op_ok(op):
    for arg, bo in op["args"].items():
        s = nearest_sync(bo, op["ev0"])
        if not s or not os.path.exists(os.path.join(D, f"{s[2]}.bo")):
            return False, 0
    tot = sum(nearest_sync(bo, op["ev0"])[3] for bo in op["args"].values())
    return True, tot

cands = [(i, op_ok(op)[1]) for i, op in enumerate(ops) if op_ok(op)[0]]
cands.sort(key=lambda x: x[1])
print(f"\nfully-captured ops (all args dumped): {len(cands)}")
if cands:
    print(f"smallest replay candidate: op#{cands[0][0]} total_bytes={cands[0][1]}")
