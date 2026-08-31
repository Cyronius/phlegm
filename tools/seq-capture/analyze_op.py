"""Deterministic op reconstruction from events.tsv (with object-graph + RUNARG).

Links each run to its EXACT elf via run->kernel->module->elf, uses RUNARG (exact
buffer bytes dumped at run::start) for inputs, and D2H syncs for expected outputs.
Emits a ready-to-run m0_replay command for a chosen non-zero-output op.

events.tsv:
  <ev> ELF    <idx> <size> <hash> <elf_self>
  <ev> MODULE <module_self> <elf_ptr>
  <ev> KERNEL <kernel_self> <module_ptr>
  <ev> RUN    <run_self> <kernel_ptr>
  <ev> SETARG <run> <arg> <bo>
  <ev> START  <run>
  <ev> RUNARG <start_ev> <arg> <bo> <size> <hash> <dumped>
  <ev> H2D|D2H <bo> <boidx> <size> <hash>
Usage: python analyze_op.py C:/caps/m0c
"""
import sys, os
from collections import defaultdict

D = sys.argv[1]
ev = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(D, "events.tsv"))]

elf_idx = {}      # elf_self -> elfidx
mod_elf = {}      # module_self -> elf_ptr
ker_mod = {}      # kernel_self -> module_ptr
run_ker = {}      # run_self -> kernel_ptr (latest)
d2h = defaultdict(list)   # bo -> list of (ev, boidx, size, hash)
ops = {}          # start_ev -> op

for a in ev:
    k = a[1]
    if k == "ELF":       elf_idx[a[5]] = a[2]
    elif k == "MODULE":  mod_elf[a[2]] = a[3]
    elif k == "KERNEL":  ker_mod[a[2]] = a[3]
    elif k == "RUN":     run_ker[a[2]] = a[3]
    elif k == "D2H":     d2h[a[2]].append((int(a[0]), a[3], int(a[4]), a[5]))
    elif k == "START":
        run = a[2]
        ker = run_ker.get(run); mod = ker_mod.get(ker) if ker else None
        elf = mod_elf.get(mod) if mod else None
        eidx = elf_idx.get(elf) if elf else None
        ops[int(a[0])] = {"start": int(a[0]), "run": run, "elf": eidx, "args": []}
    elif k == "RUNARG":
        sev = int(a[2])
        if sev in ops:
            ops[sev]["args"].append({"arg": int(a[3]), "bo": a[4], "size": int(a[5]),
                                     "hash": a[6], "dumped": a[7] == "1"})

def out_after(bo, sev):
    """expected output = first D2H of this bo strictly after the run start."""
    cand = [x for x in d2h.get(bo, []) if x[0] > sev]
    return min(cand, key=lambda x: x[0]) if cand else None

def zero_hash(sz):  # fnv1a of sz zero-bytes, to spot all-zero buffers
    h = 1469598103934665603
    for _ in range(sz): h = ((h ^ 0) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"
# cache a few common sizes
ZERO = {}
def is_zero(sz, h):
    if sz not in ZERO: ZERO[sz] = zero_hash(sz) if sz <= 16*1024*1024 else None
    return ZERO[sz] == h

good = []  # ops with an ELF, all inputs dumped, and a NON-ZERO expected output
for sev, op in ops.items():
    if op["elf"] is None or not op["args"]: continue
    ins, outs = [], []
    for a in op["args"]:
        o = out_after(a["bo"], sev)
        if o: outs.append((a["arg"], o))      # (arg, (ev,boidx,size,hash))
        else: ins.append(a)
    if not outs: continue
    nz_out = [o for o in outs if not is_zero(o[1][2], o[1][3])]
    ins_ok = all(a["dumped"] for a in ins)
    out_dumped = all(os.path.exists(os.path.join(D, f"{o[1][1]}.bo")) for o in outs)
    if nz_out and ins_ok and out_dumped:
        tot = sum(a["size"] for a in ins) + sum(o[1][2] for o in outs)
        good.append((tot, sev, op, ins, outs))

good.sort()
print(f"reconstructed {len(ops)} ops; {len(good)} with exact-elf + dumped inputs + non-zero output\n")
for tot, sev, op, ins, outs in good[:3]:
    print(f"op start_ev={sev} elf=idx{op['elf']} total={tot}")
    for a in ins:  print(f"  IN  arg{a['arg']} size={a['size']} file=run_{sev}_a{a['arg']}.bin hash={a['hash']}")
    for arg, o in outs:
        z = "ZERO" if is_zero(o[2], o[3]) else "DATA"
        print(f"  OUT arg{arg} size={o[2]} expect={o[1]}.bo hash={o[3]} [{z}]")

if good:
    tot, sev, op, ins, outs = good[0]
    XB = "C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2"
    parts = [f'"{XB}/XCLBIN.xclbin"', f'"{D}/elf_{int(op["elf"]):06d}.bin"']
    for a in ins:  parts.append(f'--in {a["arg"]} {a["size"]} "{D}/run_{sev}_a{a["arg"]}.bin"')
    for arg, o in outs:
        role = "--out" if not is_zero(o[2], o[3]) else "--zero"
        if role == "--out": parts.append(f'--out {arg} {o[2]} "{D}/{o[1]}.bo"')
        else: parts.append(f'--zero {arg} {o[2]}')
    print("\nREPLAY (swap XCLBIN for each candidate):")
    print("m0_replay.exe " + " ".join(parts))
