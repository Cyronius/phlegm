"""Map each distinct control-ELF to its NPU kernel (xclbin) by replay.

For one representative op per ELF, bind all its exact captured buffers and run
against each xclbin; only the matching kernel completes (state 4). Builds the
elf->kernel inventory for a token's forward.

Usage: python map_kernels.py C:/caps/m0c
"""
import sys, os, subprocess, re
from collections import defaultdict

D = sys.argv[1]
EXE = os.path.join(os.path.dirname(__file__), "..", "..", "npu-engine", "m0", "out", "m0_replay.exe")
XDIR = "C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2"
XCLBINS = ["mm", "dequant_mm", "conv", "attn", "GateDeltaNet_prefill", "layer", "lm_head"]

ev = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(D, "events.tsv"))]
elf_idx, mod_elf, ker_mod, run_ker = {}, {}, {}, {}
ops = {}
for a in ev:
    k = a[1]
    if k == "ELF": elf_idx[a[5]] = a[2]
    elif k == "MODULE": mod_elf[a[2]] = a[3]
    elif k == "KERNEL": ker_mod[a[2]] = a[3]
    elif k == "RUN": run_ker[a[2]] = a[3]
    elif k == "START":
        ker = run_ker.get(a[2]); mod = ker_mod.get(ker) if ker else None
        elf = mod_elf.get(mod) if mod else None
        ops[int(a[0])] = {"elf": elf_idx.get(elf) if elf else None, "args": []}
    elif k == "RUNARG":
        sev = int(a[2])
        if sev in ops:
            ops[sev]["args"].append({"arg": int(a[3]), "size": int(a[5]),
                                     "hash": a[6], "dumped": a[7] == "1"})

# one fully-dumped op per elf
by_elf = {}
for sev, op in ops.items():
    if op["elf"] is None or not op["args"]: continue
    if not all(x["dumped"] for x in op["args"]): continue
    by_elf.setdefault(op["elf"], (sev, op))

print(f"{len(by_elf)} distinct fully-dumped ELFs to map\n")
elf_kernel = {}
for elf, (sev, op) in sorted(by_elf.items()):
    args = []
    for x in op["args"]:
        args += ["--in", str(x["arg"]), str(x["size"]), f"{D}/blob_{x['size']}_{x['hash']}.bin"]
    found = None
    for xb in XCLBINS:
        cmd = [EXE, f"{XDIR}/{xb}.xclbin", f"{D}/elf_{int(elf):06d}.bin"] + args
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=25).stdout
        except subprocess.TimeoutExpired:
            continue
        m = re.search(r"run state = (\d+)", out)
        if m and m.group(1) == "4":
            found = xb; break
    elf_kernel[elf] = found
    print(f"  elf idx{elf}: {'-> ' + found if found else '(no xclbin completed)'}  (nargs={len(op['args'])}, op#{sev})")

# count ops per kernel across the whole token
per_kernel = defaultdict(int)
for sev, op in ops.items():
    if op["elf"] in elf_kernel and elf_kernel[op["elf"]]:
        per_kernel[elf_kernel[op["elf"]]] += 1
print("\nkernel op-counts across the captured token:")
for kb, n in sorted(per_kernel.items(), key=lambda x: -x[1]):
    print(f"  {kb}: {n}")
