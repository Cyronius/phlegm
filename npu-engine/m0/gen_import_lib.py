"""Generate an MSVC import lib for the real xrt_coreutil.dll (mangled C++ exports),
so our C++ probe can link xrt::device/xclbin/hw_context and bind to the system XRT.
Usage: python gen_import_lib.py <dumpbin_exports.txt> <out.def>"""
import sys, re

exports_txt, out_def = sys.argv[1], sys.argv[2]
names = []
# dumpbin -exports lines look like:  "   41   28 00002A80 ??0elf@xrt@@QEAA@PEBX_K@Z"
row = re.compile(r"^\s*(\d+)\s+([0-9A-Fa-f]+)\s+[0-9A-Fa-f]+\s+(\S+)")
row_noaddr = re.compile(r"^\s*(\d+)\s+([0-9A-Fa-f]+)\s+(\S+)\s*$")  # forwarded/no-RVA
for ln in open(exports_txt, encoding="utf-8", errors="replace"):
    m = row.match(ln) or row_noaddr.match(ln)
    if not m:
        continue
    ordn, name = m.group(1), m.group(3)
    if name in ("[NONAME]", "name"):
        continue
    names.append((int(ordn), name))

with open(out_def, "w", encoding="utf-8") as f:
    f.write("LIBRARY xrt_coreutil\nEXPORTS\n")
    for ordn, name in names:
        f.write(f"    {name} @{ordn}\n")
print(f"wrote {out_def} with {len(names)} exports")
