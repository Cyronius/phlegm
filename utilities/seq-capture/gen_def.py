"""Minimal PE export-table reader (pure stdlib). Prints exported symbols
(name, ordinal) of a Windows DLL so we can (a) size a proxy/forwarder .def
and (b) spot the control-code / kernel-submit entry points to intercept."""
import struct, sys, re

def u16(b, o): return struct.unpack_from('<H', b, o)[0]
def u32(b, o): return struct.unpack_from('<I', b, o)[0]

def read_exports(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:2] != b'MZ':
        raise SystemExit("not MZ")
    e_lfanew = u32(data, 0x3C)
    if data[e_lfanew:e_lfanew+4] != b'PE\x00\x00':
        raise SystemExit("not PE")
    coff = e_lfanew + 4
    num_sections = u16(data, coff + 2)
    opt_size = u16(data, coff + 16)
    opt = coff + 20
    magic = u16(data, opt)
    is_pe32p = (magic == 0x20b)  # PE32+ (64-bit)
    # Data directory starts after the fixed optional header.
    # PE32: 96 bytes fixed; PE32+: 112 bytes fixed, before the directories.
    dd_off = opt + (112 if is_pe32p else 96)
    export_rva = u32(data, dd_off + 0)
    export_size = u32(data, dd_off + 4)
    # Section headers, to map RVA->file offset
    sec_off = opt + opt_size
    sections = []
    for i in range(num_sections):
        s = sec_off + i*40
        name = data[s:s+8].rstrip(b'\x00').decode('latin1')
        vsize = u32(data, s+8); vaddr = u32(data, s+12)
        rawsize = u32(data, s+16); rawptr = u32(data, s+20)
        sections.append((vaddr, vsize, rawptr, rawsize, name))
    def rva2off(rva):
        for vaddr, vsize, rawptr, rawsize, name in sections:
            if vaddr <= rva < vaddr + max(vsize, rawsize):
                return rawptr + (rva - vaddr)
        return None
    eo = rva2off(export_rva)
    ordinal_base = u32(data, eo + 16)
    num_funcs = u32(data, eo + 20)
    num_names = u32(data, eo + 24)
    addr_funcs = rva2off(u32(data, eo + 28))
    addr_names = rva2off(u32(data, eo + 32))
    addr_ords  = rva2off(u32(data, eo + 36))
    # function RVAs (for detecting forwarders, which point inside export dir)
    func_rvas = [u32(data, addr_funcs + i*4) for i in range(num_funcs)]
    names = []
    for i in range(num_names):
        name_rva = u32(data, addr_names + i*4)
        no = rva2off(name_rva)
        end = data.index(b'\x00', no)
        nm = data[no:end].decode('latin1')
        ordv = u16(data, addr_ords + i*2)
        frva = func_rvas[ordv] if ordv < len(func_rvas) else 0
        is_fwd = export_rva <= frva < export_rva + export_size
        names.append((ordinal_base + ordv, nm, is_fwd))
    return names, num_funcs, num_names

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else r'C:\Windows\System32\xrt_coreutil.dll'
    names, nf, nn = read_exports(path)
    print(f"# {path}")
    print(f"# exported functions: {nf}, named exports: {nn}")
    # Highlight likely control-code / submit / bo interception points.
    hot = re.compile(r'(run|kernel|elf|module|module_int|ext.*kernel|bo\b|buffer|xclbin|hw_context|txn|ctrl|ert|start|wait|exec|submit)', re.I)
    print("\n## candidate interception symbols (submit / ctrlcode / bo) ##")
    shown = 0
    for ordv, nm, fwd in names:
        # crude demangle hint: MSVC mangles as ?name@class@@...
        pretty = nm
        if hot.search(nm):
            print(f"@{ordv:<5} {'FWD' if fwd else '   '} {pretty[:180]}")
            shown += 1
    print(f"\n# {shown} candidate symbols of {nn} total")
