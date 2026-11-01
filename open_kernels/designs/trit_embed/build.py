"""Compile the trit-embed encoder into build/ (the shipped graph) or build_<tag>/ (profiling variants).

    source target/npu-dev/env.sh                  # Linux; any mlir-aie 1.4.2 + Peano env works
    python open_kernels/designs/trit_embed/build.py
    python open_kernels/designs/trit_embed/build.py --stub attend,linear   # -> build_L3_cap128_d2_stub-attend+linear

build.json records the program-text size (16 KiB per core is the ceiling) and the source and artifact
hashes that trit_npu.py checks before loading the graph.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cap', type=int, choices=(8, 32, 128), default=128)
    p.add_argument('--layers', type=int, choices=(2, 3), default=3)
    p.add_argument('--stub', default='', help="'all', or comma-separated kernels (attend,linear,...) that return immediately")
    p.add_argument('--bdepth', type=int, choices=(1, 2, 3), default=2, help='Broadcast FIFO depth (8192-byte elements)')
    p.add_argument('--defs', default='', help='Comma-separated kernel macros, e.g. F2_PROF_NOEPI')
    p.add_argument('--opt', default='', choices=('', '0', '1', '2', '3'), help='aiecc -O level for core code')
    a = p.parse_args()
    os.environ.update(TRIT_CAP=str(a.cap), TRIT_LAYERS=str(a.layers), TRIT_STUB=a.stub,
                      TRIT_BDEPTH=str(a.bdepth), TRIT_OPT=a.opt, TRIT_DEFS=a.defs)
    packager = os.environ.get('AIE_XCLBINUTIL') or shutil.which('xclbinutil')
    if not packager:
        raise RuntimeError('Set AIE_XCLBINUTIL or put xclbinutil on PATH')
    os.environ['AIE_XCLBINUTIL'] = str(packager)
    import aie.iron as iron
    from aie.iron.device import from_name
    iron.set_current_device(from_name('npu2', n_cols=8))
    from trit_design import DESIGN, SPECIALIZE, SOURCES, PACKETS, ESZ
    # Keep failed compiler products for diagnosing the 16 KiB core budget.
    import aie.utils.compile.jit.compilabledesign as compiler_module
    compiler_module._cleanup_failed_compilation = lambda path: None
    tag = (f'L{a.layers}_cap{a.cap}_d{a.bdepth}' + (f'_O{a.opt}' if a.opt else '')
           + ''.join('_' + m.replace('F2_PROF_', '').replace('=', '').lower() for m in a.defs.split(',') if m)
           + (f"_stub-{a.stub.replace(',', '+')}" if a.stub else ''))
    out = ROOT / ('build' if tag == 'L3_cap128_d2' else f'build_{tag}')
    out.mkdir(exist_ok=True)
    work = (out / 'final.prj').resolve()
    if work.exists():
        shutil.rmtree(work)
    start = time.perf_counter()
    DESIGN.specialize(**SPECIALIZE).compile(xclbin_path=str(out / 'final.xclbin'), inst_path=str(out / 'insts.bin'))
    size_tool = Path(os.environ['PEANO_INSTALL_DIR']) / ('bin/llvm-size.exe' if os.name == 'nt' else 'bin/llvm-size')
    sections = subprocess.check_output([str(size_tool), '-A', str(work / 'elfs_main_core_0_2/elfs_main_core_0_2.elf')], text=True)
    text_bytes = int(re.search(r'^\.text\s+(\d+)', sections, re.MULTILINE)[1])
    report = {'tag': tag, 'stub': a.stub, 'broadcast_depth': a.bdepth, 'defs': [m for m in a.defs.split(',') if m], 'packets': PACKETS, 'element_bytes': ESZ, 'packet_format': 2, 'aiecc_opt': a.opt,
              'cap': a.cap, 'texts_per_dispatch': 128 // a.cap, 'cores': 32, 'layers': a.layers,
              'scope': f'One dispatch, {a.layers} full encoder layers; host embedding lookup/input norm/output projection/pooling',
              'program_text_bytes': text_bytes, 'stack_bytes_per_core': 0x1400,
              'mlir_aie': importlib.metadata.version('mlir_aie'), 'llvm_aie': importlib.metadata.version('llvm-aie'),
              'source_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in SOURCES},
              'artifact_sha256': {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ('final.xclbin', 'insts.bin')},
              'build_seconds': time.perf_counter() - start}
    (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
