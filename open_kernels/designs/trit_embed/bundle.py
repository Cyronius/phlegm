"""Put the prebuilt NPU encoder into a model folder, so it runs without mlir-aie, Peano or a phlegm checkout.

    source target/npu-dev/env.sh
    python open_kernels/designs/trit_embed/build.py              # the graph, in build/
    python open_kernels/designs/trit_embed/bundle.py path/to/trit-embed-3l

This writes, into the model folder:

    trit_npu.py           a copy of the host module (it finds npu/ next to itself)
    npu/final.xclbin      the graph, from build/
    npu/insts.bin         its runtime sequence, from build/
    npu/libtd_npu2.so     the XRT host library, compiled here
    npu/build.json        build/build.json plus "bundle": toolchain, ABI audit and phlegm commit;
                          artifact_sha256 also covers libtd_npu2.so

The host library is compiled with the system g++ (--cxx) against a packaged XRT (--xrt, default /usr; the headers of
XRT's -devel package), without an rpath: at run time trit_npu.py loads the system's libxrt_coreutil.so.2 first. The
audit fails the bundle if the library has an rpath or needs newer symbol versions than GLIBC_2.32, GLIBCXX_3.4.30 or
CXXABI_1.3.13 (glibc 2.32 and GCC 12's libstdc++; Ubuntu 22.04, Debian 12 and Fedora 36 and later have them).
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from build_host import compile_library  # noqa: E402

POLICY = {'GLIBC': (2, 32), 'GLIBCXX': (3, 4, 30), 'CXXABI': (1, 3, 13), 'GCC': (7, 0)}
LIBRARY = 'libtd_npu2.so'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def audit(library):
    """NEEDED entries, rpath, the newest imported symbol version per family and the XRT symbols imported."""
    dynamic = run('readelf', '-d', str(library))
    needed = re.findall(r'\(NEEDED\)\s+Shared library: \[(.+?)\]', dynamic)
    paths = re.findall(r'\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[(.+?)\]', dynamic)
    imports = [line for line in run('objdump', '-T', str(library)).splitlines() if '*UND*' in line]
    newest = {}
    for family in POLICY:
        versions = [tuple(map(int, v.split('.'))) for v in re.findall(rf'\b{family}_([0-9.]+)\b', '\n'.join(imports))]
        if versions:
            newest[family] = max(versions)
    xrt = sorted(set(run('c++filt', *[l.split()[-1] for l in imports if 'xrt' in l.split()[-1]]).splitlines()))
    problems = [f'rpath {paths}'] if paths else []
    problems += [f'{f}_{".".join(map(str, v))} is newer than {f}_{".".join(map(str, POLICY[f]))}'
                 for f, v in newest.items() if v > POLICY[f]]
    return {'needed': needed, 'rpath': paths,
            'newest_symbol_versions': {f: f'{f}_{".".join(map(str, v))}' for f, v in newest.items()},
            'xrt_symbols': xrt}, problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model_dir', type=Path)
    p.add_argument('--build', type=Path, default=ROOT / 'build')
    p.add_argument('--xrt', default='/usr', help='XRT SDK to compile against (include/xrt, lib64/libxrt_coreutil.so)')
    p.add_argument('--cxx', default='g++', help='C++ compiler for the host library (not a conda one: it adds an rpath)')
    a = p.parse_args()
    manifest = json.loads((a.build / 'build.json').read_text())
    if manifest.get('stub') or manifest.get('defs') or manifest.get('tag') != 'L3_cap128_d2':
        sys.exit(f'{a.build} is a variant or profiling build ({manifest.get("tag")}); bundle the shipped graph')
    for name, sha in manifest['source_sha256'].items():
        if digest(ROOT / name) != sha:
            sys.exit(f'Source {name} changed since {a.build} was built; rebuild (build.py)')
    for name, sha in manifest['artifact_sha256'].items():
        if digest(a.build / name) != sha:
            sys.exit(f'Changed build artifact {name}')
    if not (a.model_dir / 'trit_embed.py').exists():
        sys.exit(f'{a.model_dir} is not a trit-embed model folder (no trit_embed.py)')

    out = a.model_dir / 'npu'
    out.mkdir(exist_ok=True)
    host = compile_library(out / LIBRARY, xrt=a.xrt, rpath=False, cxx=a.cxx)
    abi, problems = audit(out / LIBRARY)
    if problems:
        (out / LIBRARY).unlink()
        sys.exit('The host library is not portable: ' + '; '.join(problems))
    for name in ('final.xclbin', 'insts.bin'):
        shutil.copyfile(a.build / name, out / name)
    shutil.copyfile(ROOT / 'trit_npu.py', a.model_dir / 'trit_npu.py')

    git = lambda *args: run('git', '-C', str(ROOT), *args).strip()
    packager = shutil.which('xclbinutil')
    manifest['artifact_sha256'][LIBRARY] = digest(out / LIBRARY)
    manifest['bundle'] = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'phlegm_commit': git('rev-parse', 'HEAD'),
        'phlegm_tree_dirty': bool(git('status', '--porcelain', '--', '.', '../../include')),
        'design': 'open_kernels/designs/trit_embed',
        'kernel_toolchain': {'mlir_aie': manifest['mlir_aie'], 'llvm_aie (Peano)': manifest['llvm_aie'],
                             'xclbinutil': run(packager, '--version').strip().splitlines()[:3] if packager else None},
        'host_library': {**host, **abi},
        'trit_npu_py_sha256': digest(a.model_dir / 'trit_npu.py'),
        'runtime_requires': 'Linux x86-64 (AVX2), XDNA2 NPU (PCI 1022:17f0) on amdxdna, XRT with the amdxdna plugin '
                            '(built against 2.27.0), RLIMIT_MEMLOCK of at least 64 MiB, '
                            + ', '.join(abi['newest_symbol_versions'].values()),
    }
    (out / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for path in sorted([*out.iterdir(), a.model_dir / 'trit_npu.py']):
        print(f'{digest(path)}  {path.stat().st_size:>8}  {path}')
    if manifest['bundle']['phlegm_tree_dirty']:
        print('warning: the design has uncommitted changes; commit before a release bundle', file=sys.stderr)


if __name__ == '__main__':
    main()
