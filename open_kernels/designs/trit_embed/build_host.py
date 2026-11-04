"""Build host/td_npu2.cpp, the persistent XRT C ABI that trit_npu.py drives through ctypes, against an XRT SDK.

    source target/npu-dev/env.sh                  # sets XILINX_XRT
    python open_kernels/designs/trit_embed/build_host.py     # -> build/host/libtd_npu2.so

td_npu2 adds asynchronous runs bound to chosen buffers (tdn_bind/tdn_start/tdn_wait), so the host
prepares one dispatch while the NPU executes the previous one.

The development build carries an rpath to the XRT SDK it was linked against. bundle.py builds the
shipped copy with --no-rpath (trit_npu.py then loads the system's XRT first), against a packaged XRT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'host/td_npu2.cpp'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xrt_sdk(xrt=None):
    """(XRT root, library directory) of the SDK to compile against: XILINX_XRT, else /opt/xilinx/xrt."""
    xrt = Path(xrt or os.environ.get('XILINX_XRT', '/opt/xilinx/xrt'))
    if not (xrt / 'include/xrt/xrt_device.h').exists():
        raise RuntimeError('Set XILINX_XRT to the installed XRT SDK')
    lib = next((p for p in (xrt / 'lib', xrt / 'lib64') if (p / 'libxrt_coreutil.so').exists()), None)
    if lib is None:
        raise RuntimeError('Cannot find libxrt_coreutil.so in XRT SDK')
    return xrt, lib


def compile_library(target, xrt=None, rpath=True, cxx=None):
    """Compile SOURCE into target; returns the build record (command, compiler, hashes)."""
    if os.name != 'posix':
        raise RuntimeError('This build helper targets Linux')
    xrt, lib = xrt_sdk(xrt)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    cxx = shlex.split(cxx or os.environ.get('CXX', 'g++'))
    command = cxx + [
        # -fno-math-errno: sqrtf compiles to the instruction alone (no libm call: glibc 2.43 versions sqrtf anew)
        '-std=c++17', '-shared', '-fPIC', '-O3', '-march=x86-64-v3', '-ffp-contract=off', '-fno-math-errno', str(SOURCE),
        '-I' + str(xrt / 'include'), '-L' + str(lib)] + (['-Wl,-rpath,' + str(lib)] if rpath else []) + [
        '-Wl,--as-needed', '-lxrt_coreutil', '-o', str(target)]
    subprocess.run(command, check=True)
    version = (xrt / 'include/xrt/detail/version-git.h')
    return {'command': command,
            'compiler': subprocess.run(cxx + ['--version'], capture_output=True, text=True).stdout.splitlines()[0],
            'xrt_sdk': str(xrt), 'xrt_version_header': version.read_text() if version.exists() else None,
            'rpath': rpath, 'source_sha256': digest(SOURCE), 'library_sha256': digest(target)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=ROOT / 'build/host/libtd_npu2.so')
    p.add_argument('--no-rpath', dest='rpath', action='store_false', help='no rpath to the XRT SDK (shipped copy)')
    a = p.parse_args()
    record = compile_library(a.out, rpath=a.rpath)
    (a.out.parent / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
