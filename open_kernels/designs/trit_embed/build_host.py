"""Build host/td_npu2.cpp, the persistent XRT C ABI that trit_npu.py drives through ctypes, against an XRT SDK.

    source target/npu-dev/env.sh                  # sets XILINX_XRT
    python open_kernels/designs/trit_embed/build_host.py     # -> build/host/libtd_npu2.so

td_npu2 adds asynchronous runs bound to chosen buffers (tdn_bind/tdn_start/tdn_wait), so the host
prepares one dispatch while the NPU executes the previous one."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if os.name != 'posix':
        raise RuntimeError('This build helper targets Linux')
    xrt = Path(os.environ.get('XILINX_XRT', '/opt/xilinx/xrt'))
    if not (xrt / 'include/xrt/xrt_device.h').exists():
        raise RuntimeError('Set XILINX_XRT to the installed XRT SDK')
    lib = next((p for p in (xrt / 'lib', xrt / 'lib64') if (p / 'libxrt_coreutil.so').exists()), None)
    if lib is None:
        raise RuntimeError('Cannot find libxrt_coreutil.so in XRT SDK')
    out = ROOT / 'build/host'
    out.mkdir(parents=True, exist_ok=True)
    source, target = ROOT / 'host/td_npu2.cpp', out / 'libtd_npu2.so'
    command = shlex.split(os.environ.get('CXX', 'g++')) + [
        '-std=c++17', '-shared', '-fPIC', '-O3', '-march=x86-64-v3', '-ffp-contract=off', str(source), '-I' + str(xrt / 'include'),
        '-L' + str(lib), '-Wl,-rpath,' + str(lib), '-lxrt_coreutil', '-o', str(target)]
    subprocess.run(command, check=True)
    (out / 'build.json').write_text(json.dumps({'command': command, 'source_sha256': digest(source),
                                                'library_sha256': digest(target)}, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
