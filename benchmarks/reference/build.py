#!/usr/bin/env python3
"""Build the resident reference bridge on Linux without modifying its checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def manifest_hash(files):
    content = "".join(f"{name}\0{files[name]}\n" for name in sorted(files))
    return hashlib.sha256(content.encode()).hexdigest()


def git_state(root):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()
    commit = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Cannot identify reference engine commit")
    return commit, bool(git("status", "--porcelain", "--untracked-files=no"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--xrt-include", type=Path)
    parser.add_argument("--xrt-lib", required=True, type=Path,
                        help="Exact libxrt_coreutil.so.2 (or development .so) file")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    adapter = Path(__file__).resolve().with_name("bridge.cpp")
    root = args.reference_root.resolve()
    engine = root / "src/open_qwen36"
    out = (args.out_dir or adapter.parents[2] / "target/reference-bench").resolve()
    xrt_include = (args.xrt_include or Path(os.environ.get(
        "XILINX_XRT", "/opt/xilinx/xrt")) / "include").resolve()
    xrt_lib = args.xrt_lib.resolve()
    binary = out / "phlegm-reference-bench"
    metadata = binary.with_suffix(".build.json")
    # Even failed prerequisite checks invalidate an earlier build's provenance.
    metadata.unlink(missing_ok=True)
    compiler = shutil.which(args.cxx)
    if compiler is None:
        raise RuntimeError(f"C++ compiler unavailable: {args.cxx}")
    compiler = Path(compiler).resolve()
    sources = [engine / name for name in (
        "core.cpp", "block_host.cpp", "pools.cpp", "manifest.cpp",
        "q4nx_file.cpp", "gguf_file.cpp")]
    helper = root / "open_kernels/harness/stream_patch.hpp"
    for path in [adapter, *sources, helper, xrt_lib,
                 xrt_include / "xrt/xrt_device.h",
                 root / "src/include/nlohmann/json.hpp"]:
        if not path.is_file():
            raise RuntimeError(f"Missing build input: {path}")

    def inputs():
        files = {"bridge/bridge.cpp": digest(adapter),
                 "build/build.py": digest(Path(__file__).resolve()),
                 "harness/stream_patch.hpp": digest(helper),
                 "link/" + xrt_lib.name: digest(xrt_lib),
                 "compiler/" + compiler.name: digest(compiler)}
        for base, prefix, suffixes in (
            (engine, "engine", {".cpp", ".hpp"}),
            (root / "src/include/nlohmann", "json", {".h", ".hpp"}),
            (xrt_include, "xrt", {".h", ".hpp"}),
        ):
            for path in base.rglob("*"):
                if path.is_file() and path.suffix in suffixes:
                    files[f"{prefix}/{path.relative_to(base).as_posix()}"] = digest(path)
        return files

    before = inputs()
    state = git_state(root)
    # Match the runtime's CMake Release settings. Only block_host disables
    # implicit FMA; changing core.cpp's contraction policy changes its logits.
    flags = ["-O3", "-DNDEBUG", "-std=c++17", "-fopenmp", "-mavx2", "-mfma"]
    common = [str(compiler), *flags]
    for include in (xrt_include, root / "src", root / "src/include", helper.parent):
        common += ["-I", str(include)]
    out.mkdir(parents=True, exist_ok=True)
    commands, objects = [], []
    for source in [*sources, adapter]:
        obj = out / (source.stem + ".o")
        extra = ["-ffp-contract=off"] if source.name == "block_host.cpp" else []
        command = [*common, *extra, "-c", str(source), "-o", str(obj)]
        subprocess.run(command, check=True, cwd=out)
        commands.append(command)
        objects.append(obj)
    command = [str(compiler), "-fopenmp", *map(str, objects), str(xrt_lib),
               "-Wl,-rpath," + str(xrt_lib.parent), "-o", str(binary)]
    subprocess.run(command, check=True, cwd=out)
    commands.append(command)
    if inputs() != before or git_state(root) != state:
        raise RuntimeError("Build inputs changed during compilation; rebuild required")
    build = dict(protocol=1, engine="openflowlm-cpp-reference",
                 source_commit=state[0], source_dirty=state[1],
                 source_tree_sha256=manifest_hash(before),
                 adapter_sha256=before["bridge/bridge.cpp"],
                 binary_sha256=digest(binary), compiler_flags="\n".join(map(shlex.join, commands)),
                 source_files=before)
    metadata.write_text(json.dumps(build, indent=2) + "\n")
    print(f"Built {binary}\nProvenance {metadata}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Reference bridge build failed: {error}", file=sys.stderr)
        sys.exit(1)
