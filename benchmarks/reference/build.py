#!/usr/bin/env python3
"""Build Phlegm's C++ bridge; also supports external trees during migration."""
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


INPUT_PATHS = ["open_kernels/harness/stream_patch.hpp", "benchmarks/reference/bridge.cpp",
               "benchmarks/reference/build.py", "phlegm-core/build.rs",
               "phlegm-core/build_bridge.rs", "phlegm-core/Cargo.toml"]


def git_state(root):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()
    commit = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Cannot identify reference engine commit")
    paths = list(INPUT_PATHS)
    paths.extend(path for path in git("ls-files", "--", "src/open_qwen36", "src/include/nlohmann").splitlines()
                 if Path(path).suffix in {".cpp", ".hpp", ".h"})
    for base in (root / "src/open_qwen36", root / "src/include/nlohmann"):
        paths.extend(path.relative_to(root).as_posix() for path in base.rglob("*")
                     if path.is_file() and path.suffix in {".cpp", ".hpp", ".h"})
    return commit, bool(git("status", "--porcelain", "--untracked-files=all", "--", *sorted(set(paths))))


def engine_inputs(root, adapter):
    files = {"bridge/bridge.cpp": digest(adapter),
             "harness/stream_patch.hpp": digest(root / "open_kernels/harness/stream_patch.hpp")}
    for base, prefix in ((root / "src/open_qwen36", "engine"),
                         (root / "src/include/nlohmann", "json")):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".cpp", ".hpp", ".h"}:
                files[f"{prefix}/{path.relative_to(base).as_posix()}"] = digest(path)
    return files


def loader_finds(library):
    """Use the configured loader cache, never the build shell's LD_LIBRARY_PATH."""
    try:
        cache = subprocess.check_output(["/sbin/ldconfig", "-p"], text=True)
    except (OSError, subprocess.CalledProcessError):
        cache = ""
    for line in cache.splitlines():
        if line.strip().startswith("libxrt_coreutil.so.2 ") and "=>" in line:
            if Path(line.split("=>", 1)[1].strip()).resolve() == library.resolve():
                return True
    return any((Path(p) / library.name).resolve() == library.resolve()
               for p in ("/lib", "/lib64", "/usr/lib", "/usr/lib64"))


def link_specs(compiler, version):
    # Conda's installed specs append an absolute environment rpath. Override
    # only its link_command with GCC's built-in stanza; arithmetic flags and
    # all compilation specs stay intact. No extra ELF-editing tool is needed.
    if "conda" not in version.lower():
        return None
    builtin = subprocess.check_output([str(compiler), "-dumpspecs"], text=True)
    stanza = re.search(r"(?ms)^\*link_command:\n(.*?)(?=\n\n|\Z)", builtin)
    if not stanza or "rpath" in stanza[1]:
        raise RuntimeError("Cannot obtain a conda GCC link command without an injected rpath")
    return "*link_command:\n" + stanza[1] + "\n\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--xrt-include", type=Path)
    parser.add_argument("--xrt-lib", required=True, type=Path,
                        help="Exact libxrt_coreutil.so.2 (or development .so) file")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--expected-engine-tree", help=argparse.SUPPRESS)
    args = parser.parse_args()
    adapter = Path(__file__).resolve().with_name("bridge.cpp")
    root = args.reference_root.resolve()
    engine = root / "src/open_qwen36"
    out = (args.out_dir or adapter.parents[2] / "target/reference-bench").resolve()
    xrt_include = (args.xrt_include or Path(os.environ.get(
        "XILINX_XRT", "/opt/xilinx/xrt")) / "include").resolve()
    # Preserve the SONAME symlink name; link with -l:libxrt_coreutil.so.2.
    xrt_lib = args.xrt_lib.absolute()
    binary = out / "phlegm-reference-bench"
    metadata = binary.with_suffix(".build.json")
    # Even failed prerequisite checks invalidate an earlier build's provenance.
    metadata.unlink(missing_ok=True)
    compiler = shutil.which(args.cxx)
    if compiler is None:
        raise RuntimeError(f"C++ compiler unavailable: {args.cxx}")
    # Preserve argv[0]: distro g++ may be a ccache symlink which dispatches by name.
    compiler = Path(compiler).absolute()
    compiler_version = subprocess.check_output([str(compiler), "--version"], text=True).strip()
    specs = link_specs(compiler, compiler_version)
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
        files = {**engine_inputs(root, adapter),
                 "build/build.py": digest(Path(__file__).resolve()),
                 "link/" + xrt_lib.name: digest(xrt_lib),
                 "compiler/" + compiler.name: digest(compiler)}
        for base, prefix, suffixes in (
            (xrt_include / "xrt", "xrt/xrt", {".h", ".hpp"}),
        ):
            for path in base.rglob("*"):
                if path.is_file() and path.suffix in suffixes:
                    files[f"{prefix}/{path.relative_to(base).as_posix()}"] = digest(path)
        for name in ("build.rs", "build_bridge.rs", "Cargo.toml"):
            path = adapter.parents[2] / "phlegm-core" / name
            files[f"build/{name}"] = digest(path)
        if specs is not None:
            files["compiler/link-no-rpath.specs"] = hashlib.sha256(specs.encode()).hexdigest()
        return files

    before = inputs()
    state = git_state(root)
    tree = manifest_hash(engine_inputs(root, adapter))
    if args.expected_engine_tree and args.expected_engine_tree != tree:
        raise RuntimeError("Engine inputs differ from the Rust build; rebuild required")
    tags = subprocess.check_output(["git", "-C", str(root), "tag", "--points-at", "HEAD"], text=True).splitlines()
    release_tag = next((tag for tag in tags if re.fullmatch(r"v\d+\.\d+\.\d+(?:[-+].+)?", tag)), None)
    # Match the runtime's CMake Release settings. Only block_host disables
    # implicit FMA; changing core.cpp's contraction policy changes its logits.
    flags = ["-O3", "-DNDEBUG", "-std=c++17", "-fopenmp", "-mavx2", "-mfma"]
    common = [str(compiler), *flags]
    for include in (xrt_include, root / "src", root / "src/include", helper.parent):
        common += ["-I", str(include)]
    out.mkdir(parents=True, exist_ok=True)
    specs_path = out / "link-no-rpath.specs"
    if specs is not None:
        specs_path.write_text(specs)
    commands, objects = [], []
    for source in [*sources, adapter]:
        obj = out / (source.stem + ".o")
        extra = ["-ffp-contract=off"] if source.name == "block_host.cpp" else []
        command = [*common, *extra, "-c", str(source), "-o", str(obj)]
        subprocess.run(command, check=True, cwd=out)
        commands.append(command)
        objects.append(obj)
    rpath = None if loader_finds(xrt_lib) else str(xrt_lib.parent)
    command = [str(compiler), *(["-specs=" + str(specs_path)] if specs is not None else []),
               "-fopenmp", *map(str, objects),
               "-L", str(xrt_lib.parent), "-l:libxrt_coreutil.so.2",
               "-Wl,-rpath-link," + str(xrt_lib.parent),
               *(["-Wl,-rpath," + rpath] if rpath else []), "-o", str(binary)]
    subprocess.run(command, check=True, cwd=out)
    commands.append(command)
    if inputs() != before or git_state(root) != state:
        raise RuntimeError("Build inputs changed during compilation; rebuild required")
    build = dict(protocol=1, engine="openflowlm-cpp-reference",
                 source_commit=state[0], source_dirty=state[1],
                 engine_tree_sha256=tree, release_tag=release_tag,
                 xrt_rpath=rpath,
                 compiler_version=compiler_version,
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
