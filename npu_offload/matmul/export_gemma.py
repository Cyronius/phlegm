"""Compile the twelve Embedding-Gemma NPU matmuls into a checked source set.

The marker is written last. Generated binaries stay in the requested output
directory, normally below target/; source control retains only the design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from gemma_shapes import SHAPES, tag

HERE = Path(__file__).resolve().parent
SOURCES = [HERE / name for name in
           ("matmul_whole_array.py", "gemma_shapes.py", "export_gemma.py")]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(out: Path, force: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "gemma_kernels.json"
    source_hashes = {p.name: digest(p) for p in SOURCES}
    if marker.is_file() and not force:
        old = json.loads(marker.read_text())
        if old.get("complete") and old.get("sources") == source_hashes and all(
            (out / f"{tag(s)}.{ext}").is_file() and
            old.get("artifacts", {}).get(tag(s), {}).get(f"{ext}_sha256") ==
            digest(out / f"{tag(s)}.{ext}")
            for s in SHAPES for ext in ("xclbin", "insts")
        ):
            return old
    marker.unlink(missing_ok=True)
    artifacts = {}
    for shape in SHAPES:
        m, k, n = shape
        name = tag(shape)
        # 1152 is divisible by 64 * 2, but not by 64 * 4.
        aie_cols = 2 if n == 1152 else 4
        xclbin = out / f"{name}.xclbin"
        insts = out / f"{name}.insts"
        cmd = [sys.executable, str(HERE / "matmul_whole_array.py"),
               "-M", str(m), "-K", str(k), "-N", str(n),
               "--tile-n", "64", "--n-aie-cols", str(aie_cols), "--dtype-out", "f32",
               "--dev", "npu2", "--xclbin-path", str(xclbin),
               "--insts-path", str(insts)]
        print(f"building {name}", flush=True)
        subprocess.run(cmd, check=True)
        artifacts[name] = {"shape": [m, k, n], "tile_n": 64, "aie_cols": aie_cols,
                           "output": "f32", "xclbin_sha256": digest(xclbin),
                           "insts_sha256": digest(insts)}
    manifest = {"format": "phlegm-embedding-gemma-kernels-v1", "complete": True,
                "sources": source_hashes, "artifacts": artifacts}
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    result = export(a.out.resolve(), a.force)
    print(f"complete: {len(result['artifacts'])} Embedding-Gemma NPU shapes")


if __name__ == "__main__":
    main()
