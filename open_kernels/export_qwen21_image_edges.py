"""Compile Qwen Image 2.1's fixed 512-pixel edge streams as native NPU images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from open_kernels.qwen21_image_edges import specs


ROOT = Path(__file__).resolve().parent
DESIGNS = {"gemm": ROOT / "designs/dit_gemm/dit_gemm.py",
           "ew": ROOT / "designs/dit_ew/dit_ew.py"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    streams, files = specs(), {}
    for kind, entries in streams.items():
        for name, spec in entries.items():
            directory = root / name
            env = os.environ.copy()
            if kind == "gemm":
                env.update({f"DG_{key}": str(spec[key]) for key in ("M", "K", "N")})
                env["DG_LAYOUT"] = json.dumps(spec.get("layout", {}))
            else:
                env["DE_EL"] = "4096"
                env["DE_SPEC"] = json.dumps(spec)
            subprocess.run((sys.executable, str(ROOT / "build_design.py"),
                            str(DESIGNS[kind]), str(directory)),
                           env=env, check=True)
            for artifact in ("final.xclbin", "insts.bin", "insts.elf"):
                path = directory / artifact
                files[str(path.relative_to(root))] = {
                    "bytes": path.stat().st_size, "sha256": digest(path)}
            print(f"{kind}/{name} complete", flush=True)
    (root / "manifest.json").write_text(json.dumps({
        "family": "Qwen/Qwen-Image-2.1", "resolution": 512,
        "scope": "native transformer entry and final projection",
        "streams": streams, "files": files, "complete": True},
        indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
