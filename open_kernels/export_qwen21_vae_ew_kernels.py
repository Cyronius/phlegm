"""Compile all pixelwise NPU streams needed by the 512-pixel Qwen VAE decoder."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from open_kernels.qwen21_vae_pack import physical
from open_kernels.qwen21_vae_plan import plan

ROOT = Path(__file__).resolve().parent
ELEMENT = {256: 1024, 384: 1536, 640: 1280, 1152: 2304}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def streams(model: Path) -> tuple[dict[str, dict], dict[str, str]]:
    topology = plan(model)
    specs: dict[str, dict] = {}
    operations: dict[str, str] = {}
    for op in topology["operations"]:
        if op["kind"] == "rmsnorm":
            kind = "rms" if ".attentions." in op["name"] else "rms_silu"
        elif op["kind"] == "add":
            kind = "add"
        else:
            continue
        h, c = op["height"], physical(op["channels"], False)
        el = ELEMENT[c]
        pitch = max(1, 128 // h) * (h + 2)
        size = (h + 2) * pitch * c
        spec = {"op": kind, "C": c, "H": h, "W": h, "el": el,
                "a": {"pitch": pitch}, "y": {"pitch": pitch},
                "sizes": {"X": size, "B": size if kind == "add" else 1,
                          "P": 1 if kind == "add" else 2 * el, "Y": size}}
        if kind == "add":
            spec["b"] = {"pitch": pitch}
        name = f"ew_{kind}_h{h}_c{c}"
        specs[name] = spec
        operations[op["name"]] = name
    from open_kernels.designs.qwen_vae_ew.qwen_vae_ew import check_spec
    for name, spec in specs.items():
        if why := check_spec(spec):
            raise ValueError(f"{name}: {why}")
    return specs, operations


def build_one(root: Path, name: str, spec: dict) -> tuple[str, dict]:
    target = root / name
    env = os.environ.copy()
    env["QVE_SPEC"] = json.dumps(spec)
    result = subprocess.run((sys.executable, str(ROOT / "build_design.py"),
                             str(ROOT / "designs/qwen_vae_ew/qwen_vae_ew.py"), str(target)),
                            env=env, capture_output=True, text=True)
    if result.returncode or "BUILD_OK" not in result.stdout:
        raise RuntimeError(f"{name} build failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    artifact = {part: {"bytes": (target / part).stat().st_size,
                       "sha256": digest(target / part)}
                for part in ("final.xclbin", "insts.bin", "insts.elf")}
    print(f"built {name}", flush=True)
    return name, artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    specs, operations = streams(args.model)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        artifacts = dict(pool.map(lambda item: build_one(root, *item), specs.items()))
    (root / "manifest.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "resolution": 512,
        "scope": "VAE pixelwise kernels only", "complete": True,
        "streams": specs, "operations": operations, "artifacts": artifacts,
    }, indent=2, sort_keys=True) + "\n")
    print(f"compiled {len(specs)} Qwen VAE pixelwise streams")


if __name__ == "__main__":
    main()
