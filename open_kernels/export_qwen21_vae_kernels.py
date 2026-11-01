"""Compile the 512-pixel Qwen Image 2.1 decoder's unique convolution streams.

The resulting kernels cover the checkpoint-weighted convolution geometry;
the complete VAE runtime and its single-head attention are separate work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from open_kernels.designs.dit_conv.dit_conv import resolve_spec
from open_kernels.qwen21_vae_pack import physical
from open_kernels.qwen21_vae_plan import plan


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def streams(model: Path) -> tuple[dict, dict]:
    topology = plan(model)
    specs: dict[str, dict] = {}
    operations: dict[str, str] = {}
    for op in topology["operations"]:
        if op["kind"] != "conv":
            continue
        up = ".upsampler.resample.1" in op["name"]
        h = op["height"] // (2 if up else 1)
        spec = {"H": h, "W": h,
                "Cin": physical(op["input_channels"], True),
                "Cout": physical(op["channels"], False), "up": up}
        taps = op["taps"] ** 2
        resolve_spec(spec, taps)
        name = f"conv_h{h}_ci{spec['Cin']}_co{spec['Cout']}_k{op['taps']}_up{int(up)}"
        if name in specs and specs[name] != {"spec": spec, "taps": taps}:
            raise ValueError(f"kernel name collision: {name}")
        specs[name] = {"spec": spec, "taps": taps}
        operations[op["name"]] = name
    return specs, operations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="compile only the first N unique streams")
    args = parser.parse_args()
    specs, operations = streams(args.model)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    chosen = list(specs.items())[:args.limit]
    for index, (name, item) in enumerate(chosen):
        env = os.environ.copy()
        env["DC_TAPS"] = str(item["taps"])
        env["DC_SPEC"] = json.dumps(item["spec"])
        subprocess.run((sys.executable, str(ROOT / "build_design.py"),
                        str(ROOT / "designs/dit_conv/dit_conv.py"), str(root / name)),
                       env=env, check=True)
        artifacts[name] = {part: {"bytes": (root / name / part).stat().st_size,
                                  "sha256": digest(root / name / part)}
                           for part in ("final.xclbin", "insts.bin", "insts.elf")}
        print(f"{index + 1}/{len(chosen)} {name}", flush=True)
    (root / "manifest.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "scope": "VAE convolution kernels only",
        "resolution": 512, "complete": args.limit is None,
        "streams": specs, "operations": operations, "artifacts": artifacts},
        indent=2, sort_keys=True) + "\n")
    print(json.dumps({"streams": len(chosen), "operations": len(operations),
                      "complete": args.limit is None}))


if __name__ == "__main__":
    main()
