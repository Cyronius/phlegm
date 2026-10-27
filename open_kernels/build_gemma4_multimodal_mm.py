"""Build Gemma 4 vision/audio BF16 matrix images from FLM q4nx headers.

Reads JSON headers only. Both FLM's [N/64,K/256,16384] tiles and ordinary
[N,K] BF16 matrices use the same NPU GEMM after host weight layout conversion.
The image handles one padded 256-row activation panel; a model scheduler still
needs to arrange patches or audio frames, attention, normalization and conv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN = HERE / "designs" / "gemma4_mm" / "gemma4_mm.py"
BUILD = HERE / "build_design.py"


def image_plan(headers: list[dict], names: list[str]) -> dict:
    images: dict[str, dict] = {}
    for source_name, header in zip(names, headers, strict=True):
        for tensor, item in header.items():
            if item.get("dtype") != "BF16":
                continue
            shape = item.get("shape", [])
            if len(shape) == 3 and shape[2] == 16384:
                rows, cols, layout = shape[0] * 64, shape[1] * 256, "flm_64x256"
            elif len(shape) == 2 and min(shape) >= 256:
                rows, cols, layout = shape[0], shape[1], "row_major"
            else:
                continue
            if rows % 256 or cols % 64:
                raise ValueError(f"untiled Gemma 4 matrix {tensor}: {shape}")
            tag = f"m256_n{rows}_k{cols}"
            image = images.setdefault(tag, {"m": 256, "n": rows, "k": cols, "sources": []})
            image["sources"].append({"header": source_name, "tensor": tensor,
                                     "layout": layout, "shape": shape})
    for rows, cols, operation in ((256, 64, "audio_subsample_conv0"),
                                  (256, 1152, "audio_subsample_conv1")):
        tag = f"m256_n{rows}_k{cols}"
        image = images.setdefault(tag, {"m": 256, "n": rows, "k": cols, "sources": []})
        image["sources"].append({"operation": operation, "layout": "im2col_bf16"})
    return {"format": "phlegm-gemma4-multimodal-mm-v2",
            "images": {tag: images[tag] for tag in sorted(images)}}


def build_all(headers: list[dict], names: list[str], output: Path) -> dict:
    plan = image_plan(headers, names)
    output.mkdir(parents=True, exist_ok=False)
    for tag, image in plan["images"].items():
        destination = output / tag
        env = os.environ.copy()
        env.update(PHLEGM_M=str(image["m"]), PHLEGM_K=str(image["k"]),
                   PHLEGM_N=str(image["n"]))
        subprocess.run((sys.executable, str(BUILD), str(DESIGN), str(destination)),
                       env=env, check=True)
        image["xclbin"] = f"{tag}/final.xclbin"
        image["insts"] = f"{tag}/insts.bin"
        image["xclbin_sha256"] = hashlib.sha256((destination / "final.xclbin").read_bytes()).hexdigest()
        image["insts_sha256"] = hashlib.sha256((destination / "insts.bin").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--header", required=True, type=Path, action="append")
    p.add_argument("--out", type=Path)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    headers = [json.loads(path.read_text()) for path in a.header]
    names = [path.name for path in a.header]
    if a.plan_only:
        print(json.dumps(image_plan(headers, names), indent=2))
    else:
        if a.out is None:
            p.error("--out is required unless --plan-only is set")
        result = build_all(headers, names, a.out)
        print(f"built {len(result['images'])} Gemma 4 vision/audio matrix images")
