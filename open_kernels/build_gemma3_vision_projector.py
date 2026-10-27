"""Build Gemma 3 4B's 256-token vision pool/norm and projection images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_design.py"
IMAGES = {
    "pool_norm_256x16x1152": ("gemma3_vision_pool_norm/gemma3_vision_pool_norm.py", {}),
    "projector_mm_m256_k1152_n2560": ("gemma4_mm/gemma4_mm.py",
                                      {"PHLEGM_M": "256", "PHLEGM_K": "1152",
                                       "PHLEGM_N": "2560"}),
}


def plan() -> dict:
    return {"format": "phlegm-gemma3-vision-projector-v1", "images": {
        name: {"design": design, "environment": env}
        for name, (design, env) in sorted(IMAGES.items())}}


def build_all(output: Path) -> dict:
    result = plan()
    output.mkdir(parents=True, exist_ok=False)
    for name, item in result["images"].items():
        destination = output / name
        env = os.environ.copy()
        env.update(item["environment"])
        source = HERE / "designs" / item["design"]
        subprocess.run((sys.executable, str(BUILD), str(source), str(destination)),
                       env=env, check=True)
        item["xclbin"] = f"{name}/final.xclbin"
        item["insts"] = f"{name}/insts.bin"
        item["xclbin_sha256"] = hashlib.sha256((destination / "final.xclbin").read_bytes()).hexdigest()
        item["insts_sha256"] = hashlib.sha256((destination / "insts.bin").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    if a.plan_only:
        print(json.dumps(plan(), indent=2))
    elif a.out is not None:
        built = build_all(a.out)
        print(f"built {len(built['images'])} Gemma 3 projector images")
    else:
        p.error("--out is required unless --plan-only is set")
