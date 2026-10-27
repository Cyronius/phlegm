"""Build Gemma 3 SigLIP projection bias/activation NPU images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN = HERE / "designs/gemma3_vision_bias_activation/gemma3_vision_bias_activation.py"
IMAGES = {
    "qkv_bias_n3584": {"PHLEGM_LOGICAL": "3456", "PHLEGM_PADDED": "3584", "PHLEGM_GELU": "0"},
    "attention_out_bias_n1280": {"PHLEGM_LOGICAL": "1152", "PHLEGM_PADDED": "1280", "PHLEGM_GELU": "0"},
    "fc1_bias_gelu_n4352": {"PHLEGM_LOGICAL": "4304", "PHLEGM_PADDED": "4352", "PHLEGM_GELU": "1"},
    "fc2_bias_n1280": {"PHLEGM_LOGICAL": "1152", "PHLEGM_PADDED": "1280", "PHLEGM_GELU": "0"},
}


def plan() -> dict:
    return {"format": "phlegm-gemma3-vision-post-mm-v1", "images": {
        name: {"design": str(DESIGN.relative_to(HERE)), "environment": env}
        for name, env in sorted(IMAGES.items())}}


def build_all(out: Path) -> dict:
    result = plan()
    out.mkdir(parents=True, exist_ok=False)
    for name, item in result["images"].items():
        destination = out / name
        env = os.environ.copy()
        env.update(item["environment"])
        subprocess.run((sys.executable, str(HERE / "build_design.py"),
                        str(DESIGN), str(destination)), env=env, check=True)
        item["xclbin"] = f"{name}/final.xclbin"
        item["insts"] = f"{name}/insts.bin"
        item["xclbin_sha256"] = hashlib.sha256((destination / "final.xclbin").read_bytes()).hexdigest()
        item["insts_sha256"] = hashlib.sha256((destination / "insts.bin").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    if a.plan_only:
        print(json.dumps(plan(), indent=2))
    elif a.out is not None:
        print(f"built {len(build_all(a.out)['images'])} SigLIP bias/activation images")
    else:
        p.error("--out is required unless --plan-only is set")
