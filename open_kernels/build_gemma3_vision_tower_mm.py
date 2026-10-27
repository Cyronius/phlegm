"""Build the Gemma 3 4B SigLIP transformer LayerNorm and projection images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = {
    "layernorm_256x1152": ("gemma3_vision_layernorm/gemma3_vision_layernorm.py", {}),
    "qkv_m256_k1152_n3584": ("gemma4_mm/gemma4_mm.py",
                               {"PHLEGM_M": "256", "PHLEGM_K": "1152", "PHLEGM_N": "3584"}),
    "attention_out_m256_k1152_n1280": ("gemma4_mm/gemma4_mm.py",
                                          {"PHLEGM_M": "256", "PHLEGM_K": "1152", "PHLEGM_N": "1280"}),
    "mlp_fc1_m256_k1152_n4352": ("gemma4_mm/gemma4_mm.py",
                                   {"PHLEGM_M": "256", "PHLEGM_K": "1152", "PHLEGM_N": "4352"}),
    "mlp_fc2_m256_k4352_n1280": ("gemma4_mm/gemma4_mm.py",
                                   {"PHLEGM_M": "256", "PHLEGM_K": "4352", "PHLEGM_N": "1280"}),
}


def plan() -> dict:
    return {"format": "phlegm-gemma3-vision-tower-mm-v1", "images": {
        name: {"design": design, "environment": env}
        for name, (design, env) in sorted(IMAGES.items())}}


def build_all(out: Path) -> dict:
    result = plan()
    out.mkdir(parents=True, exist_ok=False)
    for name, item in result["images"].items():
        destination = out / name
        env = os.environ.copy()
        env.update(item["environment"])
        subprocess.run((sys.executable, str(HERE / "build_design.py"),
                        str(HERE / "designs" / item["design"]), str(destination)),
                       env=env, check=True)
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
        print(f"built {len(build_all(a.out)['images'])} Gemma 3 SigLIP tower images")
    else:
        p.error("--out is required unless --plan-only is set")
