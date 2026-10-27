"""Build Gemma 4 vision/audio non-matrix NPU primitives into one manifest.

This bundle covers kernel shapes only. It does not export real model weights or
provide a scheduler. Matrix images are built by build_gemma4_multimodal_mm.py.
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
BUILD = HERE / "build_design.py"

OPERATIONS = {
    "vision_head_norm_12x64": ("gemma4_head_norm/gemma4_head_norm.py", {"PHLEGM_HEADS": "12", "PHLEGM_HEAD_DIM": "64"}),
    "vision_axial_rope_24x64": ("gemma4_vision_rope/gemma4_vision_rope.py", {}),
    "vision_online_attn_1024x64": ("gemma4_attention/gemma4_attention.py", {"PHLEGM_HEAD_DIM": "64", "PHLEGM_ATTN_ROWS": "1024"}),
    "vision_online_attn_472x64": ("gemma4_attention/gemma4_attention.py", {"PHLEGM_HEAD_DIM": "64", "PHLEGM_ATTN_ROWS": "472"}),
    "vision_hidden_norm_768": ("gemma4_hidden_norm/gemma4_hidden_norm.py", {"PHLEGM_HIDDEN": "768"}),
    "vision_gelu_mul_3072": ("gemma4_gelu_mul/gemma4_gelu_mul.py", {"PHLEGM_WIDTH": "3072"}),
    "audio_depthwise_256x1024": ("gemma4_audio_dw/gemma4_audio_dw.py", {}),
    "audio_relative_attn_8x12x24": ("gemma4_audio_attn/gemma4_audio_attn.py", {}),
    "audio_hidden_norm_1024": ("gemma4_hidden_norm/gemma4_hidden_norm.py", {"PHLEGM_HIDDEN": "1024"}),
    "audio_subnorm_128": ("gemma4_audio_subnorm/gemma4_audio_subnorm.py", {"PHLEGM_CHANNELS": "128"}),
    "audio_subnorm_32": ("gemma4_audio_subnorm/gemma4_audio_subnorm.py", {"PHLEGM_CHANNELS": "32"}),
    "audio_glu_1024": ("gemma4_audio_act/gemma4_audio_act.py", {"PHLEGM_WIDTH": "1024", "PHLEGM_GLU": "1"}),
    "audio_silu_1024": ("gemma4_audio_act/gemma4_audio_act.py", {"PHLEGM_WIDTH": "1024", "PHLEGM_GLU": "0"}),
    "audio_silu_4096": ("gemma4_audio_act/gemma4_audio_act.py", {"PHLEGM_WIDTH": "4096", "PHLEGM_GLU": "0"}),
}


def plan() -> dict:
    return {"format": "phlegm-gemma4-multimodal-ops-v1", "images": {
        name: {"design": design, "environment": env}
        for name, (design, env) in sorted(OPERATIONS.items())}}


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(plan(), indent=2))
    elif args.out is not None:
        built = build_all(args.out)
        print(f"built {len(built['images'])} Gemma 4 vision/audio operation images")
    else:
        parser.error("--out is required unless --plan-only is set")
