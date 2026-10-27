"""Build the GPT-OSS 20B decode primitives used by both FLM registry variants.

This is a kernel image bundle. It contains no model weights or token scheduler.
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

IMAGES = {
    "input_norm": ("gptoss_norm/gptoss_norm.py", {}),
    "qkv_projection_16_bands": ("gptoss_qkv_gemv/gptoss_qkv_gemv.py",
                                {"GPTOSS_QKV_BANDS": "16"}),
    "yarn_rope": ("gptoss_rope/gptoss_rope.py", {}),
    "sliding_sink_attention_128": ("gptoss_attention/gptoss_attention.py",
                                   {"GPTOSS_ATTN_ROWS": "128"}),
    "full_sink_attention_state_512": ("gptoss_attention_state/gptoss_attention_state.py",
                                      {"GPTOSS_ATTN_STATE_ROWS": "512"}),
    "attention_finalize_bf16": ("gptoss_attention_finalize/gptoss_attention_finalize.py", {}),
    "o_projection_16_bands": ("gptoss_o_gemv/gptoss_o_gemv.py", {}),
    "residual_bias": ("gptoss_residual/gptoss_residual.py", {}),
    "router_top4": ("gptoss_router/gptoss_router.py", {}),
    "expert_projection_90_bands": ("gptoss_expert_gemv/gptoss_expert_gemv.py",
                                   {"GEG_BANDS": "90"}),
    "expert_gate_45_blocks": ("gptoss_gate/gptoss_gate.py",
                              {"GPTOSS_GATE_BLOCKS": "45"}),
    "expert_mix_top4": ("gptoss_mix/gptoss_mix.py", {}),
    "lm_head_201088": ("lm_head_q4/lm_head_q4.py",
                       {"LMHEAD_N": "201088", "LMHEAD_K": "3072", "LMHEAD_CORES": "8"}),
}


def plan() -> dict:
    return {"format": "phlegm-gptoss-20b-kernels-v1", "images": {
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(plan(), indent=2))
    elif args.out is not None:
        built = build_all(args.out)
        print(f"built {len(built['images'])} GPT-OSS 20B kernel images")
    else:
        parser.error("--out is required unless --plan-only is set")
