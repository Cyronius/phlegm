"""Build the complete Gemma 4 *text-kernel* image inventory for one config.

This compiles NPU arithmetic images only. It does not pack model weights or
provide a runtime/model recipe. Run the build inside `dibs run build` after
loading the NPU toolchain environment. `--plan-only` is read-only.
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
sys.path.insert(0, str(HERE.parent))
from open_kernels.build_gemma4_text_projections import build_all as build_projections  # noqa: E402
from open_kernels.build_gemma4_text_projections import image_plan as projection_plan  # noqa: E402
from open_kernels.gemma4_text_plan import from_hf_config  # noqa: E402


def image_plan(config: dict) -> dict:
    text = from_hf_config(config)
    images = {}

    def add(tag: str, design: str, **env):
        images[tag] = {"design": design, "env": {key: str(value) for key, value in env.items()}}

    for heads, dim in sorted({(l.query_heads, l.head_dim) for l in text.layers} |
                             {(l.kv_heads, l.head_dim) for l in text.layers if l.computes_kv}):
        add(f"head_norm_h{heads}_d{dim}", "gemma4_head_norm/gemma4_head_norm.py",
            PHLEGM_HEADS=heads, PHLEGM_HEAD_DIM=dim)
    for heads, dim in sorted({(l.query_heads + l.kv_heads, l.head_dim) for l in text.layers}):
        add(f"rope_h{heads}_d{dim}", "gemma4_rope/gemma4_rope.py",
            PHLEGM_HEADS=heads, PHLEGM_HEAD_DIM=dim)
    for dim in sorted({l.head_dim for l in text.layers}):
        for rows in (1, 128, 512, 1024):
            add(f"attention_d{dim}_r{rows}", "gemma4_attention/gemma4_attention.py",
                PHLEGM_HEAD_DIM=dim, PHLEGM_ATTN_ROWS=rows)
    add("hidden_norm", "gemma4_hidden_norm/gemma4_hidden_norm.py",
        PHLEGM_HIDDEN=text.hidden)
    for width in sorted({l.mlp_width for l in text.layers}):
        add(f"mlp_gelu_{width}", "gemma4_gelu_mul/gemma4_gelu_mul.py",
            PHLEGM_WIDTH=width)
    if text.layers[0].per_layer_input_width:
        add("ple_gate", "gemma4_ple_gate/gemma4_ple_gate.py")
        for token in (0, 1):
            add(f"ple_input_token{token}", "gemma4_ple_input/gemma4_ple_input.py",
                PHLEGM_HIDDEN=text.hidden, PHLEGM_HAS_TOKEN=token)
    add("lm_head", "lm_head_q4/lm_head_q4.py",
        LMHEAD_N=text.vocab, LMHEAD_K=text.hidden)
    add("logit_cap", "gemma4_logit_cap/gemma4_logit_cap.py")
    return {"variant": text.variant, "text_images": {tag: images[tag] for tag in sorted(images)},
            "projections": projection_plan(config)}


def build_all(config: dict, output: Path) -> dict:
    plan = image_plan(config)
    output.mkdir(parents=True, exist_ok=False)
    projection_manifest = build_projections(config, output / "projections")
    for tag, entry in plan["text_images"].items():
        destination = output / tag
        env = os.environ.copy()
        env.update(entry["env"])
        source = HERE / "designs" / entry["design"]
        subprocess.run((sys.executable, str(HERE / "build_design.py"),
                        str(source), str(destination)), env=env, check=True)
        entry["xclbin"] = f"{tag}/final.xclbin"
        entry["insts"] = f"{tag}/insts.bin"
        entry["xclbin_sha256"] = hashlib.sha256((destination / "final.xclbin").read_bytes()).hexdigest()
        entry["insts_sha256"] = hashlib.sha256((destination / "insts.bin").read_bytes()).hexdigest()
    plan["format"] = "phlegm-gemma4-text-kernels-v1"
    plan["projections"] = projection_manifest
    (output / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.plan_only:
        print(json.dumps(image_plan(config), indent=2))
    else:
        if args.out is None:
            parser.error("--out is required unless --plan-only is set")
        result = build_all(config, args.out)
        print(f"built {len(result['text_images'])} text arithmetic images and "
              f"{len(result['projections']['images'])} projection images for {result['variant']}")
