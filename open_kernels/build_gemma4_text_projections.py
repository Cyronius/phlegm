"""Compile every distinct Gemma 4 text projection image from a config.json.

Run the build command inside `dibs run build`. This produces NPU images and a
geometry manifest; packing source model weights and scheduling layers remain
separate work. `--plan-only` reads the configuration without compiling.
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
from open_kernels.gemma4_text_plan import from_hf_config, projection_image_geometry, projections  # noqa: E402

DESIGN = HERE / "designs" / "lm_head_q4" / "lm_head_q4.py"
BF16_DESIGN = HERE / "designs" / "gemma4_bf16_gemv" / "gemma4_bf16_gemv.py"
BUILD = HERE / "build_design.py"


def image_plan(config: dict) -> dict:
    plan = from_hf_config(config)
    slots = projections(plan)
    images = {}
    for projection in slots:
        rows, cols, calls = projection_image_geometry(projection)
        tag = f"{projection.format}_n{rows}_k{cols}" if projection.format == "bf16" else f"n{rows}_k{cols}"
        images[tag] = {"rows": rows, "image_cols": cols, "format": projection.format,
                       "cores": min(8, rows // 64) if projection.format == "q4" else 1}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    return {"variant": plan.variant, "config_sha256": config_hash, "hidden": plan.hidden,
            "layers": len(plan.layers), "vocab": plan.vocab,
            "projection_count": len(slots),
            "images": {tag: images[tag] for tag in sorted(images)},
            "projections": [{"layer": p.layer, "name": p.name,
                             "rows": p.rows, "cols": p.cols, "format": p.format,
                             "image": (("bf16_" if p.format == "bf16" else "") +
                                       f"n{projection_image_geometry(p)[0]}_k{projection_image_geometry(p)[1]}"),
                             "calls": projection_image_geometry(p)[2]}
                            for p in slots]}


def build_all(config: dict, output: Path) -> dict:
    manifest = image_plan(config)
    output.mkdir(parents=True, exist_ok=False)
    for tag, image in manifest["images"].items():
        destination = output / tag
        env = os.environ.copy()
        if image["format"] == "q4":
            env.update(LMHEAD_N=str(image["rows"]), LMHEAD_K=str(image["image_cols"]),
                       LMHEAD_CORES=str(image["cores"]))
            design = DESIGN
        else:
            env.update(PHLEGM_N=str(image["rows"]), PHLEGM_K=str(image["image_cols"]))
            design = BF16_DESIGN
        subprocess.run((sys.executable, str(BUILD), str(design), str(destination)),
                       env=env, check=True)
        image["xclbin"] = f"{tag}/final.xclbin"
        image["insts"] = f"{tag}/insts.bin"
        image["xclbin_sha256"] = hashlib.sha256((destination / "final.xclbin").read_bytes()).hexdigest()
        image["insts_sha256"] = hashlib.sha256((destination / "insts.bin").read_bytes()).hexdigest()
    manifest["format"] = "phlegm-gemma4-text-projections-v2"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
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
        print(f"built {len(result['images'])} images for {result['variant']}")
