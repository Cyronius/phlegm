"""Inventory FLM's registry against Phlegm NPU kernel source routes.

This is a source inventory, not a model-support claim. It does not compile or
run kernels, check container packing, or validate image/audio preprocessing.
Pass a pinned FLM ``model_list.json`` to make the result reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "open_kernels"))
from recipes.families import FAMILIES  # noqa: E402

# Registry labels are not always recipe names. A mapping only says that some
# kernel source route exists; it does not imply compatible model geometry.
RECIPE_ROUTE = {
    "deepseek-r1": "llama3", "deepseek-r1-0528": "qwen3",
    "gemma3": "gemma3", "gemma3-text": "gemma3",
    "lfm2": "lfm2", "lfm2.5-tk": "lfm2",
    "llama3": "llama3", "nanbeige": "llama3", "phi4": "phi3",
    "qwen2": "qwen2", "qwen2vl": "qwen2",
    "qwen3": "qwen3", "qwen3-it": "qwen3", "qwen3-tk": "qwen3",
    "qwen3vl": "qwen3", "qwen3.5": "qwen35",
    "qwen3.6-moe": "qwen36moe",
}
SPECIAL_ROUTE = {
    "bge": "bert", "gte": "bert", "minilm": "bert", "nomic": "bert",
    "embed-gemma": "embedding-gemma-matmul",
    "flux2-klein": "diffusion", "whisper-v3": "whisper",
}
NO_ROUTE = {"gemma4e", "gemma4-12b", "gpt-oss"}
VISION_TEXT_ONLY = {"gemma3", "qwen2vl", "qwen3vl", "qwen3.5", "qwen3.6-moe"}
ALIAS_NEEDS_CONFIG = {"lfm2.5-tk", "nanbeige", "deepseek-r1", "deepseek-r1-0528",
                      "qwen3-it", "qwen3-tk"}
PACKAGE_DIR_ALIASES = {"embed-gemma:300m": "Embedding-Gemma-300M-OpenNPU2"}


def inventory(registry: Path, xclbins_root: Path | None = None) -> dict:
    raw = registry.read_bytes()
    models = json.loads(raw)["models"]
    bert = json.loads((ROOT / "npu_offload/gemm_rtp/families.json").read_text())
    bert_tags = {tag for family in bert["families"] for tag in family["serves"]}
    rows = []
    for tag, variants in models.items():
        for variant, model in variants.items():
            name = f"{tag}:{variant}"
            family = model["details"]["family"]
            if family in RECIPE_ROUTE:
                route = RECIPE_ROUTE[family]
                if route not in FAMILIES:
                    raise ValueError(f"{name}: recipe {route} is absent")
                state = "recipe_source"
                gaps = []
                if family in ALIAS_NEEDS_CONFIG:
                    gaps.append("verify model config and packed geometry")
                if model.get("vlm") and family in VISION_TEXT_ONLY:
                    gaps.append("vision tower kernel route not established")
                if name == "qwen3.8:27b":
                    gaps.append("wide DeltaNet and FFN recipe is not buildable")
            elif family in SPECIAL_ROUTE:
                route = SPECIAL_ROUTE[family]
                state = "special_source"
                gaps = []
                if route == "bert" and name not in bert_tags:
                    raise ValueError(f"{name}: missing from BERT design families")
                if route == "whisper" and not (ROOT / "open_kernels/export_whisper_kernels.py").is_file():
                    raise ValueError("Whisper exporter is absent")
                if route == "diffusion" and not (ROOT / "open_kernels/export_dit_kernels.py").is_file():
                    raise ValueError("diffusion exporter is absent")
                if route == "embedding-gemma-matmul":
                    if not (ROOT / "npu_offload/matmul/export_gemma.py").is_file():
                        raise ValueError("Embedding-Gemma matmul exporter is absent")
                    gaps.append("embedding runtime route and model validation remain open")
            elif family in NO_ROUTE:
                route = None
                state = "missing_recipe"
                gaps = ["no complete NPU recipe"]
                if family == "gpt-oss":
                    gaps.append("NPU-checked router and MoE arithmetic lack a composed layer recipe")
            else:
                raise ValueError(f"{name}: unreviewed FLM family {family!r}")
            images = None
            if xclbins_root is not None:
                image_dir = xclbins_root / PACKAGE_DIR_ALIASES.get(name, model["name"])
                images = sorted(p.name for p in image_dir.rglob("*.xclbin"))
            rows.append({"model": name, "family": family, "route": route,
                         "state": state, "gaps": gaps,
                         "closed_image_names": images})
    return {"registry_sha256": hashlib.sha256(raw).hexdigest(),
            "models": len(rows), "families": len({r["family"] for r in rows}),
            "states": dict(Counter(r["state"] for r in rows)),
            "closed_image_names": sorted({image for r in rows
                                          for image in (r["closed_image_names"] or [])}),
            "rows": rows}


def markdown(data: dict) -> str:
    lines = [
        "# FLM model kernel source matrix",
        "",
        "Generated by `utilities/audit-flm-kernel-coverage.py` from a pinned FLM registry.",
        f"Registry SHA256: `{data['registry_sha256']}`.",
        f"Inventory: {data['models']} model entries, {data['families']} family labels.",
        "",
        "`recipe_source` means a possible text kernel recipe exists; `special_source`",
        "means a specialized embedding, audio or diffusion exporter exists.",
        "Neither state asserts that this exact model has compiled, run or passed",
        "quality checks. `missing_recipe` means no complete open NPU recipe exists.",
        "A packaged FLM xclbin name is a reference to a closed binary, not a",
        "Phlegm kernel implementation or a compatible ABI.",
        "",
        "| FLM model | Family label | Phlegm route | Source state | Work still needed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in data["rows"]:
        gaps = "; ".join(row["gaps"]) or "model-specific build and validation"
        lines.append(f"| `{row['model']}` | `{row['family']}` | "
                     f"`{row['route'] or '—'}` | `{row['state']}` | {gaps} |")
    missing = [row for row in data["rows"] if row["state"] == "missing_recipe"]
    lines += ["", "## Packaged FLM image names for models without recipes", "",
              "Names identify primitive classes to investigate. They do not specify",
              "a shared shape or runtime ABI.", "",
              "| FLM model | Packaged image basenames |", "| --- | --- |"]
    for row in missing:
        images = ", ".join(f"`{x}`" for x in row["closed_image_names"] or [])
        lines.append(f"| `{row['model']}` | {images or 'No packaged image directory in this checkout'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--xclbins-root", type=Path,
                   help="optional FLM packaged xclbins root; list basenames, not ABIs")
    p.add_argument("--json", type=Path, help="write the full inventory as JSON")
    p.add_argument("--markdown", type=Path, help="write a complete Markdown matrix")
    args = p.parse_args()
    data = inventory(args.registry, args.xclbins_root)
    if args.json:
        args.json.write_text(json.dumps(data, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown(data))
    print(f"{data['models']} models, {data['families']} FLM labels; "
          f"registry SHA256 {data['registry_sha256']}")
    if args.xclbins_root:
        print(f"{len(data['closed_image_names'])} distinct packaged xclbin basenames "
              "for registry models (names do not establish ABI equivalence)")
    for row in data["rows"]:
        print(f"{row['model']:<29} {row['state']:<15} "
              f"{(row['route'] or '-'):12} {'; '.join(row['gaps'])}")


if __name__ == "__main__":
    main()
