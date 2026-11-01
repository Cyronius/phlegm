"""Run the 32 checkpoint-weighted Qwen Image 2.1 DiT blocks on the NPU.

This is a transformer-stack gate using supplied or seeded embedded tokens and
shared modulation. It does not perform image/text entry, final projection,
sampling or VAE decoding. The first block's kernel shapes are reused for all
32 blocks; each layer loads its own packed weights and q/k norm vectors.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.qwen21_block_run import (HIDDEN, IMAGE, TEXT, NativeBlock,
                                           bf, modulation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--harness", type=Path, default=(
        Path(__file__).resolve().parents[1] / "target/npu-dev/harness-build/run_kernel"))
    parser.add_argument("--text-x", type=Path)
    parser.add_argument("--image-x", type=Path)
    parser.add_argument("--text-mod", type=Path)
    parser.add_argument("--image-mod", type=Path)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--keep-stages", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.layers <= 32:
        parser.error("layers must be in 1..32")
    if bool(args.text_x) != bool(args.image_x):
        parser.error("supply both embedded text and image states")
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    text = (np.fromfile(args.text_x, bfloat16).reshape(-1, HIDDEN) if args.text_x else
            bf(rng.normal(0, 0.6, (TEXT, HIDDEN))))
    if not 1 <= len(text) <= 64:
        raise ValueError("Qwen stack needs 1..64 text tokens")
    image = (np.fromfile(args.image_x, bfloat16).reshape(IMAGE, HIDDEN) if args.image_x else
             bf(rng.normal(0, 0.6, (IMAGE, HIDDEN))))
    text_mod = modulation(args.text_mod, rng)
    image_mod = modulation(args.image_mod, rng)
    for name, value in (("input-text", text), ("input-image", image),
                        ("text-mod", text_mod), ("image-mod", image_mod)):
        value.tofile(args.out / f"{name}.bf16")
    records = []
    for layer in range(args.layers):
        layer_out = args.out / f"layer-{layer:02d}"
        layer_out.mkdir()
        runner = NativeBlock(args.kernels.resolve(), args.weights.resolve(),
                             args.model.resolve(), layer, args.harness.resolve(), layer_out)
        result = runner.run(text, image, text_mod, image_mod)
        text, image = result["text"], result["image"]
        if not np.isfinite(text.astype(np.float32)).all() or not np.isfinite(image.astype(np.float32)).all():
            raise ValueError(f"layer {layer}: non-finite NPU state")
        text.tofile(layer_out / "text.bf16")
        image.tofile(layer_out / "image.bf16")
        rms = float(np.sqrt(np.mean(image.astype(np.float32) ** 2)))
        print(f"layer {layer:02d}: image RMS {rms:.6g}", flush=True)
        records.append({"layer": layer, "image_rms": rms, "operations": 24})
        if not args.keep_stages:
            for stage in layer_out.glob("0[1-4]-*"):
                shutil.rmtree(stage)
    (args.out / "run.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "scope": "native 512-pixel DiT block stack",
        "layers": args.layers, "text_tokens": len(text), "image_tokens": IMAGE,
        "operations": 24 * args.layers, "seed": args.seed,
        "kernel_manifest": str(args.kernels.resolve()),
        "weight_manifest": str((args.weights / "packed_weights.json").resolve()),
        "per_layer": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
