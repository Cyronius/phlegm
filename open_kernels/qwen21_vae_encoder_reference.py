"""Published FP32 Qwen Image 2.1 VAE encoder on the CPU: stage captures for the native one.

Runs diffusers' ``AutoencoderKLQwenImage21.encode`` unchanged (first frame, argmax
latents, then the pipeline's ``(latents - latents_mean) / latents_std``) on an image
prepared as ``QwenImage21Pipeline`` prepares a condition image (RGBA, VaeImageProcessor
to [-1, 1]). Forward hooks record each stage boundary of ``encoder``:

    conv_in, down_blocks.0 .. 4, mid_block, conv_out (before quant_conv)

as ``[H, W, C]`` float32 files, and ``latents.f32`` holds the normalized
``[H/16, W/16, 64]`` latents the DiT consumes. ``compare`` scores a native run
directory (``qwen21_vae_encoder_run``) stage by stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGES = ["conv_in", *(f"down_blocks.{i}" for i in range(5)), "mid_block"]


def prepare_pixels(image: Path, resolution: int) -> np.ndarray:
    """[resolution, resolution, 4] float32 in [-1, 1], as the published pipeline."""
    from PIL import Image
    with Image.open(image) as source:
        img = source.convert("RGBA")
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        return np.asarray(img, np.float32) / 127.5 - 1.0


def capture(args) -> None:
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
    from PIL import Image
    torch.set_num_threads(args.threads)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    vae = AutoencoderKLQwenImage21.from_pretrained(str(args.model / "vae"),
                                                   torch_dtype=torch.float32).eval()
    with Image.open(args.image) as source:
        img = source.convert("RGBA")
    processor = VaeImageProcessor(vae_scale_factor=16)
    pixels = processor.preprocess(img, width=args.resolution,
                                  height=args.resolution).unsqueeze(2)   # [1, 4, 1, H, W]
    pixels[0, :, 0].permute(1, 2, 0).numpy().astype("<f4").tofile(out / "pixels.f32")
    captures = {}

    def hook(name):
        def fn(_m, _i, o):
            captures[name] = o[0, :, 0].permute(1, 2, 0).detach().float().numpy()
        return fn

    enc = vae.encoder
    enc.conv_in.register_forward_hook(hook("conv_in"))
    for i, block in enumerate(enc.down_blocks):
        block.register_forward_hook(hook(f"down_blocks.{i}"))
    enc.mid_block.register_forward_hook(hook("mid_block"))
    enc.conv_out.register_forward_hook(hook("conv_out"))
    with torch.inference_mode():
        mean = vae.encode(pixels).latent_dist.mode()                     # argmax
        cfg = vae.config
        m = torch.tensor(cfg.latents_mean).view(1, -1, 1, 1, 1)
        s = torch.tensor(cfg.latents_std).view(1, -1, 1, 1, 1)
        latents = (mean - m) / s
    latents[0, :, 0].permute(1, 2, 0).numpy().astype("<f4").tofile(out / "latents.f32")
    shapes = {}
    for name, value in captures.items():
        value.astype("<f4").tofile(out / f"{name}.f32")
        shapes[name] = list(value.shape)
    shapes["latents"] = list(latents[0, :, 0].permute(1, 2, 0).shape)
    (out / "fixture.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1",
        "scope": "published diffusers AutoencoderKLQwenImage21.encode, FP32 CPU, first frame, "
                 "argmax latents normalized by latents_mean/std",
        "image": str(args.image), "resolution": args.resolution, "shapes": shapes,
    }, indent=2) + "\n")
    print(f"Qwen VAE encoder reference -> {out}")


def metrics(native: np.ndarray, expected: np.ndarray) -> dict:
    x, y = native.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    return {"relative_l2": float(np.linalg.norm(x - y) / np.linalg.norm(y)),
            "cosine": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
            "max_abs": float(np.abs(x - y).max())}


def compare(args) -> None:
    ref = args.reference.resolve()
    native = args.native.resolve()
    meta = json.loads((ref / "fixture.json").read_text())
    report = {}
    for name in [*STAGES, "latents"]:
        shape = meta["shapes"][name]
        path = native / f"stage-{name}.f32"
        if not path.is_file():
            continue
        got = np.fromfile(path, "<f4").reshape(shape)
        report[name] = metrics(got, np.fromfile(ref / f"{name}.f32", "<f4").reshape(shape))
        r = report[name]
        print(f"{name:14s} {str(shape):18s} rel L2 {r['relative_l2'] * 100:7.3f}%  "
              f"cos {r['cosine']:.6f}  max abs {r['max_abs']:.4g}")
    (native / "reference-compare.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()
    capture(args) if args.command == "capture" else compare(args)


if __name__ == "__main__":
    main()
