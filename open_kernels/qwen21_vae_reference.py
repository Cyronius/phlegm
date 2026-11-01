"""Compare the native Qwen Image 2.1 VAE decoder with the FP32 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def reference(model: Path, latents: Path, out: Path) -> None:
    import torch
    from diffusers import AutoencoderKLQwenImage21

    out.mkdir(parents=True, exist_ok=False)
    raw = np.fromfile(latents, "<u2")
    if raw.size != 32 * 32 * 64:
        raise ValueError("Qwen VAE reference needs 1024x64 BF16 latents")
    grid = (raw.astype(np.uint32) << 16).view(np.float32).reshape(32, 32, 64)
    z = torch.from_numpy(grid).permute(2, 0, 1)[None, :, None].contiguous()
    config = json.loads((model / "vae/config.json").read_text())
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32).view(1, 64, 1, 1, 1)
    std = torch.tensor(config["latents_std"], dtype=torch.float32).view(1, 64, 1, 1, 1)
    z = z * std + mean
    vae = AutoencoderKLQwenImage21.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()
    with torch.inference_mode():
        decoded = vae.decode(z, return_dict=False)[0][0, :, 0]
    rgba = decoded.permute(1, 2, 0).contiguous().numpy()
    rgba.tofile(out / "reference.f32")
    print(f"FP32 checkpoint decoder: {rgba.shape}, RMS={np.sqrt(np.mean(rgba * rgba)):.6g}")


def compare(out: Path, native: Path) -> None:
    ref = np.fromfile(out / "reference.f32", "<f4").reshape(512, 512, 4)
    got = np.fromfile(native, "<f4").reshape(512, 512, 4)
    error = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    cosine = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
    mae = float(np.mean(np.abs(got - ref)))
    print(f"Qwen VAE native vs FP32: relL2={error:.6g}, cosine={cosine:.9f}, MAE={mae:.6g}")
    if not np.isfinite(got).all() or error > 0.20:
        raise SystemExit("Qwen VAE full decoder numeric gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("reference")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--latents", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()
    reference(args.model, args.latents, args.out) if args.command == "reference" else compare(
        args.out, args.native)


if __name__ == "__main__":
    main()
