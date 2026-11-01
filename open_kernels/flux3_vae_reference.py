"""Compare a native FLUX 3 VAE decode with the pinned published PyTorch decoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-src", type=Path, required=True)
    args = parser.parse_args()
    out = args.fixture.resolve()
    meta = json.loads((out / "fixture.json").read_text())
    if meta["model"] != "black-forest-labs/flux-3-image-klein":
        raise ValueError("fixture is not FLUX 3 Image Klein")
    resolution = meta["resolution"]
    latent_side = resolution // 16
    raw = np.fromfile(out / "latents.bf16", dtype="<u2")
    tokens = torch.from_numpy(raw[:latent_side * latent_side * 128].copy())
    tokens = tokens.view(torch.bfloat16).float().reshape(-1, 128)

    sys.path.insert(0, str(args.reference_src.resolve() / "src"))
    from flux3.autoencoder import AutoEncoder, latent_ids

    torch.set_num_threads(8)
    vae = AutoEncoder().eval()
    vae.load_state_dict(load_file(str(args.model.resolve() / "ae.safetensors")), strict=True)
    with torch.inference_mode():
        image = vae.decode(tokens, latent_ids(latent_side, latent_side))
    rgb = ((image.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    rgb = rgb.permute(1, 2, 0).numpy()
    ref = np.concatenate([rgb, np.full((resolution, resolution, 1), 255, np.uint8)], axis=2)
    Image.fromarray(ref).save(out / "reference-vae.png")
    native = out / "output-rgba-blocks.bin"
    if native.is_file():
        native_raw = np.fromfile(native, np.uint8)
        if native_raw.size != resolution * resolution * 8:
            raise ValueError("native decoder output has wrong size")
        got = native_raw.reshape(-1, 8192)[:, :4096].reshape(resolution, resolution, 4)
        delta = got[..., :3].astype(np.float32) - rgb.astype(np.float32)
        mae = np.mean(np.abs(delta))
        psnr = 20 * np.log10(255 / np.sqrt(np.mean(delta * delta)))
        print(f"native versus official FP32 RGB: mean abs {mae:.6g}, PSNR {psnr:.6g} dB")
    print(f"official FLUX 3 FP32 VAE reference: {resolution}x{resolution} -> "
          f"{out / 'reference-vae.png'}")


if __name__ == "__main__":
    main()
