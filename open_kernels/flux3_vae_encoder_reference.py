"""Compare native FLUX 3 VAE image tokens with the published FP32 encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-src", type=Path, required=True)
    parser.add_argument("--roundtrip", action="store_true")
    args = parser.parse_args()
    out = args.fixture.resolve()
    meta = json.loads((out / "fixture.json").read_text())
    if meta["model"] != "black-forest-labs/flux-3-image-klein" or meta["stage"] != "encoder":
        raise ValueError("fixture is not a FLUX 3 VAE encoder pass")
    resolution = meta["resolution"]
    image = np.fromfile(out / "image.f32", "<f4").reshape(resolution, resolution, 3)
    sys.path.insert(0, str(args.reference_src.resolve() / "src"))
    from flux3.autoencoder import AutoEncoder, latent_ids

    torch.set_num_threads(8)
    vae = AutoEncoder().eval()
    vae.load_state_dict(load_file(str(args.model.resolve() / "ae.safetensors")), strict=True)
    captures = {}
    if meta.get("trace"):
        def capture(name):
            def hook(_module, _inputs, output):
                captures[name] = output.detach().numpy()[0].transpose(1, 2, 0)
            return hook
        for name, module in (
            ("conv_in", vae.encoder.conv_in),
            ("down0", vae.encoder.down[0].downsample),
            ("down1", vae.encoder.down[1].downsample),
            ("down2", vae.encoder.down[2].downsample),
            ("mid0", vae.encoder.mid.block_1),
            ("attention", vae.encoder.mid.attn_1),
            ("mid1", vae.encoder.mid.block_2),
        ):
            module.register_forward_hook(capture(name))
    source = torch.from_numpy(image.copy()).permute(2, 0, 1)
    with torch.inference_mode():
        expected, _ = vae.encode(source)
    expected = expected.numpy().astype("<f4")
    expected.tofile(out / "reference-tokens.f32")
    for name, wanted in captures.items():
        channels = wanted.shape[2]
        raw = np.fromfile(out / f"native-{name}.bf16", "<u2")
        if name == "conv_in":
            buffer = "X0"
        elif name == "down0":
            buffer = "X1"
        elif name == "down1":
            buffer = "X2"
        else:
            buffer = "X3"
        grid = wanted.shape[0]
        pitch = grid + 2 if grid >= 128 else 2 * (grid + 2)
        capacity = 128 if buffer == "X0" else 128 if buffer == "X1" else \
            256 if buffer == "X2" else 512
        got = (raw.astype(np.uint32) << 16).view(np.float32)
        got = got.reshape(-1, pitch, capacity)[1:grid + 1, 1:grid + 1, :channels]
        delta = got - wanted
        print(f"{name}: relative L2 {np.linalg.norm(delta)/np.linalg.norm(wanted):.6g}, "
              f"cosine {np.vdot(got,wanted)/(np.linalg.norm(got)*np.linalg.norm(wanted)):.9g}")
    native = out / "native-tokens.bf16"
    if native.is_file():
        raw = np.fromfile(native, "<u2")
        got = (raw.astype(np.uint32) << 16).view(np.float32).reshape(expected.shape)
        delta = got - expected
        print(f"native versus published FP32 encoder: relative L2 "
              f"{np.linalg.norm(delta)/np.linalg.norm(expected):.6g}, "
              f"cosine {np.vdot(got, expected)/(np.linalg.norm(got)*np.linalg.norm(expected)):.9g}, "
              f"max abs {np.max(np.abs(delta)):.6g}")
        if args.roundtrip:
            from PIL import Image
            ids = latent_ids(resolution // 16, resolution // 16)
            with torch.inference_mode():
                ref_image = vae.decode(torch.from_numpy(expected.copy()), ids)
                native_image = vae.decode(torch.from_numpy(got.copy()), ids)
            def to_rgb(value):
                return ((value.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
            ref_rgb, native_rgb = to_rgb(ref_image), to_rgb(native_image)
            Image.fromarray(ref_rgb).save(out / "reference-roundtrip.png")
            Image.fromarray(native_rgb).save(out / "native-roundtrip.png")
            difference = native_rgb.astype(np.float32) - ref_rgb.astype(np.float32)
            mae = np.mean(np.abs(difference))
            psnr = 20 * np.log10(255 / np.sqrt(np.mean(difference * difference)))
            print(f"native versus official VAE roundtrip: mean abs {mae:.6g}, "
                  f"PSNR {psnr:.6g} dB")
    print(f"published FLUX 3 FP32 VAE encoder: {expected.shape} -> "
          f"{out / 'reference-tokens.f32'}")


if __name__ == "__main__":
    main()
