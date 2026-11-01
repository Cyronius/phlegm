"""FLUX 3 native NPU text-to-image and one-reference editing pipeline.

Text-to-image runs the native prompt upsampler first. An edit prepares its
reference exactly as the published ``prepare_reference`` (EXIF transpose, RGB,
fit, centred reflect padding to the 16-pixel grid), encodes it with the native
VAE encoder, and conditions the DiT through the published ``image_cond`` mode.
The upsampler's I2I path needs its vision tower, which has no native NPU route
yet, so an edit uses the prompt as typed: the published ``pup=False`` path.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from open_kernels import flux3_native_dit_sample, flux3_native_pup, flux3_native_text


MATCH_MAX_PIXELS = 4 * 1024 * 1024
MIN_SIDE, MAX_SIDE, RATIO = 64, 4096, 16


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """The published pipeline.reference_canvas: (fitted size, padded canvas)."""
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, -(-v // RATIO) * RATIO)) for v in fit)
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(path: Path, resolution: int, out: Path):
    """Published prepare_reference for one image; returns (PNG path, crop box or None)."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height)
    if (w, h) != (resolution, resolution):
        raise ValueError(f"FLUX 3 reference {image.width}x{image.height} prepares to a {w}x{h} canvas; "
                         f"the native NPU kernels cover {resolution}x{resolution}")
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    box = None
    if (w, h) != image.size:
        left, top = (w - image.width) // 2, (h - image.height) // 2
        padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
        mode = "reflect" if min(image.size) > RATIO else "symmetric"
        image = Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))
        box = (left, top, left + fit[0], top + fit[1])
    prepared = out / "reference.png"
    image.save(prepared)
    return prepared, box


def encode_reference(args, image: Path, out: Path) -> Path:
    import os
    from open_kernels.flux3_native_dit_sample import stage
    # open_kernels.vae_encoder imports its sibling vae_decoder as a top-level module.
    here = str(Path(__file__).resolve().parent)
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if here not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([p for p in paths if p] + [here])
    stage("open_kernels.flux3_vae_encoder_run", out, args.harness,
          kernels=args.vae_encoder_kernels, weights=args.vae_encoder_weights,
          out=out, resolution=args.resolution, image=image)
    subprocess.run([sys.executable, "-m", "open_kernels.flux3_vae_encoder_run", "check",
                    "--out", str(out)], check=True)
    return out / "native-tokens.bf16"


def generate(args) -> Path:
    edit = getattr(args, "image", None) is not None
    if args.resolution not in (512, 1024) or (edit and args.resolution != 512):
        raise ValueError("FLUX 3 has native 512x512 and 1024x1024 text-to-image kernels; "
                         "edits are native at 512x512")
    if edit and None in (getattr(args, "vae_encoder_kernels", None),
                         getattr(args, "vae_encoder_weights", None),
                         getattr(args, "cond_block_kernels", None),
                         getattr(args, "edit_shared_block_kernels", None)):
        raise ValueError("FLUX 3 editing needs VAE-encoder, image_cond and edit shared kernels")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    if edit:
        prepared, crop = prepare_reference(Path(args.image), args.resolution, out)
        reference_tokens = encode_reference(args, prepared, out / "reference-encoder")
        caption = args.prompt.strip()
        if not caption:
            raise ValueError("FLUX 3 edit prompt must be nonempty")
    else:
        crop, reference_tokens = None, None
        caption = upsample(args, out)
    return finish(args, out, caption, reference_tokens, crop)


def upsample(args, out: Path) -> str:
    pup_dir = out / "upsampler"
    caption_path = flux3_native_pup.run(SimpleNamespace(
        model=args.pup_model, kernels=args.pup_kernels, binary=args.text_binary,
        prompt=args.prompt, max_tokens=args.pup_max_tokens, out=pup_dir))
    pup_fixture = json.loads((pup_dir / "fixture.json").read_text())
    if not pup_fixture["complete"]:
        raise RuntimeError(f"FLUX 3 upsampler reached {args.pup_max_tokens} tokens without a stop token")
    caption = caption_path.read_text().strip()
    if not caption:
        raise RuntimeError("FLUX 3 upsampler returned an empty caption")
    return caption


def finish(args, out: Path, caption: str, reference_tokens, crop) -> Path:
    text_dir = flux3_native_text.encode_prompt(SimpleNamespace(
        model=args.text_model, kernels=args.text_kernels, binary=args.text_binary,
        prompt=caption, out=out / "text"))
    length = json.loads((text_dir / "fixture.json").read_text())["text_length"]
    bucket = str(length)
    try:
        edge = args.edge_kernels[bucket]
        text_block = args.text_block_kernels[bucket]
        shared_block = (args.edit_shared_block_kernels if reference_tokens is not None
                        else args.shared_block_kernels)[bucket]
    except KeyError as exc:
        raise ValueError(f"no native FLUX 3 DiT kernel set for {length} text tokens") from exc

    image = flux3_native_dit_sample.sample(SimpleNamespace(
        model=args.model, dit_weights=args.dit_weights,
        condition_kernels=args.condition_kernels, edge_kernels=edge,
        image_block_kernels=args.image_block_kernels,
        text_block_kernels=text_block, shared_block_kernels=shared_block,
        vae_kernels=args.vae_kernels, vae_weights=args.vae_weights,
        text_fixture=text_dir, harness=args.harness, out=out / "image",
        resolution=args.resolution, steps=args.steps, alpha=args.alpha,
        seed=args.seed, noise_f32=getattr(args, "noise_f32", None),
        reference_tokens=reference_tokens,
        cond_block_kernels=getattr(args, "cond_block_kernels", None),
        keep_intermediates=getattr(args, "keep_intermediates", True)))
    if crop is not None:
        from PIL import Image
        with Image.open(image) as decoded:
            decoded.crop(crop).save(out / "edited.png")
        image = out / "edited.png"
    (out / "summary.json").write_text(json.dumps({
        "model": "black-forest-labs/flux-3-image-klein",
        "task": "edit" if reference_tokens is not None else "text-to-image",
        "prompt": args.prompt, "expanded_caption": caption,
        "upsampled": reference_tokens is None,
        "reference_image": str(args.image) if reference_tokens is not None else None,
        "text_length": length, "steps": args.steps, "resolution": args.resolution,
        "seed": args.seed, "image": str(image),
    }, indent=2) + "\n")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "dit_weights", "vae_weights", "pup_model", "pup_kernels",
                 "text_model", "text_kernels", "text_binary", "condition_kernels",
                 "image_block_kernels", "vae_kernels", "harness", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ("edge_kernels", "text_block_kernels", "shared_block_kernels"):
        parser.add_argument("--" + name.replace("_", "-"), type=json.loads, required=True,
                            help='JSON mapping of text bucket length to kernel directory')
    parser.add_argument("--edit-shared-block-kernels", type=json.loads,
                        help="JSON bucket map of one-reference shared block kernels")
    for name in ("image", "vae_encoder_kernels", "vae_encoder_weights", "cond_block_kernels",
                 "noise_f32"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--discard-steps", dest="keep_intermediates", action="store_false")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--pup-max-tokens", type=int, default=512)
    args = parser.parse_args()
    for name in ("edge_kernels", "text_block_kernels", "shared_block_kernels",
                 "edit_shared_block_kernels"):
        if getattr(args, name) is not None:
            setattr(args, name, {key: Path(value) for key, value in getattr(args, name).items()})
    print(generate(args))


if __name__ == "__main__":
    main()
