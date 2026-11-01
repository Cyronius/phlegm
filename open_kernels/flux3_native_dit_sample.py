"""FLUX 3 flow-map sampling with native NPU DiT and VAE stages.

Accepts a published CPU text fixture or the native Qwen3-VL tap path.
The caller can apply the native prompt upsampler before encoding. With
``--reference-tokens`` (native VAE-encoder output for one same-size reference
image), the published ``image_cond`` mode joins the sample: its entry runs once,
its five per-mode blocks run each step with t=0 plus flow-map target
modulation, and the shared blocks attend over text, image and reference.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


EL = 3072
MODEL = "black-forest-labs/flux-3-image-klein"


def stage(module: str, directory: Path, harness: Path, **options) -> None:
    command = [sys.executable, "-m", module, "prepare"]
    for key, value in options.items():
        if value is True:
            command.append("--" + key.replace("_", "-"))
        elif value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    subprocess.run(command, check=True)
    with (directory / "npu.log").open("w") as log:
        subprocess.run([str(harness), str(directory / "run.cfg")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (directory / "npu.log").read_text().splitlines()[-1].startswith("DONE runs="):
        raise RuntimeError(f"native XRT run did not complete: {directory}")


def rows(path: Path, count: int, width: int) -> np.ndarray:
    value = np.fromfile(path, bfloat16)
    if value.size < count * width:
        raise ValueError(f"{path}: native output shorter than {count} x {width}")
    value = value[:count * width].reshape(count, width)
    if not np.isfinite(value.astype(np.float32)).all():
        raise ValueError(f"{path}: nonfinite native output")
    return value


def schedule(steps: int, alpha: float) -> np.ndarray:
    if not 1 <= steps <= 50 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("FLUX 3 diagnostic sampling requires 1..50 steps and positive finite alpha")
    t = np.linspace(1, 0, steps + 1, dtype=np.float32)
    return alpha * t / (1 + (alpha - 1) * t)


def sample(args) -> Path:
    if args.resolution not in (512, 1024):
        raise ValueError("FLUX 3 native sampling has 512- and 1024-pixel kernel sets")
    if getattr(args, "reference_tokens", None) is not None and args.resolution != 512:
        # qk_flux3 stacks the reference under the image grid (grid_h <= 64), so
        # a same-size reference needs a 512-pixel canvas.
        raise ValueError("FLUX 3 native reference conditioning covers 512-pixel canvases")
    fixture = json.loads((args.text_fixture / "fixture.json").read_text())
    if (fixture.get("model") != MODEL or
            fixture.get("scope") not in (
                "published CPU text encoder fixture; native NPU text encoder outstanding",
                "native NPU Qwen3-VL text tower; 80-token diagnostic",
                "native NPU Qwen3-VL text tower") or
            not isinstance(fixture.get("text_length"), int) or
            fixture["text_length"] < 80 or fixture["text_length"] % 80):
        raise ValueError("expected a padded published or native FLUX 3 text fixture")
    text_length = fixture["text_length"]
    text_path = args.text_fixture / "tokens.bf16"
    if text_path.stat().st_size != text_length * 20480 * 2:
        raise ValueError("FLUX 3 text fixture has wrong extent")
    times = schedule(args.steps, args.alpha)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    text_entry = out / "text-entry"
    stage("open_kernels.flux3_image_edge_run", text_entry, args.harness,
          kernels=args.edge_kernels, weights=args.dit_weights, out=text_entry,
          stage="text-entry", resolution=args.resolution, text_length=text_length, input_bf16=text_path)
    text_hidden = text_entry / "native-output.bf16"
    image_tokens = (args.resolution // 16) ** 2
    reference = getattr(args, "reference_tokens", None)
    references = int(reference is not None)
    if reference is not None:
        if getattr(args, "cond_block_kernels", None) is None:
            raise ValueError("FLUX 3 reference conditioning needs --cond-block-kernels")
        shared_manifest = json.loads((Path(args.shared_block_kernels) / "manifest.json").read_text())
        if shared_manifest.get("references", 0) != 1:
            raise ValueError("FLUX 3 reference conditioning needs a one-reference shared kernel set")
        tokens = np.fromfile(reference, bfloat16)
        if tokens.size != image_tokens * 128 or not np.isfinite(tokens.astype(np.float32)).all():
            raise ValueError("FLUX 3 reference tokens must be finite [image_tokens,128] BF16")
        cond_entry = out / "cond-entry"
        stage("open_kernels.flux3_image_edge_run", cond_entry, args.harness,
              kernels=args.edge_kernels, weights=args.dit_weights, out=cond_entry,
              stage="cond-entry", resolution=args.resolution, text_length=text_length,
              input_bf16=Path(reference).resolve())
        cond_hidden = cond_entry / "native-output.bf16"
    if args.noise_f32:
        latent = np.fromfile(args.noise_f32, "<f4")
        if latent.size != image_tokens * 128:
            raise ValueError("FLUX 3 diagnostic noise extent differs from [image_tokens,128]")
        latent = latent.reshape(image_tokens, 128)
    else:
        latent = np.random.default_rng(args.seed).standard_normal(
            (image_tokens, 128), dtype=np.float32).astype(bfloat16).astype(np.float32)
    summary = []
    for index, current in enumerate(times[:-1]):
        target = float(times[index + 1])
        step_dir = out / f"step-{index:03d}"
        step_dir.mkdir()
        latent_file = step_dir / "latent-input.bf16"
        latent.astype(bfloat16).tofile(latent_file)
        cond = step_dir / "conditioning"
        stage("open_kernels.flux3_conditioning_run", cond, args.harness,
              kernels=args.condition_kernels, weights=args.dit_weights,
              out=cond, time=float(current), target=target,
              reference=True if references else None)
        subprocess.run([sys.executable, "-m", "open_kernels.flux3_conditioning_run",
                        "check", "--out", str(cond)], check=True)
        image_entry = step_dir / "image-entry"
        stage("open_kernels.flux3_image_edge_run", image_entry, args.harness,
              kernels=args.edge_kernels, weights=args.dit_weights, out=image_entry,
              stage="image-entry", resolution=args.resolution, text_length=text_length,
              input_bf16=latent_file)
        image_early = step_dir / "image-early"
        stage("open_kernels.flux3_block_run", image_early, args.harness,
              kernels=args.image_block_kernels, weights=args.dit_weights,
              model=args.model, out=image_early, resolution=args.resolution, layer=0, layers=5,
              mode="image", stage="early", text_length=text_length, modulation_dir=cond,
              input_bf16=image_entry / "native-output.bf16")
        text_early = step_dir / "text-early"
        stage("open_kernels.flux3_block_run", text_early, args.harness,
              kernels=args.text_block_kernels, weights=args.dit_weights,
              model=args.model, out=text_early, resolution=args.resolution, layer=0, layers=5,
              mode="text", stage="early", text_length=text_length, modulation_dir=cond,
              input_bf16=text_hidden)
        parts = [rows(text_early / "native-x.bf16", text_length, EL),
                 rows(image_early / "native-x.bf16", image_tokens, EL)]
        if references:
            cond_early = step_dir / "cond-early"
            stage("open_kernels.flux3_block_run", cond_early, args.harness,
                  kernels=args.cond_block_kernels, weights=args.dit_weights,
                  model=args.model, out=cond_early, resolution=args.resolution, layer=0, layers=5,
                  mode="image_cond", stage="early", text_length=text_length,
                  modulation_dir=cond, input_bf16=cond_hidden)
            parts.append(rows(cond_early / "native-x.bf16", image_tokens, EL))
        joint = step_dir / "joint-input.bf16"
        np.concatenate(parts).tofile(joint)
        shared = step_dir / "shared"
        stage("open_kernels.flux3_block_run", shared, args.harness,
              kernels=args.shared_block_kernels, weights=args.dit_weights,
              model=args.model, out=shared, resolution=args.resolution, layer=0, layers=28,
              mode="image", stage="shared", text_length=text_length, modulation_dir=cond,
              input_bf16=joint, references=references or None)
        image_final = step_dir / "image-hidden.bf16"
        joint_out = rows(shared / "native-x.bf16",
                         text_length + image_tokens * (1 + references), EL)
        joint_out[text_length:text_length + image_tokens].tofile(image_final)
        final = step_dir / "final"
        stage("open_kernels.flux3_image_edge_run", final, args.harness,
              kernels=args.edge_kernels, weights=args.dit_weights, out=final,
              stage="final-exit", resolution=args.resolution, text_length=text_length,
              input_bf16=image_final,
              modulation_bf16=cond / "modulation-final.bf16")
        velocity = rows(final / "native-output.bf16", image_tokens, 1024)[:, :128].astype(np.float32)
        rms = float(np.sqrt(np.mean(velocity * velocity)))
        latent = latent + np.float32(target - current) * velocity
        if not np.isfinite(latent).all():
            raise ValueError(f"FLUX 3 latent state became nonfinite at step {index}")
        if not getattr(args, "keep_intermediates", True):
            import shutil
            shutil.rmtree(step_dir)
        summary.append({"step": index, "time": float(current), "target": target,
                        "velocity_rms": rms,
                        "latent_rms": float(np.sqrt(np.mean(latent * latent)))})
        print(f"FLUX 3 native step {index + 1}/{args.steps}: "
              f"velocity RMS {rms:.5g}, latent RMS {summary[-1]['latent_rms']:.5g}", flush=True)
    final_latents = out / "latents-final.f32"
    latent.T.astype("<f4").tofile(final_latents)
    decode = out / "decode"
    stage("open_kernels.flux3_vae_run", decode, args.harness,
          kernels=args.vae_kernels, weights=args.vae_weights, out=decode,
          resolution=args.resolution, latents_f32=final_latents)
    subprocess.run([sys.executable, "-m", "open_kernels.flux3_vae_run", "check",
                    "--out", str(decode)], check=True)
    (out / "summary.json").write_text(json.dumps({
        "model": MODEL, "scope": "native NPU DiT/VAE with " + fixture["scope"],
        "prompt": fixture["prompt"], "steps": args.steps, "alpha": args.alpha,
        "seed": args.seed, "resolution": args.resolution,
        "text_fixture": str(args.text_fixture), "references": references,
        "reference_tokens": str(reference) if reference is not None else None,
        "schedule": summary,
        "image": str(decode / "native-vae.png"),
    }, indent=2) + "\n")
    return decode / "native-vae.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "dit_weights", "condition_kernels", "edge_kernels",
                 "image_block_kernels", "text_block_kernels", "shared_block_kernels",
                 "vae_kernels", "vae_weights", "text_fixture", "harness", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--noise-f32", type=Path)
    parser.add_argument("--reference-tokens", type=Path,
                        help="native VAE-encoder tokens of one same-size reference image")
    parser.add_argument("--cond-block-kernels", type=Path,
                        help="image_cond per-mode block kernels (with --reference-tokens)")
    parser.add_argument("--discard-steps", dest="keep_intermediates", action="store_false",
                        help="remove each step's native buffers once it completes")
    args = parser.parse_args()
    print(sample(args))


if __name__ == "__main__":
    main()
