"""Staged, native AMD NPU MOSS Image sampler.

All learned projections, attention, modulation and VAE decoding run through
the native encoder or XRT harness. NumPy only prepares inputs and applies the
published FP32 CFG/Euler update between DiT forwards. Supported prompt lengths
reuse verified RTP variants of prebuilt NPU kernel sets.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from .moss_native_text import encode_prompt
from .moss_kernel_variant import derive
from .moss_schedule import cfg_euler_step, sigmas
from .moss_text_tokens import token_ids


def _stage(module: str, command: str, directory: Path, options: dict[str, object],
           harness: Path) -> None:
    args = [sys.executable, "-m", module, command]
    for key, value in options.items():
        if value is not None:
            args += ["--" + key.replace("_", "-"), str(value)]
    subprocess.run(args, check=True)
    with (directory / "npu.log").open("w") as log:
        subprocess.run([str(harness), str(directory / "run.cfg")],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    if "DONE runs=" not in (directory / "npu.log").read_text().splitlines()[-1]:
        raise RuntimeError(f"native XRT run did not complete: {directory}")


def _kernel(template: str, *, length: int, resolution: int, kind: str) -> Path:
    path = Path(template.format(length=length, resolution=resolution)).resolve()
    if (path / "manifest.json").is_file():
        return path
    if length in (10, 17):
        raise FileNotFoundError(f"missing exported NPU kernel witness for {length} text rows: {path}")
    base = Path(template.format(length=17, resolution=resolution)).resolve()
    witness = Path(template.format(length=10, resolution=resolution)).resolve()
    if not (base / "manifest.json").is_file() or not (witness / "manifest.json").is_file():
        raise FileNotFoundError(f"no exported NPU kernel witnesses for {length} text rows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / ".moss-kernel-variant.lock"
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if not (path / "manifest.json").is_file():
            if path.exists():
                raise FileExistsError(f"incomplete NPU kernel variant: {path}")
            partial = path.with_name(f"{path.name}.building-{uuid.uuid4().hex}")
            try:
                derive(base, witness, partial, kind, resolution, length)
                partial.rename(path)
            finally:
                if partial.exists():
                    shutil.rmtree(partial)
    return path


def _text(prompt: str, branch: str, args: argparse.Namespace, out: Path) -> tuple[Path, int]:
    length = len(token_ids(args.tokenizer_dir, prompt))
    place = out / f"text-{branch}"
    place.mkdir()
    hidden = encode_prompt(prompt, tokenizer_dir=args.tokenizer_dir,
                           model_dir=args.text_model, kernel_dir=args.text_kernels,
                           encoder_binary=args.text_binary)
    if hidden.shape != (length, 4096) or not np.isfinite(hidden).all():
        raise ValueError("native text encoder returned invalid hidden states")
    hidden_path = place / "encoder-hidden.f32"
    hidden.astype("<f4").tofile(hidden_path)
    refine = place / "refine"
    _stage("open_kernels.moss_text_refine_run", "prepare", refine, {
        "kernels": _kernel(args.refine_kernels, length=length, resolution=args.resolution,
                           kind="text"),
        "weights": args.dit_weights, "model": args.model,
        "hidden": hidden_path, "out": refine, "text_length": length,
    }, args.harness)
    return refine / "native-text.bf16", length


def _velocity(branch: str, length: int, text: Path, latents: Path,
              modulation: Path, head: Path, step: int,
              args: argparse.Namespace, out: Path) -> np.ndarray:
    place = out / f"step-{step:03d}" / branch
    place.mkdir(parents=True)
    entry = place / "entry"
    _stage("open_kernels.moss_image_edge_run", "prepare-entry", entry, {
        "kernels": args.edge_kernels, "weights": args.dit_weights,
        "model": args.model, "out": entry, "resolution": args.resolution,
        "latents_f32": latents,
    }, args.harness)
    joint = place / "joint"
    _stage("open_kernels.moss_dit_block_run", "prepare", joint, {
        "kernels": _kernel(args.block_kernels, length=length, resolution=args.resolution,
                           kind="block"),
        "weights": args.dit_weights, "model": args.model,
        "out": joint, "resolution": args.resolution, "text_length": length,
        "layer": 0, "layers": 42, "skip_layer": 22 if branch == "negative" else None,
        "image_x": entry / "native-image-x.bf16", "text_x": text,
        "modulation": modulation,
    }, args.harness)
    exit_stage = place / "exit"
    _stage("open_kernels.moss_image_edge_run", "prepare-exit", exit_stage, {
        "kernels": args.edge_kernels,
        "block_kernels": _kernel(args.block_kernels, length=length,
                                  resolution=args.resolution, kind="block"),
        "weights": args.dit_weights, "model": args.model,
        "hidden": joint / "native-x.bf16", "head": head,
        "out": exit_stage, "resolution": args.resolution,
        "text_length": length,
    }, args.harness)
    tokens = (args.resolution // 16) ** 2
    raw = np.fromfile(exit_stage / "native-velocity.bf16", bfloat16)
    if raw.size != tokens * 1024:
        raise ValueError("native MOSS velocity has wrong extent")
    velocity = raw.reshape(tokens, 1024)[:, :128].astype(np.float32).T.copy()
    if not np.isfinite(velocity).all():
        raise ValueError("native MOSS velocity contains NaN or Inf")
    return velocity


def sample(args: argparse.Namespace) -> Path:
    if args.resolution not in (512, 1024):
        raise ValueError("native MOSS currently supports 512 and 1024 pixel squares")
    if not isinstance(args.prompt, str) or not args.prompt.strip():
        raise ValueError("MOSS prompt must be nonempty")
    if not math.isfinite(args.guidance) or args.guidance < 0:
        raise ValueError("MOSS guidance must be finite and nonnegative")
    schedule = sigmas(args.steps)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    text, text_length = _text(args.prompt, "positive", args, out)
    negative = None
    if args.guidance != 1:
        negative = _text(args.negative_prompt, "negative", args, out)
    grid = args.resolution // 16
    latents = np.random.default_rng(args.seed).standard_normal(
        (128, grid * grid), dtype=np.float32)
    for step, sigma in enumerate(schedule[:-1]):
        latent_path = out / f"latents-{step:03d}.f32"
        latents.astype("<f4").tofile(latent_path)
        conditioning = out / f"step-{step:03d}" / "conditioning"
        conditioning.parent.mkdir()
        _stage("open_kernels.moss_conditioning_run", "prepare", conditioning, {
            "kernels": args.condition_kernels, "weights": args.dit_weights,
            "model": args.model, "out": conditioning, "sigma": float(sigma),
        }, args.harness)
        mod = conditioning / "native-mod.bf16"
        head = conditioning / "native-head.bf16"
        conditional = _velocity("positive", text_length, text, latent_path, mod,
                                head, step, args, out)
        if negative is None:
            latents = latents + np.float32(schedule[step + 1] - sigma) * conditional
        else:
            negative_text, negative_length = negative
            unconditional = _velocity("negative", negative_length, negative_text,
                                      latent_path, mod, head, step, args, out)
            latents = cfg_euler_step(latents, conditional, unconditional,
                                     args.guidance, sigma, schedule[step + 1])
        if not np.isfinite(latents).all():
            raise ValueError(f"MOSS latent state diverged at step {step}")
        print(f"step {step + 1}/{len(schedule) - 1}: sigma {sigma:.6g}, "
              f"latent RMS {np.sqrt(np.mean(latents * latents)):.6g}", flush=True)
        if not args.keep_intermediates:
            shutil.rmtree(out / f"step-{step:03d}")
            latent_path.unlink()
    final = out / "latents-final.f32"
    latents.astype("<f4").tofile(final)
    decoder = out / "decode"
    _stage("open_kernels.moss_vae_run", "prepare", decoder, {
        "kernels": args.vae_kernels, "weights": args.vae_weights,
        "out": decoder, "resolution": args.resolution,
        "latents_f32": final,
    }, args.harness)
    subprocess.run([sys.executable, "-m", "open_kernels.moss_vae_run",
                    "check", "--out", str(decoder)], check=True)
    (out / "run.json").write_text(json.dumps({
        "model": "OpenMOSS-Team/MOSS-Image-SFT", "backend": "native AMD NPU",
        "prompt": args.prompt, "negative_prompt": args.negative_prompt,
        "resolution": args.resolution, "seed": args.seed,
        "steps": len(schedule) - 1, "guidance": args.guidance,
        "positive_tokens": text_length,
        "negative_tokens": negative[1] if negative else None,
        "negative_skip_layer": 22 if negative else None,
        "image": str(decoder / "native-vae.png"),
    }, indent=2) + "\n")
    return decoder / "native-vae.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dit-weights", type=Path, required=True)
    parser.add_argument("--vae-weights", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-kernels", type=Path, required=True)
    parser.add_argument("--text-binary", type=Path, required=True)
    parser.add_argument("--refine-kernels", required=True,
                        help="path template with {length} for the real token count")
    parser.add_argument("--condition-kernels", type=Path, required=True)
    parser.add_argument("--block-kernels", required=True,
                        help="path template with {length} and {resolution}")
    parser.add_argument("--edge-kernels", type=Path, required=True)
    parser.add_argument("--vae-kernels", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="retain each native stage's buffers and XRT logs")
    args = parser.parse_args()
    print(sample(args))


if __name__ == "__main__":
    main()
