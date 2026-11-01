"""Diagnostic 512px Qwen Image 2.1 native DiT sampler; emits latents only.

All model projections and attention execute in the NPU harness. NumPy handles
the published FlowMatch Euler schedule, latent update and BF16 staging. This
does not claim image support until the model-specific VAE decoder is native.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

MODEL = "Qwen/Qwen-Image-2.1"
TOKENS, CHANNELS, HIDDEN = 1024, 64, 4096


def schedule(config: dict, steps: int) -> np.ndarray:
    """Mirror the shipped FlowMatchEulerDiscreteScheduler for T2I at 512px."""
    if not 2 <= steps <= 50:
        raise ValueError("Qwen Image 2.1 requires 2..50 steps; the published terminal shift is undefined for one step")
    if not config["use_dynamic_shifting"] or config["time_shift_type"] != "exponential":
        raise ValueError("unsupported Qwen scheduler configuration")
    base_len, max_len = config["base_image_seq_len"], config["max_image_seq_len"]
    mu = config["base_shift"] + (TOKENS - base_len) * (
        config["max_shift"] - config["base_shift"]) / (max_len - base_len)
    t = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    exp_mu = math.exp(mu)
    sigmas = exp_mu / (exp_mu + (1.0 / t - 1.0))
    terminal = config.get("shift_terminal")
    if terminal is not None:
        sigmas = 1.0 - (1.0 - sigmas) * (1.0 - terminal) / (1.0 - sigmas[-1])
    return np.append(sigmas.astype(np.float32), np.float32(0))


def _stage(module: str, stage: str | None, **options: object) -> None:
    script = Path(__file__).with_name(module.rsplit(".", 1)[-1] + ".py")
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [sys.executable, str(script)]
    if stage is not None:
        command.append(stage)
    for key, value in options.items():
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    subprocess.run(command, check=True, env=env)


def sample(args: argparse.Namespace) -> Path:
    model, weights = args.model.resolve(), args.weights.resolve()
    text = args.text_bf16.resolve()
    text_tokens = text.stat().st_size // (HIDDEN * 2)
    if not 1 <= text_tokens <= 64 or text.stat().st_size != text_tokens * HIDDEN * 2:
        raise ValueError("Qwen text encoder must contain 1..64 x 4096 BF16 values")
    metadata_path = text.parent / "fixture.json"
    if text.name != "tokens.bf16" or not metadata_path.is_file():
        raise ValueError("Qwen sampler needs raw text tower tokens.bf16 and its fixture.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("model") != MODEL or metadata.get("scope") != "native NPU text tower diagnostic" or \
            metadata.get("text_tokens") != text_tokens:
        raise ValueError("Qwen sampler needs raw Qwen3-VL text tower output, not image-entry rows")
    packed = json.loads((weights / "packed_weights.json").read_text())
    if packed.get("model") != "Qwen-Image-2.1" or not packed.get("full_model"):
        raise ValueError("complete Qwen Image 2.1 transformer pack is required")
    times = schedule(json.loads((model / "scheduler/scheduler_config.json").read_text()),
                     args.steps)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    latent = np.random.default_rng(args.seed).standard_normal(
        (TOKENS, CHANNELS), dtype=np.float32).astype(bfloat16).astype(np.float32)
    summary = []
    for index, current in enumerate(times[:-1]):
        place = out / f"step-{index:03d}"
        place.mkdir()
        input_path = place / "latent-input.bf16"
        latent.astype(bfloat16).tofile(input_path)
        entry = place / "entry"
        _stage("open_kernels.qwen21_image_edge_run", "entry", model=model,
               weights=weights, kernels=args.edge_kernels, harness=args.harness,
               text_bf16=text, latent_bf16=input_path, timestep=float(current), out=entry)
        stack = place / "stack"
        _stage("open_kernels.qwen21_stack_run", None, model=model, weights=weights,
               kernels=args.block_kernels, harness=args.harness,
               text_x=entry / "native-text.bf16", image_x=entry / "native-image.bf16",
               text_mod=entry / "native-text-mod.bf16",
               image_mod=entry / "native-image-mod.bf16", out=stack)
        final = place / "final"
        _stage("open_kernels.qwen21_image_edge_run", "final", weights=weights,
               kernels=args.edge_kernels, harness=args.harness,
               image_bf16=stack / "layer-31/image.bf16",
               scale_bf16=entry / "native-image-final-scale.bf16", out=final)
        velocity = np.fromfile(final / "native-velocity.bf16", bfloat16)
        if velocity.size != TOKENS * CHANNELS:
            raise ValueError("native velocity has the wrong extent")
        velocity = velocity.reshape(TOKENS, CHANNELS).astype(np.float32)
        target = float(times[index + 1])
        latent = (latent + np.float32(target - float(current)) * velocity).astype(
            bfloat16).astype(np.float32)
        if not np.isfinite(latent).all():
            raise ValueError(f"Qwen latent state diverged at step {index}")
        record = {"step": index, "sigma": float(current), "next_sigma": target,
                  "velocity_rms": float(np.sqrt(np.mean(velocity * velocity))),
                  "latent_rms": float(np.sqrt(np.mean(latent * latent)))}
        summary.append(record)
        print(f"Qwen native step {index + 1}/{args.steps}: "
              f"velocity RMS {record['velocity_rms']:.6g}, "
              f"latent RMS {record['latent_rms']:.6g}", flush=True)
        if not args.keep_intermediates:
            shutil.rmtree(place)
    final_path = out / "latents-final.bf16"
    latent.astype(bfloat16).tofile(final_path)
    (out / "summary.json").write_text(json.dumps({
        "model": MODEL, "scope": "native NPU DiT latents",
        "resolution": 512, "steps": args.steps, "seed": args.seed,
        "text_tokens": text_tokens, "text_embeddings": str(text), "schedule": summary,
        "latents": str(final_path)}, indent=2) + "\n")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "weights", "text_bf16", "block_kernels", "edge_kernels",
                 "harness", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--keep-intermediates", action="store_true")
    print(sample(parser.parse_args()))


if __name__ == "__main__":
    main()
