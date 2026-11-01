"""Validate Qwen Image 2.1 VAE channel RMSNorm on the vector-width NPU kernel.

Qwen's decoder has 144, 288, 576 and 1152 channels. The shared dit_ew
RMSNorm consumes groups of 32; zero-pad the channel vector and multiply its
gamma by sqrt(real_channels / padded_channels) to preserve the true RMS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

EL, ROWS = 4096, 512


def padded_width(channels: int, physical_channels: int | None = None) -> int:
    if channels not in (144, 288, 576, 1152):
        raise ValueError("unsupported Qwen VAE decoder channel count")
    padded = physical_channels or (channels + 31) // 32 * 32
    if padded < channels or padded % 32 or padded > EL:
        raise ValueError("physical channel count must be a supported multiple of 32")
    return padded


def generate(out: Path, kernel: Path, channels: int, seed: int,
             physical_channels: int | None = None) -> None:
    out.mkdir(parents=True, exist_ok=False)
    pad = padded_width(channels, physical_channels)
    rng = np.random.default_rng(seed)
    source = rng.normal(0, 1, (ROWS, channels)).astype(np.float32).astype(bfloat16)
    gamma = rng.uniform(0.7, 1.3, channels).astype(np.float32)
    x = np.zeros((ROWS, EL), bfloat16)
    x[:, :channels] = source
    p = np.zeros((3, EL), bfloat16)
    p[1, :channels] = (gamma * np.sqrt(channels / pad)).astype(bfloat16)
    spec = {"op": "ln_mod", "norm": "rms", "W": pad,
            "a": {"off": 0, "ld": EL, "T": ROWS, "E": 1},
            "y": {"off": 0, "ld": EL, "T": ROWS, "E": 1},
            "p_off": EL, "n_par": 1, "idx": {"scale": 0},
            "sizes": {"X": ROWS * EL, "B": EL, "P": 3 * EL,
                      "Y": ROWS * EL, "Z": EL}}
    (out / "spec.json").write_text(json.dumps(spec) + "\n")
    x.tofile(out / "X.bin")
    np.zeros(EL, bfloat16).tofile(out / "B.bin")
    p.tofile(out / "P.bin")
    gamma.tofile(out / "gamma.f32")
    run = ["device", f"xclbin G {kernel / 'final.xclbin'}",
           f"kernelx k G {kernel / 'insts.bin'}"]
    run += [f"buf {name} {size * 2} {name}.bin"
            for name, size in (("X", ROWS * EL), ("B", EL), ("P", 3 * EL))]
    run += [f"buf Y {ROWS * EL * 2}", f"buf Z {EL * 2}",
            "run k X B P Y Z", f"dump Y Y.out {ROWS * EL * 2}", ""]
    (out / "run.cfg").write_text("\n".join(run))
    (out / "fixture.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "channels": channels,
        "padded_channels": pad, "seed": seed}, indent=2) + "\n")


def check(out: Path) -> None:
    meta = json.loads((out / "fixture.json").read_text())
    channels = meta["channels"]
    x = np.fromfile(out / "X.bin", bfloat16).reshape(ROWS, EL)
    got = np.fromfile(out / "Y.out", bfloat16).reshape(ROWS, EL).astype(np.float32)
    gamma = np.fromfile(out / "gamma.f32", "<f4")
    source = x[:, :channels].astype(np.float32)
    reference = source / np.sqrt(np.mean(source * source, -1, keepdims=True) + 1e-12)
    reference *= gamma
    error = float(np.linalg.norm(got[:, :channels] - reference) /
                  np.linalg.norm(reference))
    tail = float(np.max(np.abs(got[:, channels:])))
    print(f"Qwen VAE RMSNorm C={channels} padded={meta['padded_channels']}: "
          f"relative L2 {error:.6g}, padded tail max {tail:.6g}")
    if not np.isfinite(got).all() or error > 0.01 or tail != 0:
        raise SystemExit("Qwen VAE padded RMSNorm failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--kernel", type=Path, required=True)
    prepare.add_argument("--channels", type=int, required=True)
    prepare.add_argument("--physical-channels", type=int)
    prepare.add_argument("--seed", type=int, default=37)
    validate = sub.add_parser("check")
    validate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.out, args.kernel, args.channels, args.seed,
             args.physical_channels) if args.stage == "prepare" else check(args.out)


if __name__ == "__main__":
    main()
