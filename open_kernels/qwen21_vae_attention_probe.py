"""Check the native single-head Qwen VAE attention (1152-wide decoder, 768-wide encoder)
against BF16 CPU math."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.designs.dit_conv.dit_conv import resolve_spec


TOKENS, CHANNELS = 1024, 1152


def prepare(kernel: Path, out: Path, seed: int, conv_fixture: Path | None = None,
            channels: int = CHANNELS) -> None:
    CHANNELS = channels
    out.mkdir(parents=True, exist_ok=False)
    if conv_fixture is None:
        rng = np.random.default_rng(seed)
        qkv = rng.standard_normal((TOKENS, 3, CHANNELS), dtype=np.float32).astype(bfloat16)
    else:
        meta = json.loads((conv_fixture / "fixture.json").read_text())
        if meta["name"] != "decoder.mid_block.attentions.0.to_qkv":
            raise ValueError("attention input must be the checkpoint to_qkv convolution")
        spec = resolve_spec(meta["stream"]["spec"], meta["stream"]["taps"])
        if (spec["H"], spec["W"], spec["Cout"]) != (32, 32, 3 * CHANNELS):
            raise ValueError("checkpoint to_qkv convolution geometry changed")
        y = np.fromfile(conv_fixture / "Y.out", bfloat16)
        y = y[:(32 + 2) * spec["y"]["pitch"] * spec["Cout"]].reshape(
            34, spec["y"]["pitch"], spec["Cout"])
        qkv = y[1:33, 1:33].reshape(TOKENS, 3, CHANNELS).copy()
    qkv.reshape(TOKENS, 3 * CHANNELS).tofile(out / "qkv.bf16")
    size = qkv.nbytes
    cfg = ["device", f"xclbin G {kernel / 'final.xclbin'}",
           f"kernelx k G {kernel / 'insts.bin'}",
           *[f"buf {role} {size} {out / 'qkv.bf16'}" for role in ("Q", "K", "V")],
           f"buf O {TOKENS * CHANNELS * 2}", "run k Q K V O",
           f"dump O {out / 'output.bf16'} {TOKENS * CHANNELS * 2}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    (out / "fixture.json").write_text(json.dumps({
        "seed": seed, "tokens": TOKENS, "channels": channels,
        "conv_fixture": str(conv_fixture) if conv_fixture else None}, indent=2) + "\n")


def check(out: Path) -> None:
    CHANNELS = json.loads((out / "fixture.json").read_text())["channels"]
    qkv = np.fromfile(out / "qkv.bf16", bfloat16).reshape(TOKENS, 3, CHANNELS)
    got = np.fromfile(out / "output.bf16", bfloat16).reshape(TOKENS, CHANNELS).astype(np.float32)
    if not np.isfinite(got).all():
        raise SystemExit("Qwen VAE attention produced nonfinite values")
    q, k, v = [qkv[:, i].astype(np.float32) for i in range(3)]
    rows = np.array([0, 1, 7, 8, 127, 128, 511, 1023])
    scores = q[rows] @ k.T / np.sqrt(CHANNELS)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    ref = p @ v
    error = float(np.linalg.norm(got[rows] - ref) / np.linalg.norm(ref))
    cosine = float((got[rows] * ref).sum() /
                   (np.linalg.norm(got[rows]) * np.linalg.norm(ref)))
    print(f"Qwen VAE attention: relative L2 {error:.6g}, cosine {cosine:.9f}, "
          f"output RMS {float(np.sqrt(np.mean(got * got))):.6g}")
    if error > 0.05:
        raise SystemExit("Qwen VAE attention numeric gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--kernel", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=37)
    prep.add_argument("--conv-fixture", type=Path)
    prep.add_argument("--channels", type=int, default=CHANNELS, choices=(768, 1152))
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.kernel, args.out, args.seed,
            args.conv_fixture, args.channels) if args.stage == "prepare" else check(args.out)


if __name__ == "__main__":
    main()
