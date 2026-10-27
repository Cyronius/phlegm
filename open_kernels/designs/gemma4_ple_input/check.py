"""Check Gemma 4 PLE context normalization and optional token mixing on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def fixture(hidden: int, has_token: bool):
    rng = np.random.default_rng(20261027 + hidden + int(has_token))
    context = rng.normal(0, 5.0, 256).astype(np.float32).astype(bfloat16)
    token = rng.normal(0, 0.7, 256).astype(np.float32).astype(bfloat16)
    scale = rng.uniform(0.7, 1.3, 256).astype(np.float32).astype(bfloat16)
    scaled = (context.astype(np.float64) * hidden ** -0.5).astype(bfloat16)
    x = scaled.astype(np.float64)
    norm = (x / np.sqrt(np.mean(x * x) + 1e-6) * scale.astype(np.float64)).astype(bfloat16)
    if has_token:
        mixed = (norm.astype(np.float64) + token.astype(np.float64)).astype(bfloat16)
        expected = (mixed.astype(np.float64) * 2 ** -0.5).astype(bfloat16)
    else:
        expected = norm
    return np.concatenate((context, token)), scale, expected


def run(driver: Path, build: Path, out: Path, hidden: int, has_token: bool) -> dict:
    pair, scale, expected = fixture(hidden, has_token)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pair.bin").write_bytes(pair.tobytes())
    (out / "scale.bin").write_bytes(scale.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf pair {pair.nbytes} {out / 'pair.bin'}",
        f"buf scale {scale.nbytes} {out / 'scale.bin'}",
        f"buf result {expected.nbytes}", "run k pair scale result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16)
    if actual.shape != expected.shape or not np.isfinite(actual.astype(np.float32)).all():
        raise AssertionError("invalid PLE input output")
    a, e = actual.astype(np.float32), expected.astype(np.float32)
    maximum = float(np.max(np.abs(a - e)))
    cosine = float(a.astype(np.float64) @ e.astype(np.float64) /
                   (np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(e.astype(np.float64))))
    report = {"hidden": hidden, "has_token": has_token,
              "max_absolute_error": maximum, "cosine": cosine,
              "exact_bf16_fraction": float(np.mean(a == e)),
              "passed": bool(maximum <= 0.02 and cosine > 0.99999)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hidden", required=True, type=int, choices=(1536, 2560))
    p.add_argument("--no-token", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve(),
                         args.hidden, not args.no_token), indent=2))
