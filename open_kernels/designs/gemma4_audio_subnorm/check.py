"""Check Gemma 4 audio subsampler LayerNorm/ReLU on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def run(driver: Path, build: Path, out: Path, channels: int) -> dict:
    if channels not in (32, 128):
        raise ValueError(channels)
    rng = np.random.default_rng(20261027 + channels)
    conv = rng.normal(0, 0.4, (256, channels)).astype(np.float32).astype(bfloat16)
    x = conv.astype(np.float32)
    gamma = rng.uniform(0.7, 1.3, channels).astype(np.float32).astype(bfloat16)
    packets = b"".join(x[row].tobytes() + gamma.tobytes() for row in range(256))
    data = np.frombuffer(packets, np.uint8)
    value = x.astype(np.float64)
    mean = value.mean(axis=1, keepdims=True)
    variance = ((value - mean) ** 2).mean(axis=1, keepdims=True)
    expected = np.maximum(0, (value - mean) / np.sqrt(variance + 1e-6) *
                          gamma.astype(np.float64)).astype(bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.bin").write_bytes(data.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf input {data.nbytes} {out / 'input.bin'}",
        f"buf result {expected.nbytes}", "run k input result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16)
    e = expected.astype(np.float32).ravel()
    a = actual.astype(np.float32)
    error = float(np.max(np.abs(a - e)))
    report = dict(channels=channels, pixels=256, max_absolute_error=error,
                  exact_bf16_fraction=float(np.mean(a == e)),
                  passed=bool(np.isfinite(a).all() and error <= 0.016))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--channels", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(),
                         args.out.resolve(), args.channels), indent=2))
