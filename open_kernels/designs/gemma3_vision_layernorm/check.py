"""NPU check of Gemma 3 SigLIP BF16 LayerNorm including constant rows."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def run(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261028)
    x = rng.normal(1, 0.5, (256, 1152)).astype(np.float32).astype(bfloat16)
    x[0] = bfloat16(2)
    gamma = rng.normal(1, 0.15, 1152).astype(np.float32).astype(bfloat16)
    beta = rng.normal(0, 0.1, 1152).astype(np.float32).astype(bfloat16)
    params = np.stack((gamma, beta))
    xf = x.astype(np.float64)
    mean = xf.mean(axis=1, keepdims=True)
    variance = np.mean((xf - mean) ** 2, axis=1, keepdims=True)
    expected = ((xf - mean) / np.sqrt(variance + 1e-6) *
                gamma.astype(np.float64) + beta.astype(np.float64)).astype(bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.bin").write_bytes(x.tobytes())
    (out / "params.bin").write_bytes(params.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf input {x.nbytes} {out / 'input.bin'}",
        f"buf params {params.nbytes} {out / 'params.bin'}",
        f"buf result {expected.nbytes}", "run k input params result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    result = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"NPU harness failed: {result.stdout}\n{result.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16).reshape(expected.shape)
    error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
    scale = max(float(np.max(np.abs(expected.astype(np.float32)))), 1e-20)
    report = {"rows": 256, "hidden": 1152, "max_relative_error": error / scale,
              "exact_bf16_fraction": float(np.mean(actual == expected)),
              "constant_row_exact": bool(np.array_equal(actual[0], expected[0])),
              "passed": bool(np.isfinite(actual.astype(np.float32)).all() and
                             error / scale < 0.005 and np.array_equal(actual[0], expected[0]))}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve()), indent=2))
