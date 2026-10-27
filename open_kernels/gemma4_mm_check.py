"""NPU arithmetic check for Gemma 4 vision/audio BF16 GEMM tiles."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_kernels.gemma4_mm_source import npu_weight, npu_weight_row_major  # noqa: E402


def run(driver: Path, build: Path, out: Path, m: int, k: int, n: int,
        row_major: bool = False) -> dict:
    if m % 256 or k % 64 or n % 256 or (not row_major and k % 256):
        raise ValueError("MM test expects M,N/256 and K/64 (tiled K/256)")
    rng = np.random.default_rng(20261027 + m + k + n)
    a = rng.normal(0, 0.2, (m, k)).astype(np.float32).astype(bfloat16)
    w = rng.normal(0, 0.02, (n, k)).astype(np.float32).astype(bfloat16)
    if row_major:
        b = npu_weight_row_major(w, n, k)
    else:
        flm = w.view(np.uint16).reshape(n // 64, 64, k // 256, 256).transpose(0, 2, 1, 3)
        b = npu_weight(flm, n, k)
    expected = (a.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    (out / "activation.bin").write_bytes(a.tobytes())
    (out / "weight.bin").write_bytes(b.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf activation {a.nbytes} {out / 'activation.bin'}",
        f"buf weight {b.nbytes} {out / 'weight.bin'}",
        f"buf result {expected.nbytes}", "run k activation weight result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32).reshape(m, n)
    if not np.isfinite(actual).all():
        raise AssertionError("invalid multimodal GEMM output")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(delta)))
    relative = maximum / max(float(np.max(np.abs(expected))), 1e-20)
    a64, e64 = actual.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
    cosine = float(a64 @ e64 / (np.linalg.norm(a64) * np.linalg.norm(e64)))
    report = {"m": m, "k": k, "n": n,
              "source_layout": "row_major" if row_major else "flm_64x256",
              "max_absolute_error": maximum,
              "max_relative_error": relative, "cosine": cosine,
              "passed": bool(relative < 0.005 and cosine > 0.99999)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--m", required=True, type=int)
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--n", required=True, type=int)
    p.add_argument("--row-major", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.m, a.k, a.n, a.row_major), indent=2))
