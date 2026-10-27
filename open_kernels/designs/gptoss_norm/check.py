"""Compare GPT-OSS 2880-value RMSNorm with the FP64 layer reference."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from replica_dense import rms  # noqa: E402


def fixture(case: str):
    rng = np.random.default_rng(20261029)
    if case == "random":
        x = rng.normal(0, 0.8, 2880).astype(np.float32)
    elif case == "zero":
        x = np.zeros(2880, np.float32)
    elif case == "offset":
        x = (rng.normal(0, 0.02, 2880) + 1.5).astype(np.float32)
    else:
        raise ValueError("case must be random, zero or offset")
    w = rng.normal(1, 0.2, 2880).astype(np.float32).astype(bfloat16)
    expected = (rms(x, 1e-5) * w.astype(np.float64)).astype(bfloat16)
    return x, w, expected


def run(driver: Path, build: Path, out: Path, case: str) -> dict:
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    x, w, expected = fixture(case)
    out.mkdir(parents=True, exist_ok=True)
    (out / "x.bin").write_bytes(x.tobytes())
    (out / "w.bin").write_bytes(w.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf residual {x.nbytes} {out / 'x.bin'}",
        f"buf weight {w.nbytes} {out / 'w.bin'}",
        f"buf result {expected.nbytes}", "run k residual weight result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"NPU harness failed: {completed.stdout}\n{completed.stderr}")
    actual = np.fromfile(out / "result.bin", dtype=bfloat16)
    if actual.shape != expected.shape or not np.isfinite(actual.astype(np.float32)).all():
        raise AssertionError("invalid NPU output")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    max_abs = float(np.max(np.abs(a - e)))
    max_rel = float(max_abs / max(np.max(np.abs(e)), 1e-30))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-30))
    report = {"passed": bool(max_rel < 0.01 and (case == "zero" or cosine > 0.99999)),
              "case": case, "max_absolute_error": max_abs,
              "max_relative_error": max_rel, "cosine": cosine,
              "exact_bf16_fraction": float(np.mean(a == e)),
              "reference": "replica_dense.rms, GPT-OSS epsilon 1e-5"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--case", required=True, choices=("random", "zero", "offset"))
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.case), indent=2))


if __name__ == "__main__":
    main()
