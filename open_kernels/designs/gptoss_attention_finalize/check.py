"""NPU check for GPT-OSS attention-state normalization and BF16 handoff."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def run(driver: Path, build: Path, out: Path, state_file: Path | None = None) -> dict:
    rng = np.random.default_rng(20261027)
    if state_file is None:
        state = np.zeros((64, 68), np.float32)
        state[:, :64] = rng.normal(0, 0.7, (64, 64))
        state[:, 64] = rng.normal(0, 2, 64)
        state[:, 65] = rng.uniform(0.125, 8, 64)
    else:
        state = np.fromfile(state_file, np.float32).reshape(64, 68)
    if not np.isfinite(state).all() or np.any(state[:, 65] <= 0):
        raise ValueError("attention state needs finite values and positive denominators")
    expected = (state[:, :64].astype(np.float64) /
                state[:, 65, None].astype(np.float64)).astype(bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.bin").write_bytes(state.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf state {state.nbytes} {out / 'state.bin'}",
        f"buf result {expected.nbytes}", "run k state result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16).reshape(64, 64)
    error = float(np.max(np.abs(actual.astype(np.float32) -
                                expected.astype(np.float32))))
    report = {"source": str(state_file) if state_file else "synthetic",
              "max_absolute_error": error,
              "exact_bf16_fraction": float(np.mean(actual == expected)),
              "passed": bool(np.isfinite(actual.astype(np.float32)).all() and
                             error <= 0.015625)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--state-file", type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.state_file.resolve() if a.state_file else None), indent=2))
