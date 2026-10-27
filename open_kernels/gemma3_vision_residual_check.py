"""Reuse the 1152-channel two-addend image for SigLIP residual boundaries."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def run(driver: Path, build: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    projected = rng.normal(0, 0.4, (256, 1280)).astype(np.float32)
    bias = rng.normal(0, 0.08, 1152).astype(np.float32).astype(bfloat16)
    residual = rng.normal(0, 0.7, (256, 1152)).astype(np.float32).astype(bfloat16)
    addends = np.empty((256, 2, 1152), bfloat16)
    addends[:, 0] = bias
    addends[:, 1] = residual
    biased = (projected[:, :1152] + bias.astype(np.float32)).astype(bfloat16)
    expected = (biased.astype(np.float32) + residual.astype(np.float32)).astype(bfloat16)
    (out / "projected.bin").write_bytes(projected.tobytes())
    (out / "addends.bin").write_bytes(addends.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf projected {projected.nbytes} {out / 'projected.bin'}",
        f"buf addends {addends.nbytes} {out / 'addends.bin'}",
        f"buf result {expected.nbytes}", "run k projected addends result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16).reshape(expected.shape)
    report = {"rows": 256, "hidden": 1152,
              "exact_bf16_fraction": float(np.mean(actual == expected)),
              "passed": bool(np.array_equal(actual, expected))}
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
