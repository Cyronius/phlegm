"""Check Gemma 3 4B pool/RMSNorm -> BF16 projector GEMM on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_kernels.designs.gemma3_vision_pool_norm.check import run as pool_run  # noqa: E402
from open_kernels.gemma4_mm_source import npu_weight_row_major  # noqa: E402


def run(driver: Path, pool_build: Path, mm_build: Path, out: Path) -> dict:
    pool = pool_run(driver, pool_build, out / "pool")
    activation = np.fromfile(out / "pool" / "result.bin", bfloat16).reshape(256, 1152)
    rng = np.random.default_rng(20261028)
    weights = rng.normal(0, 0.02, (2560, 1152)).astype(np.float32).astype(bfloat16)
    packed = npu_weight_row_major(weights, 2560, 1152)
    expected = (activation.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(packed.tobytes())
    cfg = out / "projector.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {mm_build / 'final.xclbin'}",
        f"kernelx k G {mm_build / 'insts.bin'}",
        f"buf activation {activation.nbytes} {out / 'pool' / 'result.bin'}",
        f"buf weights {packed.nbytes} {out / 'weights.bin'}",
        f"buf result {expected.nbytes}", "run k activation weights result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32).reshape(expected.shape)
    error = float(np.max(np.abs(actual - expected)))
    relative = error / max(float(np.max(np.abs(expected))), 1e-20)
    report = {"pool": pool, "projection": {"m": 256, "k": 1152, "n": 2560,
              "max_absolute_error": error, "max_relative_error": relative},
              "passed": bool(pool["passed"] and np.isfinite(actual).all() and
                             relative < 0.005)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--pool-build", required=True, type=Path)
    p.add_argument("--mm-build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.pool_build.resolve(),
                         a.mm_build.resolve(), a.out.resolve()), indent=2))
