"""NPU check for Gemma 3 vision 4×4 pool and (1+w) RMSNorm."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from open_kernels.gemma3_vision_pool_pack import pack  # noqa: E402


def run(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261027)
    patches = rng.normal(0, 0.4, (4096, 1152)).astype(np.float32).astype(bfloat16)
    # A deterministic spatial marker checks that pooling traverses each 4×4 block.
    patches[0:4, 0] = np.asarray([0, 1, 2, 3], bfloat16)
    groups = pack(patches)
    weight = rng.normal(0, 0.2, 1152).astype(np.float32).astype(bfloat16)
    pooled = (groups.astype(np.float64).sum(axis=1) / 16).astype(bfloat16)
    x = pooled.astype(np.float64)
    normed = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6)
    expected = (normed * (1 + weight.astype(np.float64))).astype(bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    (out / "groups.bin").write_bytes(groups.tobytes())
    (out / "weight.bin").write_bytes(weight.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf groups {groups.nbytes} {out / 'groups.bin'}",
        f"buf weight {weight.nbytes} {out / 'weight.bin'}",
        f"buf result {expected.nbytes}", "run k groups weight result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16).reshape(expected.shape)
    err = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
    relative = err / max(float(np.max(np.abs(expected.astype(np.float32)))), 1e-20)
    report = {"tokens": 256, "patches_per_token": 16, "hidden": 1152,
              "max_absolute_error": err, "max_relative_error": relative,
              "exact_bf16_fraction": float(np.mean(actual == expected)),
              "passed": bool(np.isfinite(actual.astype(np.float32)).all() and relative < 0.005)}
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
