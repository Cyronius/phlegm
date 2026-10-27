"""NPU check for Gemma 4 vision 3x3 patch pooling and BF16 boundary."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from open_kernels.gemma4_vision_pool_pack import pack  # noqa: E402


def run(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261027)
    hidden = rng.normal(0, 0.4, (2520, 768)).astype(np.float32).astype(bfloat16)
    positions = np.array([(x, y) for y in range(60) for x in range(42)], np.int32)
    positions[(positions[:, 0] >= 39) & (positions[:, 1] >= 57)] = -1
    groups, mask = pack(hidden, positions)
    averaged = (groups.astype(np.float64).sum(axis=1) / 9).astype(bfloat16)
    expected = averaged.astype(np.float32) * np.sqrt(np.float32(768))
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.bin").write_bytes(groups.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf input {groups.nbytes} {out / 'input.bin'}",
        f"buf result {expected.nbytes}", "run k input result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32).reshape(expected.shape)
    error = float(np.max(np.abs(actual - expected)))
    relative = error / max(float(np.max(np.abs(expected))), 1e-20)
    report = {"tokens": 280, "patches_per_token": 9, "hidden": 768,
              "valid_tokens": int(mask.sum()), "max_absolute_error": error,
              "max_relative_error": relative,
              "passed": bool(np.isfinite(actual).all() and relative < 0.0001
                             and np.all(actual[~mask] == 0))}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve()), indent=2))
