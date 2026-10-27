"""Validate Gemma 4 audio depthwise 5-tap convolution on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from open_kernels.gemma4_audio_dw_pack import pack, unpack  # noqa: E402


def run(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261027)
    x = rng.normal(0, 0.2, (260, 1024)).astype(np.float32).astype(bfloat16)
    w = rng.normal(0, 0.06, (5, 1024)).astype(np.float32).astype(bfloat16)
    panels, weights = pack(x, w)
    expected = sum(x[t:t + 256].astype(np.float64) * w[t].astype(np.float64)
                   for t in range(5)).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    (out / "panels.bin").write_bytes(panels.tobytes())
    (out / "weights.bin").write_bytes(weights.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf panels {panels.nbytes} {out / 'panels.bin'}",
        f"buf weights {weights.nbytes} {out / 'weights.bin'}",
        f"buf result {expected.nbytes}", "run k panels weights result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = unpack(np.fromfile(out / "result.bin", np.float32))
    if not np.isfinite(actual).all():
        raise AssertionError("nonfinite audio convolution output")
    maximum = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    scale = float(np.max(np.abs(expected)))
    report = {"frames": 256, "channels": 1024, "taps": 5,
              "max_absolute_error": maximum,
              "max_relative_error": maximum / max(scale, 1e-20),
              "passed": bool(maximum / max(scale, 1e-20) < 1e-5)}
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
