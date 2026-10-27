"""Validate Gemma 4 audio SiLU/GLU BF16 activation images on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def run(driver: Path, build: Path, out: Path, width: int, glu: bool) -> dict:
    if (glu and width != 1024) or (not glu and width not in (1024, 4096)):
        raise ValueError("unsupported audio activation geometry")
    rng = np.random.default_rng(20261027 + width + int(glu))
    a = rng.normal(0, 1.5, width).astype(np.float32).astype(bfloat16)
    if glu:
        gate = rng.normal(0, 1.5, width).astype(np.float32).astype(bfloat16)
        data = np.stack((a.reshape(-1, 256), gate.reshape(-1, 256)), axis=1).ravel()
        expected = (a.astype(np.float64) /
                    (1 + np.exp(-gate.astype(np.float64)))).astype(bfloat16)
    else:
        data = a
        expected = (a.astype(np.float64) /
                    (1 + np.exp(-a.astype(np.float64)))).astype(bfloat16)
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
    delta = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    report = dict(width=width, mode="glu" if glu else "silu",
                  max_absolute_error=float(delta.max()),
                  exact_bf16_fraction=float(np.mean(delta == 0)),
                  passed=bool(np.isfinite(actual.astype(np.float32)).all() and delta.max() < 0.016))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--glu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(),
                         args.out.resolve(), args.width, args.glu), indent=2))
