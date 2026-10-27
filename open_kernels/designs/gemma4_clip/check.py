"""NPU check for Gemma 4 vision/audio pre/post linear clamping."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from open_kernels.gemma4_clip_reference import clip, limits  # noqa: E402


def run(driver: Path, build: Path, out: Path, width: int, post_linear: bool) -> dict:
    rng = np.random.default_rng(20261027 + width + int(post_linear))
    data = rng.normal(0, 1.2, (256, width)).astype(np.float32)
    values = data if post_linear else data.astype(bfloat16)
    bounds = limits(-0.625, 0.8125)
    expected = clip(values, bounds, post_linear=post_linear)
    out.mkdir(parents=True, exist_ok=True)
    (out / "values.bin").write_bytes(values.tobytes())
    (out / "bounds.bin").write_bytes(bounds.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf values {values.nbytes} {out / 'values.bin'}",
        f"buf bounds {bounds.nbytes} {out / 'bounds.bin'}",
        f"buf result {expected.nbytes}", "run k values bounds result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16).reshape(expected.shape)
    error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
    report = {"rows": 256, "width": width, "post_linear": post_linear,
              "max_absolute_error": error,
              "exact_bf16_fraction": float(np.mean(actual == expected)),
              "passed": bool(np.isfinite(actual.astype(np.float32)).all() and error == 0)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--post-linear", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve(),
                         args.width, args.post_linear), indent=2))
