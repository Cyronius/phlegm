"""Check Gemma 4 FLM-layout BF16 per-layer matrix-vector products on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

SHAPES = ((256, 1536), (256, 2560), (1536, 256), (2560, 256))


def fixture(n: int, k: int):
    rng = np.random.default_rng(20261027 + n * 10 + k)
    source = rng.normal(0, 0.025, (n // 32, k, 32)).astype(np.float32).astype(bfloat16)
    x = rng.normal(0, 0.7, k).astype(np.float32).astype(bfloat16)
    raster = source.transpose(0, 2, 1).reshape(n, k)
    expected = (raster.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    return source, x, expected


def run(driver: Path, build: Path, out: Path, n: int, k: int) -> dict:
    if (n, k) not in SHAPES:
        raise ValueError((n, k))
    source, x, expected = fixture(n, k)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weight.bin").write_bytes(source.tobytes())
    (out / "activation.bin").write_bytes(x.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf weight {source.nbytes} {out / 'weight.bin'}",
        f"buf activation {x.nbytes} {out / 'activation.bin'}",
        f"buf result {expected.nbytes}", "run k weight activation result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("invalid BF16 GEMV output")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    maximum = float(np.max(np.abs(a - e)))
    relative = maximum / max(float(np.max(np.abs(e))), 1e-20)
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    report = {"rows": n, "cols": k, "max_absolute_error": maximum,
              "max_relative_error": relative, "cosine": cosine,
              "passed": bool(relative < 0.0005 and cosine > 0.999999),
              "reference": "FP64 matvec of FLM's [N/32,K,32] BF16 layout"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rows", required=True, type=int)
    p.add_argument("--cols", required=True, type=int)
    args = p.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve(),
                         args.rows, args.cols), indent=2))
