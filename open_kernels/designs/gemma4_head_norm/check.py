"""Check one Gemma 4 per-head norm image with learned and unit scales."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def fixture(heads: int, width: int, unit: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261027 + heads * width + int(unit))
    x = rng.normal(0, 0.7, (heads, width)).astype(np.float32).astype(bfloat16)
    w = (np.ones(width, np.float32) if unit else
         rng.uniform(0.6, 1.4, width).astype(np.float32)).astype(bfloat16)
    a = x.astype(np.float64)
    inv = np.power(np.mean(a * a, axis=1, keepdims=True) + 1e-6, -0.5)
    expected = (a * inv * w.astype(np.float64)).astype(bfloat16)
    return x, w, expected


def run(driver: Path, build: Path, out: Path, heads: int, width: int, unit: bool) -> dict:
    x, w, expected = fixture(heads, width, unit)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.bin").write_bytes(x.tobytes())
    (out / "scale.bin").write_bytes(w.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf input {x.nbytes} {out / 'input.bin'}",
        f"buf scale {w.nbytes} {out / 'scale.bin'}",
        f"buf result {x.nbytes}", "run k input scale result",
        f"dump result {out / 'result.bin'} {x.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", dtype=bfloat16)
    if actual.size != expected.size or not np.isfinite(actual.astype(np.float32)).all():
        raise AssertionError("invalid NPU result")
    a, e = actual.astype(np.float32), expected.ravel().astype(np.float32)
    cosine = float(np.dot(a.astype(np.float64), e.astype(np.float64)) /
                   (np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(e.astype(np.float64))))
    report = {"heads": heads, "head_dim": width, "unit_scale": unit,
              "max_absolute_error": float(np.max(np.abs(a - e))),
              "exact_bf16_fraction": float(np.mean(a == e)), "cosine": cosine,
              "passed": bool(np.max(np.abs(a - e)) <= 0.015 and cosine > 0.99999)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--heads", required=True, type=int, choices=(1, 2, 8, 12, 16))
    p.add_argument("--head-dim", required=True, type=int, choices=(64, 256, 512))
    p.add_argument("--unit-scale", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.heads, a.head_dim, a.unit_scale), indent=2))


if __name__ == "__main__":
    main()
