"""Check Gemma 4 hidden RMSNorm and residual BF16 boundaries on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def run(driver: Path, build: Path, out: Path, hidden: int, zero_residual: bool) -> dict:
    rng = np.random.default_rng(20261027 + hidden + int(zero_residual))
    value = rng.normal(0, 0.7, hidden).astype(np.float32).astype(bfloat16)
    residual = (np.zeros(hidden, bfloat16) if zero_residual else
                rng.normal(0, 0.7, hidden).astype(np.float32).astype(bfloat16))
    scale = rng.uniform(0.7, 1.3, hidden).astype(np.float32).astype(bfloat16)
    x = value.astype(np.float64)
    normalized = (x / np.sqrt(np.mean(x * x) + 1e-6) * scale.astype(np.float64)).astype(bfloat16)
    expected = (residual.astype(np.float64) + normalized.astype(np.float64)).astype(bfloat16)
    pair = np.concatenate((value, residual))
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pair.bin").write_bytes(pair.tobytes())
    (out / "scale.bin").write_bytes(scale.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf pair {pair.nbytes} {out / 'pair.bin'}",
        f"buf scale {scale.nbytes} {out / 'scale.bin'}",
        f"buf result {value.nbytes}", "run k pair scale result",
        f"dump result {out / 'result.bin'} {value.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16)
    if actual.shape != expected.shape or not np.isfinite(actual.astype(np.float32)).all():
        raise AssertionError("invalid hidden norm output")
    a, e = actual.astype(np.float32), expected.astype(np.float32)
    maximum = float(np.max(np.abs(a - e)))
    cosine = float(a.astype(np.float64) @ e.astype(np.float64) /
                   (np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(e.astype(np.float64))))
    report = {"hidden": hidden, "zero_residual": zero_residual,
              "max_absolute_error": maximum, "cosine": cosine,
              "exact_bf16_fraction": float(np.mean(a == e)),
              "passed": bool(maximum <= 0.015 and cosine > 0.99999)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hidden", required=True, type=int, choices=(768, 1024, 1536, 2560, 3840))
    p.add_argument("--zero-residual", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.hidden, a.zero_residual), indent=2))


if __name__ == "__main__":
    main()
