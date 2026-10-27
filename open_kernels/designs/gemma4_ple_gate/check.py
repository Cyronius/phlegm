"""Check E2B/E4B per-layer input GELU gate on NPU against BF16 boundaries."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def run(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261027)
    projection = rng.normal(0, 1.5, 256).astype(np.float32).astype(bfloat16)
    layer_input = rng.normal(0, 0.8, 256).astype(np.float32).astype(bfloat16)
    projection[:8] = np.array([0, -0.0, 0.01, -0.01, 3, -3, 8, -8], np.float32).astype(bfloat16)
    p = projection.astype(np.float64)
    z = np.sqrt(2 / np.pi) * (p + 0.044715 * p**3)
    activated = (0.5 * p * (1 + np.tanh(z))).astype(bfloat16)
    expected = (activated.astype(np.float64) * layer_input.astype(np.float64)).astype(bfloat16)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "projection.bin").write_bytes(projection.tobytes())
    (out / "layer_input.bin").write_bytes(layer_input.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf projection {projection.nbytes} {out / 'projection.bin'}",
        f"buf layer_input {layer_input.nbytes} {out / 'layer_input.bin'}",
        f"buf result {projection.nbytes}", "run k projection layer_input result",
        f"dump result {out / 'result.bin'} {projection.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16)
    if actual.shape != expected.shape or not np.isfinite(actual.astype(np.float32)).all():
        raise AssertionError("invalid gate output")
    a, e = actual.astype(np.float32), expected.astype(np.float32)
    maximum = float(np.max(np.abs(a - e)))
    cosine = float(a.astype(np.float64) @ e.astype(np.float64) /
                   (np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(e.astype(np.float64))))
    report = {"width": 256, "max_absolute_error": maximum, "cosine": cosine,
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
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
