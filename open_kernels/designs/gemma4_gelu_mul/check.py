"""Check a full Gemma 4 MLP GELU-tanh gate and up product on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

WIDTHS = (3072, 6144, 10240, 12288, 15360)


def run(driver: Path, build: Path, out: Path, width: int) -> dict:
    if width not in WIDTHS:
        raise ValueError(width)
    rng = np.random.default_rng(20261027 + width)
    gate = rng.normal(0, 1.5, width).astype(np.float32).astype(bfloat16)
    up = rng.normal(0, 0.8, width).astype(np.float32).astype(bfloat16)
    g = gate.astype(np.float64)
    z = np.sqrt(2 / np.pi) * (g + 0.044715 * g**3)
    activated = (0.5 * g * (1 + np.tanh(z))).astype(bfloat16)
    expected = (activated.astype(np.float64) * up.astype(np.float64)).astype(bfloat16)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.bin").write_bytes(gate.tobytes())
    (out / "up.bin").write_bytes(up.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf gate {gate.nbytes} {out / 'gate.bin'}",
        f"buf up {up.nbytes} {out / 'up.bin'}",
        f"buf result {gate.nbytes}", "run k gate up result",
        f"dump result {out / 'result.bin'} {gate.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16)
    if actual.shape != expected.shape or not np.isfinite(actual.astype(np.float32)).all():
        raise AssertionError("invalid MLP gate output")
    a, e = actual.astype(np.float32), expected.astype(np.float32)
    maximum = float(np.max(np.abs(a - e)))
    cosine = float(a.astype(np.float64) @ e.astype(np.float64) /
                   (np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(e.astype(np.float64))))
    report = {"width": width, "chunks": width // 256,
              "max_absolute_error": maximum, "cosine": cosine,
              "exact_bf16_fraction": float(np.mean(a == e)),
              "passed": bool(maximum <= 0.03 and cosine > 0.99999)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--width", required=True, type=int, choices=WIDTHS)
    args = p.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve(),
                         args.width), indent=2))
