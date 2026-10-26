"""Check generated Embedding-Gemma NPU matmuls against fp64 products.

All fixtures and device outputs are written below --out, normally target/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from gemma_shapes import SHAPES, tag


def check(shape: tuple[int, int, int], assets: Path, driver: Path,
          output: Path, seed: int) -> dict:
    m, k, n = shape
    name = tag(shape)
    d = output / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((m, k)) * 0.1).astype(np.float32).astype(bfloat16)
    b = (rng.standard_normal((k, n)) * 0.1).astype(np.float32).astype(bfloat16)
    ref = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    (d / "a.bin").write_bytes(a.tobytes())
    (d / "b.bin").write_bytes(b.tobytes())
    xclbin = assets / f"{name}.xclbin"
    insts = assets / f"{name}.insts"
    if not xclbin.is_file() or not insts.is_file():
        raise FileNotFoundError(f"{name}: build {xclbin} and {insts} first")
    config = d / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {xclbin}", f"kernelx k G {insts}",
        f"buf a {a.nbytes} {d / 'a.bin'}", f"buf b {b.nbytes} {d / 'b.bin'}",
        f"buf c {ref.nbytes}", "run k a b c", "run k a b c",
        f"dump c {d / 'result.bin'} {ref.nbytes}", "",
    ]))
    result = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{name}: harness failed: {result.stdout}\n{result.stderr}")
    got = np.fromfile(d / "result.bin", np.float32)
    if got.size != ref.size or not np.isfinite(got).all():
        raise AssertionError(f"{name}: invalid output count or nonfinite value")
    got64, ref64 = got.astype(np.float64), ref.astype(np.float64).ravel()
    max_abs = float(np.max(np.abs(got64 - ref64)))
    max_rel = max_abs / (float(np.max(np.abs(ref64))) + 1e-30)
    cosine = float(got64 @ ref64 / (np.linalg.norm(got64) * np.linalg.norm(ref64) + 1e-30))
    passed = cosine > 0.99999 and max_rel < 5e-3
    row = {"shape": name, "passed": passed, "cosine": cosine,
           "max_absolute_error": max_abs, "max_relative_error": max_rel}
    (d / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    if not passed:
        raise AssertionError(row)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", required=True, type=Path)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--all", action="store_true", help="check all twelve packaged shapes")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    shapes = SHAPES if args.all else [SHAPES[0]]
    rows = [check(shape, args.assets.resolve(), args.driver.resolve(),
                  args.out.resolve(), args.seed) for shape in shapes]
    report = {"passed": True, "shapes": rows}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
