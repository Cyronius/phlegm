"""Check the 2880-value GPT-OSS projection bias and residual NPU kernel."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HIDDEN = 2880


def cases() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20261027)
    random = [(f"random-{i}",
               rng.normal(0, 3, HIDDEN).astype(np.float32),
               rng.normal(0, 3, HIDDEN).astype(np.float32),
               rng.normal(0, 0.2, HIDDEN).astype(np.float32).astype(bfloat16))
              for i in range(3)]
    # A different association order would preserve the tiny bias here.
    cancellation = ("cancellation", np.full(HIDDEN, 1_000_000, np.float32),
                    np.full(HIDDEN, -1_000_000, np.float32),
                    np.full(HIDDEN, 0.01, np.float32).astype(bfloat16))
    zero_bias = ("zero-bias", rng.normal(0, 1, HIDDEN).astype(np.float32),
                 rng.normal(0, 1, HIDDEN).astype(np.float32),
                 np.zeros(HIDDEN, bfloat16))
    return [*random, cancellation, zero_bias]


def run(driver: Path, build: Path, out: Path) -> dict:
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    results = []
    for name, residual, projection, bias in cases():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        pair = np.ascontiguousarray(np.stack([residual.reshape(45, 64),
                                              projection.reshape(45, 64)], axis=1))
        for label, array in (("pair", pair), ("bias", bias)):
            (d / f"{label}.bin").write_bytes(array.tobytes())
        lines = ["device", f"xclbin G {build / 'final.xclbin'}",
                 f"kernelx k G {build / 'insts.bin'}"]
        for label, array in (("pair", pair), ("bias", bias)):
            lines.append(f"buf {label} {array.nbytes} {d / f'{label}.bin'}")
        lines += [f"buf result {HIDDEN * 4}",
                  "run k pair bias result",
                  f"dump result {d / 'result.bin'} {HIDDEN * 4}", ""]
        config = d / "run.cfg"
        config.write_text("\n".join(lines))
        completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"{name}: harness failed: {completed.stdout}\n{completed.stderr}")
        actual = np.fromfile(d / "result.bin", dtype=np.float32)
        if actual.shape != (HIDDEN,) or not np.isfinite(actual).all():
            raise AssertionError(f"{name}: invalid result")
        expected = residual + (projection + bias.astype(np.float32))
        err = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        max_abs = float(err.max())
        if max_abs > 1e-5:
            raise AssertionError(f"{name}: maximum absolute error {max_abs}")
        results.append({"case": name, "max_absolute_error": max_abs})
    out.mkdir(parents=True, exist_ok=True)
    report = {"passed": True, "reference": "FP32 residual + (projection + BF16 bias)",
              "cases": results}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
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
