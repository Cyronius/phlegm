"""Check six GPT-OSS source output GEMVs and NPU residual against decoded q4."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from designs.gptoss_o_gemv.check import fixture  # noqa: E402
from gptoss_o_run import STARTS, run  # noqa: E402
from recipes.gptoss_source import GptOssSource  # noqa: E402


def check(driver: Path, o_build: Path, residual_build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261029)
    activation = rng.normal(0, 0.5, 4096).astype(np.float32).astype(bfloat16)
    residual = rng.normal(0, 0.5, 2880).astype(np.float32)
    bias = rng.normal(0, 0.05, 2880).astype(np.float32).astype(bfloat16)
    chunks = {}
    reference = []
    for start in STARTS:
        _, selected, expected = fixture(start, activation)
        chunks[start] = selected
        reference.append(expected)

    class FixtureSource(GptOssSource):
        def __init__(self):
            pass

        def constant(self, layer, name):
            if name != "o_bias":
                raise ValueError(name)
            return bias

        def attention(self, layer, role, start):
            if role != "o":
                raise ValueError(role)
            return chunks[start]

    run_report = run(driver, o_build, residual_build, FixtureSource(), 0,
                     activation, residual, out)
    got = np.fromfile(out / "output.f32", np.float32)
    projected = np.concatenate(reference)[:2880]
    want = residual + (projected + bias.astype(np.float32))
    error = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
    scale = max(float(np.max(np.abs(want))), 1e-20)
    report = {"projection_calls": run_report["projection_calls"],
              "residual_calls": run_report["residual_calls"],
              "max_relative_error": error / scale,
              "passed": bool(error / scale < 0.001)}
    (out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "o-build", "residual-build", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.driver.resolve(), a.o_build.resolve(),
                           a.residual_build.resolve(), a.out.resolve()), indent=2))
