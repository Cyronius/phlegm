"""Check the ten-call GPT-OSS source QKV runner against decoded q4 weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from designs.gptoss_qkv_gemv.check import fixture  # noqa: E402
from gptoss_qkv_run import CALLS, run  # noqa: E402
from recipes.gptoss_source import GptOssSource  # noqa: E402


def check(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261027)
    activation = rng.normal(0, 0.5, 2880).astype(np.float32).astype(bfloat16)
    chunks = {}
    bias = {role: np.empty(4096 if role == "q" else 512, bfloat16) for role in CALLS}
    expected = {}
    for role, starts in CALLS.items():
        pieces = []
        for start in starts:
            packed, selected, reference = fixture(role, start, 16, activation)
            chunks[role, start] = selected
            bias[role][start * 32:(start + 16) * 32] = packed[2944:]
            pieces.append(reference)
        expected[role] = np.concatenate(pieces)

    class FixtureSource(GptOssSource):
        def __init__(self):
            pass

        def constant(self, layer, name):
            return bias[name[0]]

        def attention(self, layer, role, start):
            return chunks[role, start]

    run_report = run(driver, build, FixtureSource(), 0, activation, out)
    got_qk = np.fromfile(out / "qk.bf16", bfloat16).astype(np.float32)
    got_v = np.fromfile(out / "v.bf16", bfloat16).astype(np.float32)
    want_qk = np.concatenate((expected["q"], expected["k"])).astype(bfloat16).astype(np.float32)
    want_v = expected["v"].astype(bfloat16).astype(np.float32)
    error = max(float(np.max(np.abs(got_qk - want_qk))),
                float(np.max(np.abs(got_v - want_v))))
    scale = max(float(np.max(np.abs(want_qk))), float(np.max(np.abs(want_v))), 1e-20)
    report = {"kernel_calls": run_report["kernel_calls"],
              "max_relative_error": error / scale,
              "passed": bool(error / scale < 0.005)}
    (out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "build", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.driver.resolve(), a.build.resolve(), a.out.resolve()), indent=2))
