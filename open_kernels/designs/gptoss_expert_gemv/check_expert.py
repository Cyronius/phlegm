"""Run one complete GPT-OSS expert FFN through four NPU kernel invocations.

This checks arithmetic and the FP32-to-BF16 activation boundary. The router,
expert selection and production layer runtime are outside this check.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "model"))
from replica_gptoss import clamped_swiglu  # noqa: E402
from check import fixture, reference  # noqa: E402

HIDDEN = 2880
PADDED = 23 * 128


def invoke(driver: Path, build: Path, out: Path, inputs: list[np.ndarray],
           output_dtype: np.dtype, output_count: int) -> np.ndarray:
    out.mkdir(parents=True, exist_ok=True)
    lines = ["device", f"xclbin G {build / 'final.xclbin'}",
             f"kernelx k G {build / 'insts.bin'}"]
    names = []
    for i, array in enumerate(inputs):
        name = f"input{i}"
        path = out / f"{name}.bin"
        path.write_bytes(array.tobytes())
        lines.append(f"buf {name} {array.nbytes} {path}")
        names.append(name)
    nbytes = output_count * np.dtype(output_dtype).itemsize
    lines += [f"buf result {nbytes}", f"run k {' '.join(names)} result",
              f"dump result {out / 'result.bin'} {nbytes}", ""]
    config = out / "run.cfg"
    config.write_text("\n".join(lines))
    completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{out.name}: harness failed: {completed.stdout}\n{completed.stderr}")
    result = np.fromfile(out / "result.bin", dtype=output_dtype)
    if result.size != output_count or not np.isfinite(result.astype(np.float32)).all():
        raise AssertionError(f"{out.name}: invalid NPU output")
    return result


def score(actual: np.ndarray, expected: np.ndarray) -> dict:
    a = actual.astype(np.float64)
    e = expected.astype(np.float64)
    error = np.abs(a - e)
    return {"max_absolute_error": float(error.max()),
            "max_relative_error": float(error.max() / max(np.max(np.abs(e)), 1e-30)),
            "cosine": float((a @ e) / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-30))}


def run(driver: Path, projection_build: Path, gate_build: Path, out: Path) -> dict:
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for build in (projection_build, gate_build):
        for name in ("final.xclbin", "insts.bin"):
            if not (build / name).is_file():
                raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    x, gate_chunks = fixture(90, 20261027)
    _, up_chunks = fixture(90, 20261028)
    _, down_chunks = fixture(90, 20261029)
    gate = invoke(driver, projection_build, out / "gate_projection",
                  [x, gate_chunks], np.float32, HIDDEN)
    up = invoke(driver, projection_build, out / "up_projection",
                [x, up_chunks], np.float32, HIDDEN)
    gate_ref = reference(x, gate_chunks).ravel()
    up_ref = reference(x, up_chunks).ravel()
    hidden = invoke(driver, gate_build, out / "activation",
                    [gate, up], bfloat16, HIDDEN)
    hidden_ref = clamped_swiglu(gate_ref, up_ref).astype(bfloat16)
    down_input = np.zeros(PADDED, dtype=bfloat16)
    down_input[:HIDDEN] = hidden
    down = invoke(driver, projection_build, out / "down_projection",
                  [down_input, down_chunks], np.float32, HIDDEN)
    down_ref_input = np.zeros(PADDED, dtype=bfloat16)
    down_ref_input[:HIDDEN] = hidden_ref
    down_ref = reference(down_ref_input, down_chunks).ravel()
    results = {"gate_projection": score(gate, gate_ref),
               "up_projection": score(up, up_ref),
               "activation": score(hidden, hidden_ref),
               "down_projection": score(down, down_ref)}
    limits = {"gate_projection": 0.005, "up_projection": 0.005,
              "activation": 0.015, "down_projection": 0.02}
    passed = all(results[name]["max_relative_error"] < limit and
                 results[name]["cosine"] > 0.999 for name, limit in limits.items())
    report = {"passed": passed, "stages": results,
              "reference": "FP64 decoded MXFP4 products and GPT-OSS clamp with BF16 activation boundary",
              "scope": "one expert; no routing, selection, layer runtime or model acceptance"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--projection-build", required=True, type=Path)
    p.add_argument("--gate-build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.projection_build.resolve(),
                         a.gate_build.resolve(), a.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
