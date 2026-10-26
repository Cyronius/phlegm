"""NPU check for GPT-OSS 20B's biased 2880-to-32 top-four router."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def cases() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20261026)
    rows = []
    for i in range(3):
        x = rng.normal(0, 1, 2880).astype(np.float32).astype(bfloat16)
        w = rng.normal(0, 0.03, (2880, 32)).astype(np.float32).astype(bfloat16)
        b = rng.normal(0, 0.1, 32).astype(np.float32)
        if i == 1:
            b[[3, 17, 9, 26]] += [12, 10, 8, 6]
        rows.append((f"random-{i}", x, w, b.astype(bfloat16)))
    rows.append(("stable-tie", np.zeros(2880, dtype=bfloat16),
                 np.zeros((2880, 32), dtype=bfloat16),
                 np.zeros(32, dtype=bfloat16)))
    return rows


def validate(ids: np.ndarray, weights: np.ndarray, logits: np.ndarray,
             expected: np.ndarray) -> tuple[float, float]:
    if (ids.shape != (4,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= 32) or np.unique(ids).size != 4):
        raise AssertionError("expected four distinct integer expert IDs in [0,32)")
    for label, values, count in (("weights", weights, 4), ("logits", logits, 32),
                                 ("reference logits", expected, 32)):
        if values.shape != (count,) or not np.isfinite(values).all():
            raise AssertionError(f"{label}: expected {count} finite values")
    if (np.any(weights < 0) or np.any(weights > 1)
            or abs(float(weights.astype(np.float64).sum()) - 1.0) > 0.002):
        raise AssertionError("router weights must be in [0,1] and sum to one within 0.002")
    top = np.argsort(-expected, kind="stable")[:4]
    expv = np.exp(expected[top] - expected[top].max())
    ref_weights = expv / expv.sum()
    if not np.isfinite(ref_weights).all():
        raise AssertionError("nonfinite reference weights")
    max_logit_error = float(np.max(np.abs(logits.astype(np.float64) - expected)))
    max_weight_error = float(np.max(np.abs(weights.astype(np.float64) - ref_weights)))
    if (not np.array_equal(ids, top) or not np.isfinite(max_logit_error)
            or not np.isfinite(max_weight_error) or max_logit_error > 0.002 or max_weight_error > 0.002):
        raise AssertionError(f"ids {ids} vs {top}, logit {max_logit_error}, weight {max_weight_error}")
    return max_logit_error, max_weight_error


def run(driver: Path, build: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, x, w, b in cases():
        for label, values, shape in (("x", x, (2880,)), ("w", w, (2880, 32)), ("bias", b, (32,))):
            if values.shape != shape or not np.isfinite(values.astype(np.float64)).all():
                raise AssertionError(f"{name}: invalid {label} input")
        d = out / name
        d.mkdir(exist_ok=True)
        x_bias = np.concatenate([x, b])
        for key, arr in (("x_bias", x_bias), ("w", w)):
            (d / f"{key}.bin").write_bytes(arr.tobytes())
        config = d / "run.cfg"
        config.write_text("\n".join([
            "device", f"xclbin G {build / 'final.xclbin'}",
            f"kernelx k G {build / 'insts.bin'}",
            f"buf x_bias {x_bias.nbytes} {d / 'x_bias.bin'}",
            f"buf w {w.nbytes} {d / 'w.bin'}", "buf result 256",
            "run k x_bias w result", f"dump result {d / 'result.bin'} 256", "",
        ]))
        completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"{name}: harness failed: {completed.stdout}\n{completed.stderr}")
        got = (d / "result.bin").read_bytes()
        if len(got) != 256:
            raise AssertionError(f"{name}: expected 256 output bytes, got {len(got)}")
        ids = np.frombuffer(got, dtype=np.int32, count=4).copy()
        weights = np.frombuffer(got, dtype=np.float32, count=4, offset=16).copy()
        logits = np.frombuffer(got, dtype=np.float32, count=32, offset=32).copy()
        expected = x.astype(np.float64) @ w.astype(np.float64)
        expected += b.astype(np.float64)
        max_logit_error, max_weight_error = validate(ids, weights, logits, expected)
        results.append({"case": name, "ids": ids.tolist(),
                        "max_logit_error": max_logit_error,
                        "max_weight_error": max_weight_error})
    report = {"passed": True, "reference": "FP64 biased 2880x32 router and stable top-four softmax",
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
