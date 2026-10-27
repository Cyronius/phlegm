"""Schedule all 16 SigLIP heads and 16 query panels with the NPU attention bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_attention_check import SCALE, maxrel, run_image
from open_kernels.gemma3_vision_attention_pack import key_weight, query_panel, value_weight
from open_kernels.gemma4_mm_source import npu_weight_row_major


def synthetic_qkv() -> np.ndarray:
    rng = np.random.default_rng(20261028)
    qkv = np.zeros((4096, 3584), bfloat16)
    for part, magnitude in ((0, 0.8), (1, 0.8), (2, 0.4)):
        begin = part * 1152
        qkv[:, begin:begin + 1152] = rng.normal(
            0, magnitude, (4096, 1152)).astype(np.float32).astype(bfloat16)
    return qkv


def selected_reference(qkv: np.ndarray, output: np.ndarray) -> dict:
    cases = []
    for head in range(16):
        first = head * 72
        for row in (0, 257, 4095):
            query = qkv[row, first:first + 72].astype(np.float64)
            key = qkv[:, 1152 + first:1152 + first + 72].astype(np.float64)
            value = qkv[:, 2304 + first:2304 + first + 72].astype(np.float64)
            score = key @ query * SCALE
            score -= score.max()
            exp = np.exp(score)
            probability = (exp / exp.sum()).astype(bfloat16)
            reference = (probability.astype(np.float64) @ value).astype(np.float32).astype(bfloat16)
            got = output[row, first:first + 72]
            cases.append({"head": head, "row": row,
                          "max_relative_error": maxrel(got, reference)})
    return {"cases": cases, "max_relative_error": max(case["max_relative_error"] for case in cases)}


def run(driver: Path, bundle: Path, out: Path, qkv_path: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    synthetic = qkv_path is None
    qkv = synthetic_qkv() if synthetic else np.fromfile(qkv_path, bfloat16).reshape(4096, 3584)
    output = np.empty((4096, 1152), bfloat16)
    work = out / "work"
    for head in range(16):
        key = npu_weight_row_major(key_weight(qkv, head), 4096, 128)
        value = npu_weight_row_major(value_weight(qkv, head), 256, 4096)
        for start in range(0, 4096, 256):
            query = query_panel(qkv, head, start)
            scores_path = run_image(driver, bundle / "qk_m256_k128_n4096", work / "qk",
                                    [("query", query), ("key", key)], 256 * 4096 * 4)
            scores = np.fromfile(scores_path, np.float32).reshape(256, 4096)
            probabilities_path = run_image(driver, bundle / "softmax_256x4096",
                                           work / "softmax", [("scores", scores)],
                                           256 * 4096 * 2)
            probabilities = np.fromfile(probabilities_path, bfloat16).reshape(256, 4096)
            pv_path = run_image(driver, bundle / "pv_m256_k4096_n256", work / "pv",
                                [("probabilities", probabilities), ("value", value)],
                                256 * 256 * 4)
            projected = np.fromfile(pv_path, np.float32).reshape(256, 256)
            if not (np.isfinite(projected).all() and np.all(projected[:, 72:] == 0)):
                raise AssertionError(f"invalid attention output at head {head}, row {start}")
            output[start:start + 256, head * 72:(head + 1) * 72] = projected[:, :72].astype(bfloat16)
        print(f"completed head {head + 1}/16", flush=True)
    (out / "result.bin").write_bytes(output.tobytes())
    reference = selected_reference(qkv, output) if synthetic else None
    report = {"heads": 16, "head_dim": 72, "queries": 4096, "keys": 4096,
              "query_panels": 16, "kernel_dispatches": 16 * 16 * 3,
              "synthetic": synthetic, "sampled_reference": reference,
              "passed": bool(np.isfinite(output.astype(np.float32)).all() and
                             (reference is None or reference["max_relative_error"] < 0.01))}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--qkv", type=Path, help="BF16[4096,3584] after QKV bias; synthetic if absent")
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.bundle.resolve(), a.out.resolve(),
                         a.qkv.resolve() if a.qkv else None), indent=2))
