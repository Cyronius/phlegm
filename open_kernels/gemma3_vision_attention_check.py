"""Check one 256-query × 4096-key SigLIP attention head end to end on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_attention_pack import key_weight, query_panel, value_weight
from open_kernels.gemma4_mm_source import npu_weight_row_major

SCALE = 1 / np.sqrt(72)


def run_image(driver: Path, build: Path, case: Path, args: list[tuple[str, np.ndarray]],
              output_bytes: int) -> Path:
    case.mkdir(parents=True, exist_ok=True)
    lines = ["device", f"xclbin G {build / 'final.xclbin'}",
             f"kernelx k G {build / 'insts.bin'}"]
    for name, value in args:
        path = case / f"{name}.bin"
        path.write_bytes(value.tobytes())
        lines.append(f"buf {name} {value.nbytes} {path}")
    lines += [f"buf result {output_bytes}",
              "run k " + " ".join(name for name, _ in args) + " result",
              f"dump result {case / 'result.bin'} {output_bytes}", ""]
    cfg = case / "run.cfg"
    cfg.write_text("\n".join(lines))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{case.name} NPU failed: {proc.stdout}\n{proc.stderr}")
    return case / "result.bin"


def maxrel(actual: np.ndarray, expected: np.ndarray) -> float:
    error = np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
    scale = max(float(np.max(np.abs(expected.astype(np.float64)))), 1e-20)
    return float(error / scale)


def run(driver: Path, bundle: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    qkv = np.zeros((4096, 3584), bfloat16)
    head, start = 5, 256
    for part, magnitude in ((0, 0.8), (1, 0.8), (2, 0.4)):
        first = part * 1152 + head * 72
        qkv[:, first:first + 72] = rng.normal(0, magnitude, (4096, 72)).astype(np.float32).astype(bfloat16)
    query = query_panel(qkv, head, start)
    key = key_weight(qkv, head)
    value = value_weight(qkv, head)
    tiled_key = npu_weight_row_major(key, 4096, 128)
    scores_path = run_image(driver, bundle / "qk_m256_k128_n4096", out / "qk",
                            [("query", query), ("key", tiled_key)], 256 * 4096 * 4)
    scores = np.fromfile(scores_path, np.float32).reshape(256, 4096)
    probabilities_path = run_image(driver, bundle / "softmax_256x4096", out / "softmax",
                                   [("scores", scores)], 256 * 4096 * 2)
    probabilities = np.fromfile(probabilities_path, bfloat16).reshape(256, 4096)
    tiled_value = npu_weight_row_major(value, 256, 4096)
    output_path = run_image(driver, bundle / "pv_m256_k4096_n256", out / "pv",
                            [("probabilities", probabilities), ("value", tiled_value)],
                            256 * 256 * 4)
    actual = np.fromfile(output_path, np.float32).reshape(256, 256)

    q = query[:, :72].astype(np.float64)
    k = key[:, :72].astype(np.float64)
    v = value[:72].astype(np.float64)
    score_ref = (q @ k.T).astype(np.float32)
    scaled = score_ref.astype(np.float64) * SCALE
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    prob_ref = (np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)).astype(bfloat16)
    output_ref = (prob_ref.astype(np.float64) @ v.T).astype(np.float32)
    actual_softmax = (scores.astype(np.float64) * SCALE)
    actual_softmax -= actual_softmax.max(axis=1, keepdims=True)
    prob_from_actual_scores = (np.exp(actual_softmax) /
                               np.exp(actual_softmax).sum(axis=1, keepdims=True)).astype(bfloat16)
    pv_from_actual_probs = (probabilities.astype(np.float64) @ v.T).astype(np.float32)
    report = {"head": head, "query_start": start, "queries": 256, "keys": 4096,
              "head_dim": 72, "qk_max_relative_error": maxrel(scores, score_ref),
              "softmax_exact_bf16_fraction": float(np.mean(probabilities == prob_from_actual_scores)),
              "softmax_max_absolute_error": float(np.max(np.abs(
                  probabilities.astype(np.float32) - prob_from_actual_scores.astype(np.float32)))),
              "pv_max_relative_error": maxrel(actual[:, :72], pv_from_actual_probs),
              "end_to_end_max_relative_error": maxrel(actual[:, :72], output_ref),
              "padded_output_zero": bool(np.all(actual[:, 72:] == 0))}
    report["passed"] = bool(np.isfinite(actual).all() and
                            report["qk_max_relative_error"] < 0.005 and
                            report["softmax_max_absolute_error"] < 0.005 and
                            report["pv_max_relative_error"] < 0.005 and
                            report["end_to_end_max_relative_error"] < 0.01 and
                            report["padded_output_zero"])
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.bundle.resolve(), a.out.resolve()), indent=2))
