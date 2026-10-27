"""Compose one complete synthetic Gemma 3 SigLIP encoder layer on NPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_attention_check import maxrel, run_image
from open_kernels.gemma3_vision_attention_run import run as run_attention, selected_reference
from open_kernels.gemma3_vision_tower_weights import pad_matrix
from open_kernels.gemma4_mm_source import npu_weight_row_major


def bf(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, np.float32).astype(bfloat16)


def fixture() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(20261028)
    hidden = bf(rng.normal(0, 0.4, (4096, 1152)))
    weights = {}
    for name in ("ln1", "ln2"):
        weights[name + "_gamma"] = bf(rng.normal(1, 0.03, 1152))
        weights[name + "_beta"] = bf(rng.normal(0, 0.03, 1152))
    for name, shape in (("qkv", (3456, 1152)), ("attention_out", (1152, 1152)),
                        ("mlp_fc1", (4304, 1152)), ("mlp_fc2", (1152, 4304))):
        weights[name + "_weight"] = bf(rng.normal(0, 0.025, shape))
        weights[name + "_bias"] = bf(rng.normal(0, 0.025, shape[0]))
    return hidden, weights


def layernorm_ref(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    y = x.astype(np.float64)
    mean = y.mean()
    variance = np.mean((y - mean) ** 2)
    return ((y - mean) / np.sqrt(variance + 1e-6) *
            gamma.astype(np.float64) + beta.astype(np.float64)).astype(bfloat16)


def downstream_reference(row: int, x: np.ndarray, attention: np.ndarray,
                         weights: dict[str, np.ndarray]) -> np.ndarray:
    projected = weights["attention_out_weight"].astype(np.float64) @ attention[row].astype(np.float64)
    biased = (projected + weights["attention_out_bias"].astype(np.float64)).astype(bfloat16)
    residual = (biased.astype(np.float64) + x[row].astype(np.float64)).astype(bfloat16)
    normalized = layernorm_ref(residual, weights["ln2_gamma"], weights["ln2_beta"])
    fc1 = weights["mlp_fc1_weight"].astype(np.float64) @ normalized.astype(np.float64)
    fc1 = (fc1 + weights["mlp_fc1_bias"].astype(np.float64)).astype(bfloat16)
    h = fc1.astype(np.float64)
    activated = (0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) *
                 (h + 0.044715 * h ** 3)))).astype(bfloat16)
    fc2 = weights["mlp_fc2_weight"].astype(np.float64) @ activated.astype(np.float64)
    fc2 = (fc2 + weights["mlp_fc2_bias"].astype(np.float64)).astype(bfloat16)
    return (fc2.astype(np.float64) + residual.astype(np.float64)).astype(bfloat16)


def read(path: Path, dtype, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype).reshape(shape)


def run(driver: Path, tower: Path, post: Path, attention_bundle: Path,
        residual_build: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    x, weights = fixture()
    params = {name: np.stack((weights[name + "_gamma"], weights[name + "_beta"]))
              for name in ("ln1", "ln2")}
    packed_weights = {}
    for name in ("qkv", "attention_out", "mlp_fc1", "mlp_fc2"):
        padded = pad_matrix(weights[name + "_weight"], name)
        packed_weights[name] = npu_weight_row_major(padded, *padded.shape)
    qkv_bias = np.zeros(3584, bfloat16)
    qkv_bias[:3456] = weights["qkv_bias"]
    fc1_bias = np.zeros(4352, bfloat16)
    fc1_bias[:4304] = weights["mlp_fc1_bias"]
    qkv = np.empty((4096, 3584), bfloat16)
    work = out / "work"

    for start in range(0, 4096, 256):
        panel = x[start:start + 256]
        ln_path = run_image(driver, tower / "layernorm_256x1152", work / "ln1",
                            [("input", panel), ("params", params["ln1"])], 256 * 1152 * 2)
        ln = read(ln_path, bfloat16, (256, 1152))
        qkv_path = run_image(driver, tower / "qkv_m256_k1152_n3584", work / "qkv_mm",
                             [("activation", ln), ("weight", packed_weights["qkv"])],
                             256 * 3584 * 4)
        projected = read(qkv_path, np.float32, (256, 3584))
        bias_path = run_image(driver, post / "qkv_bias_n3584", work / "qkv_bias",
                              [("projected", projected), ("bias", qkv_bias)],
                              256 * 3584 * 2)
        qkv[start:start + 256] = read(bias_path, bfloat16, (256, 3584))
    (out / "qkv.bin").write_bytes(qkv.tobytes())
    run_attention(driver, attention_bundle, out / "attention", out / "qkv.bin")
    attended = read(out / "attention/result.bin", bfloat16, (4096, 1152))
    attention_reference = selected_reference(qkv, attended)
    final = np.empty_like(x)

    for start in range(0, 4096, 256):
        panel = attended[start:start + 256]
        projected_path = run_image(driver, tower / "attention_out_m256_k1152_n1280",
                                   work / "attention_out", [("activation", panel),
                                   ("weight", packed_weights["attention_out"])], 256 * 1280 * 4)
        projected = read(projected_path, np.float32, (256, 1280))
        addends = np.empty((256, 2, 1152), bfloat16)
        addends[:, 0] = weights["attention_out_bias"]
        addends[:, 1] = x[start:start + 256]
        residual_path = run_image(driver, residual_build, work / "attention_residual",
                                  [("projected", projected), ("addends", addends)],
                                  256 * 1152 * 2)
        residual = read(residual_path, bfloat16, (256, 1152))
        ln_path = run_image(driver, tower / "layernorm_256x1152", work / "ln2",
                            [("input", residual), ("params", params["ln2"])],
                            256 * 1152 * 2)
        ln = read(ln_path, bfloat16, (256, 1152))
        fc1_path = run_image(driver, tower / "mlp_fc1_m256_k1152_n4352", work / "fc1_mm",
                             [("activation", ln), ("weight", packed_weights["mlp_fc1"])],
                             256 * 4352 * 4)
        fc1_projected = read(fc1_path, np.float32, (256, 4352))
        activated_path = run_image(driver, post / "fc1_bias_gelu_n4352", work / "fc1_gelu",
                                   [("projected", fc1_projected), ("bias", fc1_bias)],
                                   256 * 4352 * 2)
        activated = read(activated_path, bfloat16, (256, 4352))
        fc2_path = run_image(driver, tower / "mlp_fc2_m256_k4352_n1280", work / "fc2_mm",
                             [("activation", activated), ("weight", packed_weights["mlp_fc2"])],
                             256 * 1280 * 4)
        fc2_projected = read(fc2_path, np.float32, (256, 1280))
        addends[:, 0] = weights["mlp_fc2_bias"]
        addends[:, 1] = residual
        result_path = run_image(driver, residual_build, work / "mlp_residual",
                                [("projected", fc2_projected), ("addends", addends)],
                                256 * 1152 * 2)
        final[start:start + 256] = read(result_path, bfloat16, (256, 1152))
        print(f"completed output panel {start // 256 + 1}/16", flush=True)
    (out / "result.bin").write_bytes(final.tobytes())
    checks = [{"row": row, "max_relative_error": maxrel(
        final[row], downstream_reference(row, x, attended, weights))}
        for row in (0, 257, 4095)]
    report = {"model": "synthetic Gemma 3 4B SigLIP encoder layer",
              "tokens": 4096, "hidden": 1152, "heads": 16,
              "attention": attention_reference,
              "downstream_checks": checks,
              "passed": bool(np.isfinite(final.astype(np.float32)).all() and
                             attention_reference["max_relative_error"] < 0.01 and
                             all(item["max_relative_error"] < 0.02 for item in checks))}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("driver", "tower", "post", "attention", "residual", "out"):
        p.add_argument(f"--{key}", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.tower.resolve(), a.post.resolve(),
                         a.attention.resolve(), a.residual.resolve(), a.out.resolve()), indent=2))
