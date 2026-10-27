"""Run all four SigLIP transformer GEMM shapes on NPU with BF16 weights."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_tower_weights import SHAPES, pad_matrix
from open_kernels.gemma4_mm_source import npu_weight_row_major

BUILD_NAMES = {
    "qkv": "qkv_m256_k1152_n3584",
    "attention_out": "attention_out_m256_k1152_n1280",
    "mlp_fc1": "mlp_fc1_m256_k1152_n4352",
    "mlp_fc2": "mlp_fc2_m256_k4352_n1280",
}


def run(driver: Path, bundle: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    cases = {}
    for operation, (rows, cols, padded_rows, padded_cols) in SHAPES.items():
        case = out / operation
        case.mkdir(parents=True, exist_ok=True)
        x = rng.normal(0, 0.35, (256, padded_cols)).astype(np.float32).astype(bfloat16)
        weight = rng.normal(0, 0.02, (rows, cols)).astype(np.float32).astype(bfloat16)
        packed = npu_weight_row_major(pad_matrix(weight, operation), padded_rows, padded_cols)
        (case / "activation.bin").write_bytes(x.tobytes())
        (case / "weight.bin").write_bytes(packed.tobytes())
        build = bundle / BUILD_NAMES[operation]
        cfg = case / "run.cfg"
        cfg.write_text("\n".join([
            "device", f"xclbin G {build / 'final.xclbin'}",
            f"kernelx k G {build / 'insts.bin'}",
            f"buf activation {x.nbytes} {case / 'activation.bin'}",
            f"buf weight {packed.nbytes} {case / 'weight.bin'}",
            f"buf result {256 * padded_rows * 4}", "run k activation weight result",
            f"dump result {case / 'result.bin'} {256 * padded_rows * 4}", "",
        ]))
        result = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{operation} NPU failed: {result.stdout}\n{result.stderr}")
        actual = np.fromfile(case / "result.bin", np.float32).reshape(256, padded_rows)
        check_rows = np.array([0, 17, 255])
        check_cols = np.unique(np.r_[0:128, rows - 128:rows])
        expected = (x[check_rows, :cols].astype(np.float64) @
                    weight[check_cols].astype(np.float64).T).astype(np.float32)
        observed = actual[np.ix_(check_rows, check_cols)]
        error = float(np.max(np.abs(observed - expected)))
        relative = error / max(float(np.max(np.abs(expected))), 1e-20)
        padding_zero = bool(np.all(actual[:, rows:] == 0))
        cases[operation] = {"logical_shape": [rows, cols],
                            "padded_shape": [padded_rows, padded_cols],
                            "sampled_max_relative_error": relative,
                            "sampled_rows": len(check_rows),
                            "sampled_columns": len(check_cols),
                            "padding_zero": padding_zero,
                            "passed": bool(np.isfinite(actual).all() and
                                           relative < 0.005 and padding_zero)}
    report = {"cases": cases, "passed": all(case["passed"] for case in cases.values())}
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
