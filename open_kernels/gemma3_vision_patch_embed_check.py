"""Check Gemma 3 patch GEMM -> bias -> learned position on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_patch_pack import K, HIDDEN, pack_pixels, pad_weight, panel
from open_kernels.gemma4_mm_source import npu_weight_row_major


def invoke(driver: Path, cfg: Path) -> None:
    result = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"NPU harness failed: {result.stdout}\n{result.stderr}")


def finish_reference(projected: np.ndarray, bias: np.ndarray,
                     positions: np.ndarray) -> np.ndarray:
    biased = (projected[:, :HIDDEN] + bias.astype(np.float32)).astype(bfloat16)
    return (biased.astype(np.float32) + positions.astype(np.float32)).astype(bfloat16)


def run(driver: Path, mm_build: Path, finish_build: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261027)
    image = rng.normal(0, 0.3, (3, 896, 896)).astype(np.float32).astype(bfloat16)
    weight = rng.normal(0, 0.03, (HIDDEN, 3, 14, 14)).astype(np.float32).astype(bfloat16)
    bias = rng.normal(0, 0.04, HIDDEN).astype(np.float32).astype(bfloat16)
    positions = rng.normal(0, 0.08, (256, HIDDEN)).astype(np.float32).astype(bfloat16)
    patches = panel(pack_pixels(image), 256)
    packed_weight = npu_weight_row_major(pad_weight(weight), 1280, 640)
    addends = np.empty((256, 2, HIDDEN), bfloat16)
    addends[:, 0] = bias
    addends[:, 1] = positions
    for name, data in (("patches", patches), ("weights", packed_weight),
                       ("addends", addends)):
        (out / f"{name}.bin").write_bytes(data.tobytes())

    mm_cfg = out / "matrix.cfg"
    mm_cfg.write_text("\n".join([
        "device", f"xclbin G {mm_build / 'final.xclbin'}",
        f"kernelx k G {mm_build / 'insts.bin'}",
        f"buf patches {patches.nbytes} {out / 'patches.bin'}",
        f"buf weights {packed_weight.nbytes} {out / 'weights.bin'}",
        f"buf result {256 * 1280 * 4}", "run k patches weights result",
        f"dump result {out / 'projected.bin'} {256 * 1280 * 4}", "",
    ]))
    invoke(driver, mm_cfg)
    projected = np.fromfile(out / "projected.bin", np.float32).reshape(256, 1280)

    finish_cfg = out / "finish.cfg"
    finish_cfg.write_text("\n".join([
        "device", f"xclbin G {finish_build / 'final.xclbin'}",
        f"kernelx k G {finish_build / 'insts.bin'}",
        f"buf projected {projected.nbytes} {out / 'projected.bin'}",
        f"buf addends {addends.nbytes} {out / 'addends.bin'}",
        f"buf result {256 * HIDDEN * 2}", "run k projected addends result",
        f"dump result {out / 'result.bin'} {256 * HIDDEN * 2}", "",
    ]))
    invoke(driver, finish_cfg)
    actual = np.fromfile(out / "result.bin", bfloat16).reshape(256, HIDDEN)
    expected_finish = finish_reference(projected, bias, positions)
    finish_exact = float(np.mean(actual == expected_finish))

    # This reference starts at the source RGB pixels and Conv2d weights, with
    # no NPU intermediate substituted into the arithmetic.
    expected_mm = (patches[:, :K].astype(np.float64) @
                   weight.reshape(HIDDEN, K).astype(np.float64).T).astype(np.float32)
    expected = finish_reference(expected_mm, bias, positions)
    error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
    scale = max(float(np.max(np.abs(expected.astype(np.float32)))), 1e-20)
    matrix_error = float(np.max(np.abs(projected[:, :HIDDEN] - expected_mm)))
    matrix_scale = max(float(np.max(np.abs(expected_mm))), 1e-20)
    report = {"panel_start": 256, "rows": 256, "patch_size": 14,
              "patch_input": K, "vision_hidden": HIDDEN,
              "matrix_max_relative_error": matrix_error / matrix_scale,
              "finish_exact_bf16_fraction": finish_exact,
              "end_to_end_max_relative_error": error / scale,
              "passed": bool(np.isfinite(actual.astype(np.float32)).all() and
                             finish_exact == 1 and matrix_error / matrix_scale < 0.005 and
                             error / scale < 0.005)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--mm-build", required=True, type=Path)
    p.add_argument("--finish-build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.mm_build.resolve(),
                         a.finish_build.resolve(), a.out.resolve()), indent=2))
