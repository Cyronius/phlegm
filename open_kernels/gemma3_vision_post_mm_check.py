"""Check SigLIP projection bias boundaries and tanh GELU on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.build_gemma3_vision_post_mm import IMAGES


def run(driver: Path, bundle: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    cases = {}
    for name, env in IMAGES.items():
        case = out / name
        case.mkdir(parents=True, exist_ok=True)
        logical = int(env["PHLEGM_LOGICAL"])
        padded = int(env["PHLEGM_PADDED"])
        gelu = bool(int(env["PHLEGM_GELU"]))
        projected = rng.normal(0, 0.7, (256, padded)).astype(np.float32)
        projected[0, :8] = np.array([-6, -3, -1, -0.1, 0, 0.1, 1, 6], np.float32)
        bias = rng.normal(0, 0.08, padded).astype(np.float32).astype(bfloat16)
        bias[logical:] = bfloat16(0)
        biased = (projected[:, :logical] + bias[:logical].astype(np.float32)).astype(bfloat16)
        expected = biased
        if gelu:
            x = biased.astype(np.float64)
            expected = (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) *
                         (x + 0.044715 * x ** 3)))).astype(bfloat16)
        (case / "projected.bin").write_bytes(projected.tobytes())
        (case / "bias.bin").write_bytes(bias.tobytes())
        build = bundle / name
        cfg = case / "run.cfg"
        cfg.write_text("\n".join([
            "device", f"xclbin G {build / 'final.xclbin'}",
            f"kernelx k G {build / 'insts.bin'}",
            f"buf projected {projected.nbytes} {case / 'projected.bin'}",
            f"buf bias {bias.nbytes} {case / 'bias.bin'}",
            f"buf result {256 * padded * 2}", "run k projected bias result",
            f"dump result {case / 'result.bin'} {256 * padded * 2}", "",
        ]))
        proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"{name} NPU failed: {proc.stdout}\n{proc.stderr}")
        actual = np.fromfile(case / "result.bin", bfloat16).reshape(256, padded)
        error = float(np.max(np.abs(actual[:, :logical].astype(np.float32) -
                                    expected.astype(np.float32))))
        scale = max(float(np.max(np.abs(expected.astype(np.float32)))), 1e-20)
        padding_zero = bool(np.all(actual[:, logical:] == 0))
        cases[name] = {"logical": logical, "padded": padded, "gelu": gelu,
                       "max_relative_error": error / scale,
                       "exact_bf16_fraction": float(np.mean(actual[:, :logical] == expected)),
                       "padding_zero": padding_zero,
                       "passed": bool(np.isfinite(actual.astype(np.float32)).all() and
                                      error / scale < 0.005 and padding_zero)}
    report = {"cases": cases, "passed": all(item["passed"] for item in cases.values())}
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
