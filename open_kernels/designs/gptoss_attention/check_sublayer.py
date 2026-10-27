"""Check the GPT-OSS NPU attention path through output projection and residual."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from designs.gptoss_attention.check_chain import invoke, run as attention_chain  # noqa: E402
from designs.gptoss_o_gemv.check import fixture as o_fixture, run as o_run  # noqa: E402


def run(driver: Path, qkv_build: Path, rope_build: Path, attn_build: Path,
        o_build: Path, residual_build: Path, out: Path, pos: int,
        activation: np.ndarray | None = None,
        oracle_activation: np.ndarray | None = None,
        residual_input: np.ndarray | None = None,
        full_state_build: Path | None = None) -> dict:
    chain = out / "chain"
    chain_report = attention_chain(driver, qkv_build, rope_build, attn_build, chain,
                                   pos, activation, oracle_activation,
                                   full_state_build)
    actual_attn = np.fromfile(chain / "attention" / "result.bin", np.float32).astype(bfloat16)
    golden_attn = np.fromfile(chain / "attention" / "expected.bin", np.float32).astype(bfloat16)
    if actual_attn.shape != (4096,) or golden_attn.shape != (4096,):
        raise AssertionError("attention projection input must have 4096 values")

    projected = []
    golden_projected = []
    projection_reports = []
    for band in range(0, 96, 16):
        directory = out / f"o-{band:02d}"
        projection_reports.append(o_run(driver, o_build, directory, band, actual_attn))
        projected.append(np.fromfile(directory / "result.bin", np.float32))
        golden_projected.append(o_fixture(band, golden_attn)[2])
    projected = np.concatenate(projected)[:2880]
    golden_projected = np.concatenate(golden_projected)[:2880]

    rng = np.random.default_rng(20261029 + pos)
    residual = (rng.normal(0, 0.7, 2880).astype(np.float32)
                if residual_input is None else np.asarray(residual_input, np.float32))
    if residual.shape != (2880,):
        raise ValueError("attention residual must be FP32[2880]")
    bias = rng.normal(0, 0.1, 2880).astype(np.float32).astype(bfloat16)
    pair = np.stack((residual.reshape(45, 64), projected.reshape(45, 64)), axis=1)
    actual = invoke(driver, residual_build, out / "residual",
                    [("pair", pair), ("bias", bias)], np.dtype(np.float32), 2880)
    expected = residual.astype(np.float64) + golden_projected.astype(np.float64) + \
               bias.astype(np.float64)
    (out / "residual" / "expected.bin").write_bytes(expected.astype(np.float32).tobytes())
    a = actual.astype(np.float64)
    max_abs = float(np.max(np.abs(a - expected)))
    max_rel = float(max_abs / max(np.max(np.abs(expected)), 1e-30))
    cosine = float(a @ expected / (np.linalg.norm(a) * np.linalg.norm(expected)))
    report = {"passed": bool(max_rel < 0.005 and cosine > 0.9999),
              "position": pos, "attention": chain_report,
              "o_projection_calls": len(projection_reports),
              "o_projection_max_relative_error": max(r["max_relative_error"] for r in projection_reports),
              "sublayer_max_absolute_error": max_abs,
              "sublayer_max_relative_error": max_rel,
              "sublayer_cosine": cosine,
              "reference": "q4nx QKV/O decode, YaRN, FP64 sink attention, output bias and residual"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--qkv-build", required=True, type=Path)
    p.add_argument("--rope-build", required=True, type=Path)
    p.add_argument("--attn-build", required=True, type=Path)
    p.add_argument("--o-build", required=True, type=Path)
    p.add_argument("--residual-build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--pos", type=int, default=127)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.qkv_build.resolve(), a.rope_build.resolve(),
                         a.attn_build.resolve(), a.o_build.resolve(),
                         a.residual_build.resolve(), a.out.resolve(), a.pos), indent=2))


if __name__ == "__main__":
    main()
