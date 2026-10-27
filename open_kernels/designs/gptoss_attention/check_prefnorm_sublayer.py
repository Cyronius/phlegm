"""Check GPT-OSS input RMSNorm through the full attention residual path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from designs.gptoss_attention.check_chain import invoke  # noqa: E402
from designs.gptoss_attention.check_sublayer import run as attention_sublayer  # noqa: E402
from replica_dense import rms  # noqa: E402


def run(driver: Path, norm_build: Path, qkv_build: Path, rope_build: Path,
        attn_build: Path, o_build: Path, residual_build: Path,
        out: Path, pos: int) -> dict:
    rng = np.random.default_rng(20261030 + pos)
    xres = rng.normal(0, 0.7, 2880).astype(np.float32)
    weight = rng.normal(1, 0.2, 2880).astype(np.float32).astype(bfloat16)
    actual_norm = invoke(driver, norm_build, out / "input-norm",
                         [("residual", xres), ("weight", weight)],
                         np.dtype(bfloat16), 2880)
    golden_norm = (rms(xres, 1e-5) * weight.astype(np.float64)).astype(bfloat16)
    norm_err = float(np.max(np.abs(actual_norm.astype(np.float32) -
                                   golden_norm.astype(np.float32))))
    sublayer = attention_sublayer(driver, qkv_build, rope_build, attn_build,
                                   o_build, residual_build, out / "sublayer", pos,
                                   activation=actual_norm,
                                   oracle_activation=golden_norm,
                                   residual_input=xres)
    report = {"passed": bool(sublayer["passed"] and norm_err < 0.02),
              "position": pos, "input_norm_max_absolute_error": norm_err,
              "input_norm_exact_bf16_fraction": float(np.mean(actual_norm == golden_norm)),
              "sublayer_max_relative_error": sublayer["sublayer_max_relative_error"],
              "sublayer_cosine": sublayer["sublayer_cosine"],
              "reference": "FP64 input RMSNorm, q4nx projections, YaRN, sink attention, residual"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "norm-build", "qkv-build", "rope-build", "attn-build",
                 "o-build", "residual-build", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--pos", type=int, default=127)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.norm_build.resolve(),
                         a.qkv_build.resolve(), a.rope_build.resolve(),
                         a.attn_build.resolve(), a.o_build.resolve(),
                         a.residual_build.resolve(), a.out.resolve(), a.pos), indent=2))


if __name__ == "__main__":
    main()
