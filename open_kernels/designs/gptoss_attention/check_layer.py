"""Check one GPT-OSS attention/MoE layer across NPU images and FP64 oracle."""
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
from designs.gptoss_attention.check_prefnorm_sublayer import run as pre_attn  # noqa: E402
from designs.gptoss_mix.check_moe import run as moe_run  # noqa: E402
from replica_dense import rms  # noqa: E402


def run(driver: Path, norm_build: Path, qkv_build: Path, rope_build: Path,
        attn_build: Path, o_build: Path, residual_build: Path,
        router_build: Path, expert_build: Path, gate_build: Path,
        mixer_build: Path, out: Path, pos: int,
        full_state_build: Path | None = None,
        finalize_build: Path | None = None) -> dict:
    attn = pre_attn(driver, norm_build, qkv_build, rope_build, attn_build,
                    o_build, residual_build, out / "attention", pos,
                    full_state_build, finalize_build)
    sub = out / "attention" / "sublayer" / "residual"
    actual_res = np.fromfile(sub / "result.bin", np.float32)
    golden_res = np.fromfile(sub / "expected.bin", np.float32)
    if actual_res.shape != (2880,) or golden_res.shape != (2880,):
        raise AssertionError("attention residual must have 2880 values")
    rng = np.random.default_rng(20261031 + pos)
    post_weight = rng.normal(1, 0.2, 2880).astype(np.float32).astype(bfloat16)
    actual_norm = invoke(driver, norm_build, out / "post-attention-norm",
                         [("residual", actual_res), ("weight", post_weight)],
                         np.dtype(bfloat16), 2880)
    golden_norm = (rms(golden_res, 1e-5) * post_weight.astype(np.float64)).astype(bfloat16)
    norm_err = float(np.max(np.abs(actual_norm.astype(np.float32) -
                                   golden_norm.astype(np.float32))))
    moe = moe_run(driver, router_build, expert_build, gate_build, mixer_build,
                  out / "moe", residual_build,
                  activation=actual_norm, oracle_activation=golden_norm,
                  residual_input=actual_res, oracle_residual_input=golden_res)
    final = moe["ffn_residual"]
    report = {"passed": bool(attn["passed"] and moe["passed"] and norm_err < 0.02),
              "position": pos, "mode": "full-state" if full_state_build else "sliding-128",
              "npu_attention_finalizer": finalize_build is not None,
              "attention_relative_error": attn["sublayer_max_relative_error"],
              "post_attention_norm_max_absolute_error": norm_err,
              "post_attention_norm_exact_bf16_fraction": float(np.mean(actual_norm == golden_norm)),
              "expert_ids": moe["expert_ids"],
              "layer_max_absolute_error": final["max_absolute_error"],
              "layer_max_relative_error": final["max_relative_error"],
              "layer_cosine": final["cosine"],
              "reference": "synthetic GPT-OSS 20B one-token FP64 layer arithmetic"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "norm-build", "qkv-build", "rope-build", "attn-build",
                 "o-build", "residual-build", "router-build", "expert-build",
                 "gate-build", "mixer-build", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--pos", type=int, default=127)
    p.add_argument("--full-state-build", type=Path)
    p.add_argument("--finalize-build", type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.norm_build.resolve(),
                         a.qkv_build.resolve(), a.rope_build.resolve(),
                         a.attn_build.resolve(), a.o_build.resolve(),
                         a.residual_build.resolve(), a.router_build.resolve(),
                         a.expert_build.resolve(), a.gate_build.resolve(),
                         a.mixer_build.resolve(), a.out.resolve(), a.pos,
                         a.full_state_build.resolve() if a.full_state_build else None,
                         a.finalize_build.resolve() if a.finalize_build else None), indent=2))


if __name__ == "__main__":
    main()
