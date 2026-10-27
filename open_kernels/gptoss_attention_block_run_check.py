"""Check source GPT-OSS norm through attention residual against decoded q4."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from designs.gptoss_o_gemv.check import fixture as o_fixture  # noqa: E402
from designs.gptoss_qkv_gemv.check import fixture as qkv_fixture  # noqa: E402
from designs.gptoss_rope.check import CONFIG  # noqa: E402
from gptoss_attention_block_run import run  # noqa: E402
from gptoss_o_run import STARTS  # noqa: E402
from gptoss_qkv_run import CALLS  # noqa: E402
from model.replica_dense import rms, rope  # noqa: E402
from model.replica_gptoss import sink_attention  # noqa: E402
from recipes.gptoss_source import GptOssSource  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    return float(np.max(np.abs(a - e)) / max(np.max(np.abs(e)), 1e-20))


def check(driver: Path, kernels: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261030)
    layer, pos = 0, 128
    residual = rng.normal(0, 0.5, 2880).astype(np.float32)
    previous_k = rng.normal(0, 0.4, (127, 8, 64)).astype(np.float32).astype(bfloat16)
    previous_v = rng.normal(0, 0.5, (127, 8, 64)).astype(np.float32).astype(bfloat16)
    norm_weight = rng.normal(1, 0.2, 2880).astype(np.float32).astype(bfloat16)
    sinks = rng.normal(0, 1, 64).astype(np.float32).astype(bfloat16)
    o_bias = rng.normal(0, 0.05, 2880).astype(np.float32).astype(bfloat16)
    norm_ref = (rms(residual, 1e-5) * norm_weight.astype(np.float64)).astype(bfloat16)
    chunks = {}
    constants = {"input_norm": norm_weight, "sinks": sinks, "o_bias": o_bias}
    qkv_ref = {}
    for role, starts in CALLS.items():
        bias = np.empty(4096 if role == "q" else 512, bfloat16)
        pieces = []
        for start in starts:
            packed, selected, reference = qkv_fixture(role, start, 16, norm_ref)
            chunks[role, start] = selected
            bias[start * 32:(start + 16) * 32] = packed[2944:]
            pieces.append(reference)
        constants[role + "_bias"] = bias
        qkv_ref[role] = np.concatenate(pieces).astype(bfloat16)
    spec = ModelSpec.from_hf_config(CONFIG)
    inv = np.asarray(spec.rope_inv_freq(ctx=pos + 1), np.float64)
    scale = spec.rope_scale()
    q = rope(qkv_ref["q"].reshape(64, 64), pos, 64, spec.rope_theta,
             inv, scale).astype(bfloat16)
    k = rope(qkv_ref["k"].reshape(8, 64), pos, 64, spec.rope_theta,
             inv, scale).astype(bfloat16)
    v = qkv_ref["v"].reshape(8, 64)
    K = np.concatenate((previous_k, k[None]), axis=0)
    V = np.concatenate((previous_v, v[None]), axis=0)
    attention_ref = sink_attention(q, K, V, sinks, sliding_window=128,
                                   pos=127).ravel().astype(bfloat16)
    projections = []
    for start in STARTS:
        _, selected, reference = o_fixture(start, attention_ref)
        chunks["o", start] = selected
        projections.append(reference)
    projected = np.concatenate(projections)[:2880]
    output_ref = residual + (projected + o_bias.astype(np.float32))

    class FixtureSource(GptOssSource):
        def __init__(self):
            pass

        def constant(self, requested_layer, name):
            return constants[name]

        def attention(self, requested_layer, role, start):
            return chunks[role, start]

    run_report = run(driver, kernels, FixtureSource(), CONFIG, layer, pos,
                     residual, previous_k, previous_v, out)
    norm_actual = np.fromfile(out / "norm" / "result.bin", bfloat16)
    attention_actual = np.fromfile(out / "attention" / "attention.bf16", bfloat16)
    output_actual = np.fromfile(out / "output" / "output.f32", np.float32)
    if (norm_actual.size, attention_actual.size, output_actual.size) != (2880, 4096, 2880):
        raise AssertionError("incomplete GPT-OSS source attention block output")
    errors = {"norm": relative(norm_actual, norm_ref),
              "attention": relative(attention_actual, attention_ref),
              "output": relative(output_actual, output_ref)}
    report = {"layer": layer, "position": pos, "mode": run_report["mode"],
              "kernel_calls": run_report["kernel_calls"],
              "max_relative_error": errors,
              "passed": bool(errors["norm"] < 0.01 and
                             errors["attention"] < 0.005 and errors["output"] < 0.005)}
    (out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "kernels", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.driver.resolve(), a.kernels.resolve(),
                           a.out.resolve()), indent=2))
