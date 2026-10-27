"""Check a full source GPT-OSS layer against decoded q4/MXFP4 CPU arithmetic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from designs.gptoss_expert_gemv.check import fixture as expert_fixture  # noqa: E402
from designs.gptoss_expert_gemv.check import reference as expert_reference  # noqa: E402
from designs.gptoss_o_gemv.check import fixture as o_fixture  # noqa: E402
from designs.gptoss_qkv_gemv.check import fixture as qkv_fixture  # noqa: E402
from designs.gptoss_rope.check import CONFIG  # noqa: E402
from gptoss_layer_run import run  # noqa: E402
from gptoss_o_run import STARTS  # noqa: E402
from gptoss_qkv_run import CALLS  # noqa: E402
from model.replica_dense import rms, rope  # noqa: E402
from model.replica_gptoss import clamped_swiglu, sink_attention  # noqa: E402
from recipes.gptoss_source import GptOssSource  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    return float(np.max(np.abs(a - e)) / max(np.max(np.abs(e)), 1e-20))


def check(driver: Path, kernels: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261031)
    layer, pos = 0, 128
    residual = rng.normal(0, 0.5, 2880).astype(np.float32)
    previous_k = rng.normal(0, 0.4, (127, 8, 64)).astype(np.float32).astype(bfloat16)
    previous_v = rng.normal(0, 0.5, (127, 8, 64)).astype(np.float32).astype(bfloat16)
    norm_weight = rng.normal(1, 0.2, 2880).astype(np.float32).astype(bfloat16)
    post_weight = rng.normal(1, 0.2, 2880).astype(np.float32).astype(bfloat16)
    sinks = rng.normal(0, 1, 64).astype(np.float32).astype(bfloat16)
    o_bias = rng.normal(0, 0.05, 2880).astype(np.float32).astype(bfloat16)
    router_bias = np.zeros(32, bfloat16)
    experts = [17, 9, 3, 26]
    router_bias[experts] = np.array([12, 10, 8, 6], bfloat16)
    router_weight = np.zeros((2880, 32), bfloat16)
    constants = {"input_norm": norm_weight, "post_attention_norm": post_weight,
                 "sinks": sinks, "o_bias": o_bias, "router_bias": router_bias}
    chunks = {}
    norm_ref = (rms(residual, 1e-5) * norm_weight.astype(np.float64)).astype(bfloat16)
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
    attn_ref = sink_attention(q, K, V, sinks, sliding_window=128,
                              pos=127).ravel().astype(bfloat16)
    projections = []
    for start in STARTS:
        _, selected, reference = o_fixture(start, attn_ref)
        chunks["o", start] = selected
        projections.append(reference)
    attention_residual_ref = residual + (np.concatenate(projections)[:2880] +
                                          o_bias.astype(np.float32))
    post_ref = (rms(attention_residual_ref, 1e-5) *
                post_weight.astype(np.float64)).astype(bfloat16)
    expert_chunks, expert_biases = {}, {}
    down_refs = []
    x_pad = np.zeros(2944, bfloat16)
    x_pad[:2880] = post_ref
    for expert_id in experts:
        packed = {}
        for role_index, role in enumerate(("gate", "up", "down")):
            _, raw = expert_fixture(90, 20261027 + expert_id * 10 + role_index)
            expert_chunks[expert_id, role] = raw.copy()
            expert_biases[expert_id, role] = np.frombuffer(
                raw[:, 0, 128:192].tobytes(), bfloat16).copy().reshape(2880)
            packed[role] = raw
        gate = expert_reference(x_pad, packed["gate"]).ravel()
        up = expert_reference(x_pad, packed["up"]).ravel()
        with np.errstate(over="ignore"):
            hidden = clamped_swiglu(gate, up).astype(bfloat16)
        down_x = np.zeros(2944, bfloat16)
        down_x[:2880] = hidden
        down_refs.append(expert_reference(down_x, packed["down"]).ravel())
    logits = router_bias.astype(np.float64)[experts]
    route_weights = np.exp(logits - logits.max())
    route_weights /= route_weights.sum()
    output_ref = attention_residual_ref.astype(np.float64) + \
        route_weights @ np.stack(down_refs).astype(np.float64)

    class FixtureSource(GptOssSource):
        def __init__(self):
            pass

        def constant(self, requested_layer, name):
            return constants[name]

        def attention(self, requested_layer, role, start):
            return chunks[role, start]

        def router_weight(self, requested_layer):
            return router_weight

        def expert(self, requested_layer, expert_id, role):
            raw = expert_chunks[expert_id, role].copy()
            raw[:, 0, 128:192] = 0
            return raw

        def expert_bias(self, requested_layer, expert_id, role):
            return expert_biases[expert_id, role]

    run_report = run(driver, kernels, FixtureSource(), CONFIG, layer, pos,
                     residual, previous_k, previous_v, out)
    attention_actual = np.fromfile(out / "attention_block" / "output" / "output.f32", np.float32)
    post_actual = np.fromfile(out / "post_attention_norm" / "result.bin", bfloat16)
    output_actual = np.fromfile(out / "output.f32", np.float32)
    if (attention_actual.size, post_actual.size, output_actual.size) != (2880, 2880, 2880):
        raise AssertionError("incomplete GPT-OSS source layer output")
    errors = {"attention_residual": relative(attention_actual, attention_residual_ref),
              "post_attention_norm": relative(post_actual, post_ref),
              "layer_output": relative(output_actual, output_ref)}
    route = np.fromfile(out / "moe" / "router" / "result.bin", np.uint8)
    selected = np.frombuffer(route, np.int32, count=4)
    weights = np.frombuffer(route, np.float32, count=4, offset=16)
    conditioned_down = []
    for expert_id in selected:
        stage = out / "moe" / f"expert-{int(expert_id):02d}" / "down"
        x = np.fromfile(stage / "x.bin", bfloat16)
        packed = np.fromfile(stage / "chunks.bin", np.uint8).reshape(90, 23, 2560)
        conditioned_down.append(expert_reference(x, packed).ravel())
    conditioned = attention_actual.astype(np.float64) + \
        weights.astype(np.float64) @ np.stack(conditioned_down).astype(np.float64)
    conditioned_error = relative(output_actual, conditioned)
    report = {"kernel_calls": run_report["kernel_calls"],
              "expert_ids": run_report["expert_ids"], "max_relative_error": errors,
              "moe_conditioned_max_relative_error": conditioned_error,
              "passed": bool(run_report["expert_ids"] == experts and
                             errors["attention_residual"] < 0.005 and
                             errors["post_attention_norm"] < 0.01 and
                             errors["layer_output"] < 0.02 and
                             conditioned_error < 0.001)}
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
