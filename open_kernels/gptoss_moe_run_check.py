"""Check source GPT-OSS top-four MoE against decoded MXFP4 and FP64 routing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from designs.gptoss_expert_gemv.check import fixture, reference  # noqa: E402
from designs.gptoss_router.check import cases, validate  # noqa: E402
from gptoss_moe_run import run  # noqa: E402
from model.replica_gptoss import clamped_swiglu  # noqa: E402
from recipes.gptoss_source import GptOssSource  # noqa: E402


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    return float(np.max(np.abs(a - e)) / max(np.max(np.abs(e)), 1e-20))


def check(driver: Path, kernels: Path, out: Path) -> dict:
    _, activation, router_weight, router_bias = cases()[1]
    rng = np.random.default_rng(20261030)
    residual = rng.normal(0, 0.5, 2880).astype(np.float32)
    logits = activation.astype(np.float64) @ router_weight.astype(np.float64)
    logits += router_bias.astype(np.float64)
    ids = np.argsort(-logits, kind="stable")[:4]
    expv = np.exp(logits[ids] - logits[ids].max())
    route_weights = expv / expv.sum()
    chunks, biases = {}, {}
    down_refs = []
    x_pad = np.zeros(2944, bfloat16)
    x_pad[:2880] = activation
    for expert_id in ids:
        packed = {}
        for role_index, role in enumerate(("gate", "up", "down")):
            _, raw = fixture(90, 20261027 + int(expert_id) * 10 + role_index)
            chunks[int(expert_id), role] = raw.copy()
            biases[int(expert_id), role] = np.frombuffer(
                raw[:, 0, 128:192].tobytes(), bfloat16).copy().reshape(2880)
            packed[role] = raw
        gate = reference(x_pad, packed["gate"]).ravel()
        up = reference(x_pad, packed["up"]).ravel()
        with np.errstate(over="ignore"):
            hidden = clamped_swiglu(gate, up).astype(bfloat16)
        down_x = np.zeros(2944, bfloat16)
        down_x[:2880] = hidden
        down_refs.append(reference(down_x, packed["down"]).ravel())
    output_ref = residual.astype(np.float64) + route_weights @ np.stack(down_refs).astype(np.float64)

    class FixtureSource(GptOssSource):
        def __init__(self):
            pass

        def constant(self, layer, name):
            if name != "router_bias":
                raise ValueError(name)
            return router_bias

        def router_weight(self, layer):
            return router_weight

        def expert(self, layer, expert_id, role):
            raw = chunks[expert_id, role].copy()
            raw[:, 0, 128:192] = 0
            return raw

        def expert_bias(self, layer, expert_id, role):
            return biases[expert_id, role]

    run_report = run(driver, kernels, FixtureSource(), 0, activation, residual, out)
    route = np.fromfile(out / "router" / "result.bin", np.uint8)
    got_ids = np.frombuffer(route, np.int32, count=4).copy()
    got_weights = np.frombuffer(route, np.float32, count=4, offset=16).copy()
    got_logits = np.frombuffer(route, np.float32, count=32, offset=32).copy()
    validate(got_ids, got_weights, got_logits, logits)
    output_actual = np.fromfile(out / "output.f32", np.float32)
    if output_actual.size != 2880:
        raise AssertionError("incomplete GPT-OSS source MoE output")
    down_errors = {}
    for expert_id, expected in zip(ids, down_refs):
        actual = np.fromfile(out / f"expert-{int(expert_id):02d}" / "down" /
                             "result.bin", np.float32)
        down_errors[str(int(expert_id))] = relative(actual, expected)
    error = relative(output_actual, output_ref)
    report = {"kernel_calls": run_report["kernel_calls"],
              "expert_ids": got_ids.tolist(),
              "down_max_relative_error": down_errors,
              "output_max_relative_error": error,
              "passed": bool(np.array_equal(got_ids, ids) and
                             max(down_errors.values()) < 0.01 and error < 0.01)}
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
