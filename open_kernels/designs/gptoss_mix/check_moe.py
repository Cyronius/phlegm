"""Check one full top-four GPT-OSS MoE FFN across compiled NPU kernels.

This is an arithmetic integration check through the harness, not a layer
recipe, weight manifest, or model runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "model"))
from gptoss_router.check import cases, validate  # noqa: E402
from gptoss_expert_gemv.check import fixture, reference  # noqa: E402
from replica_gptoss import clamped_swiglu  # noqa: E402
from check import invoke  # noqa: E402

HIDDEN = 2880
PADDED = 23 * 128


def f32(raw: bytes) -> np.ndarray:
    result = np.frombuffer(raw, dtype=np.float32).copy()
    if result.shape != (HIDDEN,) or not np.isfinite(result).all():
        raise AssertionError("expected 2880 finite FP32 outputs")
    return result


def score(actual: np.ndarray, expected: np.ndarray) -> dict:
    a = actual.astype(np.float64)
    e = expected.astype(np.float64)
    err = np.abs(a - e)
    return {"max_absolute_error": float(err.max()),
            "max_relative_error": float(err.max() / max(np.max(np.abs(e)), 1e-30)),
            "cosine": float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-30))}


def run(driver: Path, router_build: Path, projection_build: Path,
        gate_build: Path, mixer_build: Path, out: Path,
        residual_build: Path | None = None) -> dict:
    if not driver.is_file():
        raise FileNotFoundError(driver)
    builds = (router_build, projection_build, gate_build, mixer_build)
    for build in builds + ((residual_build,) if residual_build is not None else ()):
        for name in ("final.xclbin", "insts.bin"):
            if not (build / name).is_file():
                raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    _, x, router_matrix, bias = cases()[0]
    x_bias = np.concatenate([x, bias])
    route_bytes = invoke(driver, router_build, out / "router",
                         [x_bias, router_matrix], 256)
    ids = np.frombuffer(route_bytes, dtype=np.int32, count=4).copy()
    weights = np.frombuffer(route_bytes, dtype=np.float32, count=4, offset=16).copy()
    logits = np.frombuffer(route_bytes, dtype=np.float32, count=32, offset=32).copy()
    logits_ref = x.astype(np.float64) @ router_matrix.astype(np.float64)
    logits_ref += bias.astype(np.float64)
    validate(ids, weights, logits, logits_ref)
    ref_top = np.argsort(-logits_ref, kind="stable")[:4]
    e = np.exp(logits_ref[ref_top] - logits_ref[ref_top].max())
    ref_weights = e / e.sum()
    x_pad = np.zeros(PADDED, dtype=bfloat16)
    x_pad[:HIDDEN] = x
    down_outputs = []
    down_refs = []
    stage_errors = []
    for position, expert_id in enumerate(ids):
        d = out / f"expert-{int(expert_id)}"
        role_chunks = [fixture(90, 20261027 + int(expert_id) * 10 + role)[1]
                       for role in range(3)]
        gate = f32(invoke(driver, projection_build, d / "gate_projection",
                          [x_pad, role_chunks[0]], HIDDEN * 4))
        up = f32(invoke(driver, projection_build, d / "up_projection",
                        [x_pad, role_chunks[1]], HIDDEN * 4))
        hidden_raw = invoke(driver, gate_build, d / "activation", [gate, up], HIDDEN * 2)
        hidden = np.frombuffer(hidden_raw, dtype=bfloat16).copy()
        down_x = np.zeros(PADDED, dtype=bfloat16)
        down_x[:HIDDEN] = hidden
        down = f32(invoke(driver, projection_build, d / "down_projection",
                          [down_x, role_chunks[2]], HIDDEN * 4))
        gate_ref = reference(x_pad, role_chunks[0]).ravel()
        up_ref = reference(x_pad, role_chunks[1]).ravel()
        # Large random MXFP4 products legitimately saturate the sigmoid.
        with np.errstate(over="ignore"):
            hidden_ref = clamped_swiglu(gate_ref, up_ref).astype(bfloat16)
        down_ref_x = np.zeros(PADDED, dtype=bfloat16)
        down_ref_x[:HIDDEN] = hidden_ref
        down_ref = reference(down_ref_x, role_chunks[2]).ravel()
        stage_errors.append({"expert_id": int(expert_id), "position": position,
                             **score(down, down_ref)})
        down_outputs.append(down)
        down_refs.append(down_ref)
    selected = np.stack(down_outputs)
    packed = np.ascontiguousarray(selected.reshape(4, 45, 64).transpose(1, 0, 2))
    route = np.frombuffer(route_bytes, dtype=np.uint8).copy()
    actual = f32(invoke(driver, mixer_build, out / "mix", [packed, route], HIDDEN * 4))
    expected = ref_weights @ np.stack(down_refs).astype(np.float64)
    result_score = score(actual, expected)
    residual_score = None
    if residual_build is not None:
        rng = np.random.default_rng(20261028)
        residual = rng.normal(0, 2, HIDDEN).astype(np.float32)
        pair = np.ascontiguousarray(np.stack([residual.reshape(45, 64),
                                              actual.reshape(45, 64)], axis=1))
        zero_bias = np.zeros(HIDDEN, dtype=bfloat16)
        got = f32(invoke(driver, residual_build, out / "ffn_residual",
                         [pair, zero_bias], HIDDEN * 4))
        residual_score = score(got, residual.astype(np.float64) + expected)
    passed = (np.array_equal(ids, ref_top) and
              all(s["max_relative_error"] < 0.01 for s in stage_errors) and
              result_score["max_relative_error"] < 0.01 and
              result_score["cosine"] > 0.9999 and
              (residual_score is None or
               (residual_score["max_relative_error"] < 0.01 and
                residual_score["cosine"] > 0.9999)))
    report = {"passed": bool(passed), "expert_ids": ids.tolist(),
              "router_max_weight_error": float(np.max(np.abs(weights - ref_weights))),
              "expert_down_stages": stage_errors, "moe_output": result_score,
              "ffn_residual": residual_score,
              "reference": "FP64 router, decoded MXFP4 and clamped gate with BF16 activation boundary",
              "scope": "one token FFN arithmetic through separate images; no layer/runtime/model acceptance"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--router-build", required=True, type=Path)
    p.add_argument("--projection-build", required=True, type=Path)
    p.add_argument("--gate-build", required=True, type=Path)
    p.add_argument("--mixer-build", required=True, type=Path)
    p.add_argument("--residual-build", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.router_build.resolve(),
                         a.projection_build.resolve(), a.gate_build.resolve(),
                         a.mixer_build.resolve(), a.out.resolve(),
                         a.residual_build.resolve() if a.residual_build else None), indent=2))


if __name__ == "__main__":
    main()
