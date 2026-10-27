"""Compose one Gemma 4 12B text decoder layer from checked NPU images.

This is a decode layer, with one new token and optional prior BF16 K/V rows.
It implements the 12B local/full schedule, including full-layer K=V before
the separate K/V norms. Model payload and output acceptance are separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.designs.gemma4_attention.check_chain import invoke
from open_kernels.designs.gemma4_rope.check import table_for
from open_kernels.gemma4_text_plan import Projection, TextPlan, from_hf_config, projection_image_geometry, projections
from open_kernels.gemma4_text_source import Gemma4TextSource
from open_kernels.model.q4nx import Q4NX


def _image(bundle: Path, tag: str) -> Path:
    return bundle / tag


def _q4(driver: Path, bundle: Path, source: Gemma4TextSource, projection: Projection,
        activation: np.ndarray, out: Path) -> tuple[np.ndarray, int]:
    rows, image_cols, calls = projection_image_geometry(projection)
    if activation.shape != (projection.cols,) or activation.dtype != bfloat16:
        raise ValueError(f"{projection.name}: expected BF16[{projection.cols}] activation")
    image = _image(bundle / "projections", f"n{rows}_k{image_cols}")
    partials = []
    for stripe in range(calls):
        weights = np.concatenate([source.projection_band(projection, band, stripe, image_cols)
                                  for band in range(rows // 64)])
        x = np.ascontiguousarray(activation[stripe * image_cols:(stripe + 1) * image_cols])
        partials.append(invoke(driver, image, out / f"stripe-{stripe:02d}",
                               [weights, x], rows * 4, np.float32))
    return np.sum(partials, axis=0, dtype=np.float32).astype(bfloat16), calls


def _norm(driver: Path, bundle: Path, source: Gemma4TextSource, layer: int,
          suffix: str, value: np.ndarray, residual: np.ndarray, out: Path) -> np.ndarray:
    width = source.plan.hidden
    if value.shape != (width,) or residual.shape != (width,):
        raise ValueError("Gemma 4 hidden norm input width differs from config")
    scale = source.vector(f"model.layers.{layer}.{suffix}.weight", width)
    pair = np.concatenate((value.astype(bfloat16), residual.astype(bfloat16)))
    return invoke(driver, _image(bundle, "hidden_norm"), out,
                  [pair, scale], width * 2, bfloat16)


def _head_norm(driver: Path, bundle: Path, source: Gemma4TextSource, layer: int,
               role: str, x: np.ndarray, out: Path) -> np.ndarray:
    heads, dim = x.shape
    scale = (source.vector(f"model.layers.{layer}.self_attn.{role}_norm.weight", dim)
             if role != "v" else np.ones(dim, bfloat16))
    return invoke(driver, _image(bundle, f"head_norm_h{heads}_d{dim}"), out,
                  [x, scale], x.nbytes, bfloat16).reshape(heads, dim)


def _attention(driver: Path, bundle: Path, layer, q: np.ndarray, k: np.ndarray,
               v: np.ndarray, out: Path) -> tuple[np.ndarray, int]:
    rows, kvh, dim = k.shape
    if v.shape != k.shape or q.shape != (layer.query_heads, dim):
        raise ValueError("Gemma 4 attention cache geometry differs from layer plan")
    chunks = (1024, 512, 128, 1)
    result = np.empty((layer.query_heads, dim), np.float32)
    calls = 0
    for head in range(layer.query_heads):
        selected = head // (layer.query_heads // kvh)
        state = np.zeros(2 * dim + 16, np.float32)
        cursor = 0
        while cursor < rows:
            n = next(n for n in chunks if n <= rows - cursor)
            kv = np.stack((k[cursor:cursor + n, selected],
                           v[cursor:cursor + n, selected]), axis=1)
            meta = np.frombuffer(q[head].tobytes() + state.tobytes(), np.uint8)
            state = invoke(driver, _image(bundle, f"attention_d{dim}_r{n}"),
                           out / f"head-{head:02d}-row-{cursor:05d}",
                           [meta, kv], state.nbytes, np.float32)
            cursor += n
            calls += 1
        result[head] = state[dim + 16:]
    return result.reshape(-1).astype(bfloat16), calls


def run(driver: Path, bundle: Path, source: Gemma4TextSource, layer_index: int,
        position: int, hidden: np.ndarray, previous_k: np.ndarray,
        previous_v: np.ndarray, out: Path) -> dict:
    plan: TextPlan = source.plan
    if plan.variant != "12B" or not isinstance(layer_index, int) or not 0 <= layer_index < len(plan.layers):
        raise ValueError("expected a Gemma 4 12B layer")
    layer = plan.layers[layer_index]
    if (not isinstance(position, int) or position < 0 or position >= plan.max_context or
            hidden.shape != (plan.hidden,) or hidden.dtype != bfloat16 or
            previous_k.shape != (position, layer.kv_heads, layer.head_dim) or
            previous_v.shape != previous_k.shape or previous_k.dtype != bfloat16 or
            previous_v.dtype != bfloat16):
        raise ValueError("Gemma 4 decode state differs from layer geometry")
    lookup = {p.name: p for p in projections(plan) if p.layer == layer_index}
    out.mkdir(parents=True, exist_ok=False)
    calls = 0
    zero = np.zeros(plan.hidden, bfloat16)
    x = _norm(driver, bundle, source, layer_index, "input_layernorm",
              hidden, zero, out / "input_norm")
    calls += 1
    q, n = _q4(driver, bundle, source, lookup["q"], x, out / "q")
    calls += n
    q = _head_norm(driver, bundle, source, layer_index, "q",
                   q.reshape(layer.query_heads, layer.head_dim), out / "q_norm")
    calls += 1
    k_raw, n = _q4(driver, bundle, source, lookup["k"], x, out / "k")
    calls += n
    v_raw = k_raw.copy() if layer.alternative_k_eq_v else None
    if v_raw is None:
        v_raw, n = _q4(driver, bundle, source, lookup["v"], x, out / "v")
        calls += n
    k = _head_norm(driver, bundle, source, layer_index, "k",
                   k_raw.reshape(layer.kv_heads, layer.head_dim), out / "k_norm")
    v = _head_norm(driver, bundle, source, layer_index, "v",
                   v_raw.reshape(layer.kv_heads, layer.head_dim), out / "v_norm")
    calls += 2
    qk = np.concatenate((q, k))
    rotated = invoke(driver, _image(bundle, f"rope_h{layer.query_heads + layer.kv_heads}_d{layer.head_dim}"),
                     out / "rope", [qk, table_for(position, layer.head_dim)], qk.nbytes,
                     bfloat16).reshape(qk.shape)
    calls += 1
    q = rotated[:layer.query_heads]
    k = np.concatenate((previous_k, rotated[layer.query_heads:][None]), axis=0)
    v = np.concatenate((previous_v, v[None]), axis=0)
    first = max(0, k.shape[0] - layer.sliding_window) if layer.sliding_window else 0
    attended, n = _attention(driver, bundle, layer, q, k[first:], v[first:], out / "attention")
    calls += n
    o, n = _q4(driver, bundle, source, lookup["o"], attended, out / "o")
    calls += n
    residual = _norm(driver, bundle, source, layer_index, "post_attention_layernorm",
                     o, hidden, out / "post_attention_norm")
    calls += 1
    x = _norm(driver, bundle, source, layer_index, "pre_feedforward_layernorm",
              residual, zero, out / "pre_feedforward_norm")
    calls += 1
    gate, n = _q4(driver, bundle, source, lookup["gate"], x, out / "gate")
    calls += n
    up, n = _q4(driver, bundle, source, lookup["up"], x, out / "up")
    calls += n
    activated = invoke(driver, _image(bundle, f"mlp_gelu_{layer.mlp_width}"),
                       out / "gelu_mul", [gate, up], gate.nbytes, bfloat16)
    calls += 1
    down, n = _q4(driver, bundle, source, lookup["down"], activated, out / "down")
    calls += n
    result = _norm(driver, bundle, source, layer_index, "post_feedforward_layernorm",
                   down, residual, out / "post_feedforward_norm")
    calls += 1
    scalar = source.vector(f"model.layers.{layer_index}.layer_output_scale.weight", 1)
    result = (result.astype(np.float32) * scalar.astype(np.float32)).astype(bfloat16)
    (out / "output.bf16").write_bytes(result.tobytes())
    report = {"variant": plan.variant, "layer": layer_index, "position": position,
              "attention": layer.attention, "alternative_k_eq_v": layer.alternative_k_eq_v,
              "npu_calls": calls, "passed": True}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {**report, "output": result, "key": k, "value": v}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "bundle", "config", "model", "hidden", "previous-k",
                 "previous-v", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--position", type=int, required=True)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        plan = from_hf_config(json.loads(a.config.read_text()))
        source = Gemma4TextSource(reader, plan)
        source.validate()
        layer = plan.layers[a.layer]
        report = run(a.driver.resolve(), a.bundle.resolve(), source, a.layer, a.position,
                     np.fromfile(a.hidden, bfloat16),
                     np.fromfile(a.previous_k, bfloat16).reshape(-1, layer.kv_heads, layer.head_dim),
                     np.fromfile(a.previous_v, bfloat16).reshape(-1, layer.kv_heads, layer.head_dim),
                     a.out.resolve())
        print(json.dumps({k: v for k, v in report.items() if not isinstance(v, np.ndarray)}, indent=2))
    finally:
        reader.mm.close()
        reader.f.close()
