"""Validate FLM Gemma 4 q4nx text tensors and repack 12B's 16-row tiles.

The published E2B/E4B package uses 32-row q4_1 tiles and BF16 PLE
matrices. The published 12B package uses 16-row signed q4 tiles and a
16-row q8 LM head. Repacking two adjacent 16-row tiles yields the pool's
32-row tile order without dequantizing or changing any q4 values.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_kernels.gemma4_text_plan import Projection, TextPlan, from_hf_config, projections
from open_kernels.model.q4nx import bf16_to_f32, f32_to_bf16


def tensor_name(p: Projection) -> str:
    if p.name == "per_layer_model_projection":
        return f"model.per_layer_model_proj.weight_layer{p.layer}"
    prefix = f"model.layers.{p.layer}."
    names = {
        "q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight",
        "v": "self_attn.v_proj.weight", "o": "self_attn.o_proj.weight",
        "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight",
        "down": "mlp.down_proj.weight", "per_layer_input_gate": "inp_gate.weight",
        "per_layer_projection": "per_layer_projection.weight",
    }
    return prefix + names[p.name]


def source_shape(plan: TextPlan, p: Projection) -> tuple[str, tuple[int, ...]]:
    if p.format == "bf16":
        return "BF16", (p.rows // 32, p.cols, 32)
    if plan.variant == "12B":
        return "I8", (p.rows // 16, p.cols // 256, 2304)
    return "I8", (p.rows * p.cols // (32 * 256), 5120)


def validate_header(plan: TextPlan, header: Mapping[str, dict]) -> dict:
    """Check required source tensors against the exact published decode shapes."""
    errors = []
    source_formats: dict[str, int] = {}

    def require(name: str, dtype: str, shape: tuple[int, ...]) -> None:
        item = header.get(name)
        if item is None:
            errors.append(f"missing {name}")
            return
        actual = (item.get("dtype"), tuple(item.get("shape", ())))
        if actual != (dtype, shape):
            errors.append(f"{name}: expected {dtype} {shape}, got {actual}")
            return
        offsets = item.get("data_offsets")
        item_bytes = {"I8": 1, "BF16": 2, "F32": 4}[dtype]
        expected_bytes = math.prod(shape) * item_bytes
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                offsets[1] - offsets[0] != expected_bytes):
            errors.append(f"{name}: byte length differs from {expected_bytes}")
            return
        source_formats[f"{dtype}:{shape[-1]}"] = source_formats.get(f"{dtype}:{shape[-1]}", 0) + 1

    for p in projections(plan):
        dtype, shape = source_shape(plan, p)
        require(tensor_name(p), dtype, shape)
    for layer in plan.layers:
        prefix = f"model.layers.{layer.index}."
        for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight",
                       "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"):
            require(prefix + suffix, "BF16", (plan.hidden,))
        require(prefix + "layer_output_scale.weight", "BF16", (1,))
        require(prefix + "self_attn.q_norm.weight", "BF16", (layer.head_dim,))
        require(prefix + "self_attn.k_norm.weight", "BF16", (layer.head_dim,))
        if layer.per_layer_input_width:
            require(prefix + "post_layernorm.weight", "BF16", (plan.hidden,))
    if plan.variant == "12B":
        require("lm_head.weight", "I8", (plan.vocab // 16, plan.hidden // 256, 4352))
        require("model.embed_tokens.weight", "BF16", (plan.vocab, plan.hidden))
        require("rope_freqs.weight", "F32", (256,))
    else:
        require("lm_head.weight", "I8", (plan.vocab * plan.hidden // (32 * 256), 5120))
        require("model.embed_tokens.weight", "I8", (plan.vocab, plan.hidden))
        require("model.embed_tokens.weight.scale", "F32", (plan.vocab, plan.hidden // 32))
        require("model.per_layer_token_embd.weight", "I8",
                (plan.vocab, len(plan.layers) * plan.layers[0].per_layer_input_width))
        require("model.per_layer_token_embd.weight.scale", "F32",
                (plan.vocab, len(plan.layers) * plan.layers[0].per_layer_input_width // 32))
    require("model.norm.weight", "BF16", (plan.hidden,))
    if plan.variant != "12B":
        require("model.per_layer_proj_norm.weight", "BF16",
                (plan.layers[0].per_layer_input_width,))
    return {"variant": plan.variant, "checked": sum(source_formats.values()),
            "source_formats": source_formats, "errors": errors, "passed": not errors}


def pair_q4_0_16row(source: np.ndarray) -> np.ndarray:
    """[row16,K/256,2304] signed q4 -> [row32,K/256,5120] q4_1.

    The low and high nibbles code adjacent rows. Two consecutive 16-row
    source tiles are the two half-row planes in one 32-row pool tile.
    """
    src = np.asarray(source, np.uint8)
    if src.ndim != 3 or src.shape[0] % 2 or src.shape[2] != 2304:
        raise ValueError("expected paired [row16,K/256,2304] q4 tiles")
    pairs, kt = src.shape[0] // 2, src.shape[1]
    halves = src.reshape(pairs, 2, kt, 2304).transpose(0, 2, 1, 3)
    scale = np.ascontiguousarray(halves[..., :256]).view(np.uint16).reshape(pairs, kt, 2, 8, 16)
    d = scale.transpose(0, 1, 3, 2, 4).reshape(pairs, kt, 256)
    m = f32_to_bf16(-8 * bf16_to_f32(d))
    codes = (halves[..., 256:] ^ 0x88).reshape(pairs, kt, 4096)
    out = np.empty((pairs, kt, 5120), np.uint8)
    out[..., :512] = d.view(np.uint8).reshape(pairs, kt, 512)
    out[..., 512:1024] = m.view(np.uint8).reshape(pairs, kt, 512)
    out[..., 1024:] = codes
    return out


def pair_q8_16row(source: np.ndarray) -> np.ndarray:
    """[row16,K/256,4352] q8 -> [row32,K/256,8704] q8, exactly."""
    src = np.asarray(source, np.uint8)
    if src.ndim != 3 or src.shape[0] % 2 or src.shape[2] != 4352:
        raise ValueError("expected paired [row16,K/256,4352] q8 tiles")
    pairs, kt = src.shape[0] // 2, src.shape[1]
    halves = src.reshape(pairs, 2, kt, 4352).transpose(0, 2, 1, 3)
    scale = np.ascontiguousarray(halves[..., :256]).view(np.uint16).reshape(pairs, kt, 2, 8, 16)
    d = scale.transpose(0, 1, 3, 2, 4).reshape(pairs, kt, 256)
    out = np.empty((pairs, kt, 8704), np.uint8)
    out[..., :512] = d.view(np.uint8).reshape(pairs, kt, 512)
    out[..., 512:] = halves[..., 256:].reshape(pairs, kt, 8192)
    return out


def q4_projection_pool(source: np.ndarray, rows: int, cols: int,
                       *, source_tile_bytes: int) -> np.ndarray:
    """FLM source -> 64-row band, K-tile, 32-row-half order of q4 GEMV.

    E2B/E4B supply flat 5120-byte q4_1 chunks in rowblock-major order. 12B
    supplies [row16,Ktile,2304] signed q4 tiles, paired losslessly first.
    """
    if rows % 64 or cols % 256 or source_tile_bytes not in (5120, 2304):
        raise ValueError("q4 projection requires 64 rows and 256-column tiles")
    kt = cols // 256
    raw = np.asarray(source, np.uint8)
    if source_tile_bytes == 5120:
        if raw.size != rows * kt // 32 * 5120:
            raise ValueError("wrong q4_1 source byte count")
        chunks = raw.reshape(rows // 32, kt, 5120)
    else:
        if raw.size != rows * kt // 16 * 2304:
            raise ValueError("wrong signed q4 source byte count")
        chunks = pair_q4_0_16row(raw.reshape(rows // 16, kt, 2304))
    return np.ascontiguousarray(chunks.reshape(rows // 64, 2, kt, 5120)
                                 .transpose(0, 2, 1, 3)).reshape(-1)


def q8_head_pool(source: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """12B FLM 16-row q8 head -> 128-row band, K-tile, 32-row-quarter order."""
    if rows % 128 or cols % 256:
        raise ValueError("q8 head requires 128 rows and 256-column tiles")
    kt = cols // 256
    raw = np.asarray(source, np.uint8)
    if raw.size != rows * kt // 16 * 4352:
        raise ValueError("wrong q8 source byte count")
    chunks = pair_q8_16row(raw.reshape(rows // 16, kt, 4352))
    return np.ascontiguousarray(chunks.reshape(rows // 128, 4, kt, 8704)
                                 .transpose(0, 2, 1, 3)).reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True,
                        help="JSON q4nx header, without its eight-byte length prefix")
    args = parser.parse_args()
    report = validate_header(from_hf_config(json.loads(args.config.read_text())),
                             json.loads(args.header.read_text()))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
