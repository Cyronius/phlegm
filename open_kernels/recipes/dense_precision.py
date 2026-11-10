"""Per-tensor dense carriage for compile-only GGUF fit probes.

DENSE_GGUF_LAYER selects one actual layer, never a publisher-specific pattern.
The current q8 GEMV uses padded 5120-B half-tiles (10 bpw), not compact
8704-B chunks. Runtime GGUF packing / variant dispatch is a separate gate.
"""
from __future__ import annotations

import os
import re
from dataclasses import replace
from .spec import ModelSpec, SpecError

PROJECTIONS = ("attn_q", "attn_k", "attn_v", "attn_output",
               "ffn_up", "ffn_gate", "ffn_down")
ROLES = {p: "attn" if p.startswith("attn_") else "ffn" for p in PROJECTIONS}
NAME = re.compile(r"blk\.(0|[1-9][0-9]*)\.(" + "|".join(PROJECTIONS) + r")\.weight$")
CARRIAGE = {"Q4_0": "q4_1", "Q4_1": "q4_1", "Q4_K": "q4_1",
            "Q8_0": "q8", "Q6_K": "q8", "Q5_K": "q8", "Q5_0": "q8",
            "Q5_1": "q8", "F16": "q8", "BF16": "q8"}


def validate_selector(selector, num_layers):
    if not isinstance(selector, dict):
        raise SpecError("tensor_quant must be a GGUF tensor name -> carriage map")
    for name, fmt in selector.items():
        m = NAME.fullmatch(name)
        if not m or int(m[1]) >= num_layers or fmt not in ("q4_1", "q8"):
            raise SpecError(f"tensor_quant: invalid tensor or carriage {name!r}: {fmt!r}")


def from_gguf_types(tensor_types, num_layers):
    """Keep every present projection, including q4, so absence is never guessed."""
    result = {}
    for name, ty in tensor_types.items():
        if not NAME.fullmatch(name):
            continue
        if ty not in CARRIAGE:
            raise SpecError(f"{name}: unsupported dense GGUF type {ty!r}")
        result[name] = CARRIAGE[ty]
    validate_selector(result, num_layers)
    return result


def layer_formats(spec, layer=None):
    if not spec.tensor_quant:
        return {p: spec.quant_of(ROLES[p]) for p in PROJECTIONS}
    validate_selector(spec.tensor_quant, spec.num_layers)
    if spec.quant != "q4_1":
        raise SpecError("tensor_quant needs the q4_1 base law (q8 uses padded half-tiles)")
    if layer is None:
        raw = os.environ.get("DENSE_GGUF_LAYER", "")
        if not raw.isdecimal():
            raise SpecError("tensor_quant requires DENSE_GGUF_LAYER for this compile-only probe")
        layer = int(raw)
    if not 0 <= layer < spec.num_layers:
        raise SpecError(f"DENSE_GGUF_LAYER {layer} outside [0, {spec.num_layers})")
    names = {p: f"blk.{layer}.{p}.weight" for p in PROJECTIONS}
    missing = [n for n in names.values() if n not in spec.tensor_quant]
    if missing:
        raise SpecError(f"tensor_quant: missing projections for layer {layer}: {missing}")
    return {p: spec.tensor_quant[n] for p, n in names.items()}


def q8_projections(spec):
    return frozenset(p for p, fmt in layer_formats(spec).items() if fmt == "q8")


def projection_spec(spec, projection):
    """Reuse the existing catalogue and pool laws for one projection's format."""
    if not spec.tensor_quant:
        return spec
    return replace(spec, quant={ROLES[projection]: layer_formats(spec)[projection]})
