"""Validated text-layer geometry for the three published Gemma 4 FLM entries.

This is a scheduler plan, not a model recipe. It does not claim that the
weights are packed or that text generation is connected to the NPU images.
"""

from dataclasses import dataclass
from typing import Mapping


class Gemma4PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Layer:
    index: int
    attention: str
    head_dim: int
    query_heads: int
    kv_heads: int
    rotary_dim: int
    rope_theta: float
    rope_type: str
    sliding_window: int | None
    kv_source_layer: int
    stores_shared_kv: bool
    alternative_k_eq_v: bool
    mlp_width: int
    per_layer_input_width: int

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def computes_kv(self) -> bool:
        return self.kv_source_layer == self.index

    @property
    def has_v_projection(self) -> bool:
        return self.computes_kv and not self.alternative_k_eq_v


@dataclass(frozen=True)
class TextPlan:
    variant: str
    hidden: int
    vocab: int
    max_context: int
    softcap: float
    norm_eps: float
    layers: tuple[Layer, ...]


@dataclass(frozen=True)
class Projection:
    layer: int | None
    name: str
    rows: int
    cols: int


def projections(plan: TextPlan) -> tuple[Projection, ...]:
    """All text GEMVs required by the published layer schedule.

    Shared layers consume provider K/V states rather than recomputing them.
    Matrix naming here is a logical role; source-container names are separate.
    """
    out = []
    if plan.layers[0].per_layer_input_width:
        out.append(Projection(None, "per_layer_model_projection",
                              len(plan.layers) * plan.layers[0].per_layer_input_width, plan.hidden))
    for layer in plan.layers:
        i, hidden, ff = layer.index, plan.hidden, layer.mlp_width
        out.append(Projection(i, "q", layer.query_width, hidden))
        if layer.computes_kv:
            out.append(Projection(i, "k", layer.kv_width, hidden))
            if layer.has_v_projection:
                out.append(Projection(i, "v", layer.kv_width, hidden))
        out.append(Projection(i, "o", hidden, layer.query_width))
        out.extend((Projection(i, "gate", ff, hidden),
                    Projection(i, "up", ff, hidden),
                    Projection(i, "down", hidden, ff)))
        if layer.per_layer_input_width:
            out.extend((Projection(i, "per_layer_input_gate", layer.per_layer_input_width, hidden),
                        Projection(i, "per_layer_projection", hidden, layer.per_layer_input_width)))
    return tuple(out)


def projection_geometries(plan: TextPlan) -> tuple[tuple[int, int], ...]:
    """Distinct (rows, cols) image geometries, sorted for deterministic builds."""
    shapes = {(p.rows, p.cols) for p in projections(plan)}
    if any(n % 64 or k % 256 for n, k in shapes):
        raise Gemma4PlanError("a projection is outside the q4 GEMV tile geometry")
    return tuple(sorted(shapes))


def projection_image_geometry(projection: Projection) -> tuple[int, int, int]:
    """(rows, cols per image, number of calls) for the 64 KiB AIE tile budget.

    The existing q4 GEMV holds the BF16 activation, preparation table, two
    10 KiB weight FIFOs, output FIFOs and stack in one tile. The largest
    verified safe input width is 8192; wider reductions run equal K stripes.
    """
    n, k = projection.rows, projection.cols
    if n % 64 or k % 256:
        raise Gemma4PlanError("a projection is outside the q4 GEMV tile geometry")
    calls = (k + 8191) // 8192
    if k % calls or (k // calls) % 256:
        raise Gemma4PlanError(f"cannot split {k}-wide q4 reduction evenly")
    return n, k // calls, calls


# Pin the published text geometries. A different configuration must be
# investigated before we build images or interpret its tensors as one of these.
_PROFILES = {
    ("gemma4_text", 1536): ("E2B", 35, 6144, 8, 1, None, 512, 20, 256, True, False, 131072, 5),
    ("gemma4_text", 2560): ("E4B", 42, 10240, 8, 2, None, 512, 18, 256, False, False, 131072, 6),
    ("gemma4_unified_text", 3840): ("12B", 48, 15360, 16, 8, 1, 1024, 0, 0, False, True, 262144, 6),
}


def _expect(config: Mapping, key: str, expected) -> None:
    if config.get(key) != expected:
        raise Gemma4PlanError(f"{key}: expected {expected!r}, got {config.get(key)!r}")


def from_hf_config(config: Mapping) -> TextPlan:
    """Read either a top-level Gemma 4 config or its nested text_config."""
    if "text_config" in config:
        if config.get("model_type") not in ("gemma4", "gemma4_unified"):
            raise Gemma4PlanError("unsupported top-level model_type")
        config = config["text_config"]
    if not isinstance(config, Mapping):
        raise Gemma4PlanError("text_config must be a mapping")
    key = (config.get("model_type"), config.get("hidden_size"))
    if key not in _PROFILES:
        raise Gemma4PlanError(f"unsupported Gemma 4 text geometry: {key!r}")
    (variant, depth, ff, heads, local_kvh, full_kvh, window, shared,
     ple, double_mlp, k_eq_v, context, period) = _PROFILES[key]
    pinned = {
        "num_hidden_layers": depth, "intermediate_size": ff,
        "num_attention_heads": heads, "num_key_value_heads": local_kvh,
        "num_global_key_value_heads": full_kvh, "head_dim": 256,
        "global_head_dim": 512, "sliding_window": window,
        "num_kv_shared_layers": shared, "hidden_size_per_layer_input": ple,
        "use_double_wide_mlp": double_mlp, "attention_k_eq_v": k_eq_v,
        "max_position_embeddings": context, "vocab_size": 262144,
        "hidden_activation": "gelu_pytorch_tanh", "enable_moe_block": False,
        "attention_bias": False, "final_logit_softcapping": 30.0,
        "rms_norm_eps": 1e-6,
    }
    for name, value in pinned.items():
        _expect(config, name, value)
    rope = config.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise Gemma4PlanError("missing rope_parameters")
    local_rope = rope.get("sliding_attention", {})
    full_rope = rope.get("full_attention", {})
    if local_rope != {"rope_theta": 10000.0, "rope_type": "default"}:
        raise Gemma4PlanError("unsupported sliding_attention RoPE")
    if full_rope != {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}:
        raise Gemma4PlanError("unsupported full_attention RoPE")

    types = config.get("layer_types")
    if not isinstance(types, list) or len(types) != depth:
        raise Gemma4PlanError("layer_types must contain one entry per layer")
    first_shared = depth - shared
    expected_types = ["full_attention" if (i + 1) % period == 0 else "sliding_attention" for i in range(depth)]
    if types != expected_types:
        raise Gemma4PlanError("unsupported Gemma 4 local/full layer schedule")
    providers = {kind: max(i for i in range(first_shared) if types[i] == kind)
                 for kind in ("sliding_attention", "full_attention") if shared}
    layers = []
    for i, kind in enumerate(types):
        full = kind == "full_attention"
        source = providers[kind] if i >= first_shared and shared else i
        layers.append(Layer(
            index=i, attention=kind, head_dim=512 if full else 256,
            query_heads=heads, kv_heads=(full_kvh or local_kvh) if full and k_eq_v else local_kvh,
            rotary_dim=128 if full else 256,
            rope_theta=1000000.0 if full else 10000.0,
            rope_type="proportional" if full else "default",
            sliding_window=None if full else window,
            kv_source_layer=source, stores_shared_kv=i in providers.values(),
            alternative_k_eq_v=full and k_eq_v,
            mlp_width=ff * (2 if double_mlp and i >= first_shared and shared else 1),
            per_layer_input_width=ple,
        ))
    return TextPlan(variant, key[1], 262144, context, 30.0, 1e-6, tuple(layers))
