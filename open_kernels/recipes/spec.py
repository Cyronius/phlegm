"""ModelSpec: the hyperparameter tuple a recipe composes kernels from.

One plain dataclass, JSON on disk, built from either of the two places a model
already publishes its shape: the HF-style `config.json` FLM ships beside the
`.q4nx` container, or a GGUF's metadata (`general.architecture`,
`<arch>.embedding_length`, ...). Anything a recipe needs that is not here is
a recipe constant (a family property), not a model property.

Traces: OPEN-SPEC-DERIVE (specs/open-engine/spec.md).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

LINEAR, FULL, DENSE = "linear_attention", "full_attention", "dense"
LAYER_TYPES = (LINEAR, FULL, DENSE)


class SpecError(ValueError):
    """The metadata cannot be turned into a ModelSpec; the message names the key."""


@dataclass(frozen=True)
class ModelSpec:
    family: str                       # the recipe: "qwen36moe" | "qwen3"
    hidden: int
    num_layers: int
    layer_types: tuple[str, ...]      # per layer: LINEAR | FULL | DENSE
    vocab: int                        # lm_head rows (padded)
    real_vocab: int                   # the tokenizer's ids; logits above it are undefined
    # full attention
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int                   # partial RoPE: rotated dims per head
    rope_theta: float
    qk_norm: bool = True
    attn_gate: bool = True            # sigmoid output gate (a second q-width projection)
    # linear attention (Gated DeltaNet); zeros for a family without it
    lin_key_heads: int = 0
    lin_value_heads: int = 0
    lin_key_dim: int = 0
    lin_value_dim: int = 0
    conv_kernel: int = 0
    # dense FFN (silu(gate) * up @ down); 0 for a MoE-only family
    intermediate: int = 0
    # MoE; num_experts == 0 means dense
    num_experts: int = 0
    experts_per_tok: int = 0
    moe_intermediate: int = 0
    shared_expert_intermediate: int = 0   # 0 = no shared expert
    norm_eps: float = 1e-6
    quant: str = "q4_1"
    extra: dict = field(default_factory=dict)   # informational (model name, source)

    # ---- derived
    @property
    def lin_qkv_dim(self) -> int:
        """Rows of the fused q|k|v projection of a linear layer: 2 key groups + value."""
        return 2 * self.lin_key_heads * self.lin_key_dim + self.lin_value_heads * self.lin_value_dim

    @property
    def lin_value_width(self) -> int:
        return self.lin_value_heads * self.lin_value_dim

    @property
    def attn_q_width(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def attn_kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def has_linear(self) -> bool:
        return LINEAR in self.layer_types

    @property
    def has_full(self) -> bool:
        return FULL in self.layer_types

    @property
    def has_dense(self) -> bool:
        return DENSE in self.layer_types

    # ---- serialisation
    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_types"] = list(self.layer_types)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelSpec":
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise SpecError(f"unknown ModelSpec field(s): {unknown}")
        kw = dict(d)
        kw["layer_types"] = tuple(kw["layer_types"])
        for t in kw["layer_types"]:
            if t not in LAYER_TYPES:
                raise SpecError(f"layer_types: unknown layer type {t!r}")
        if len(kw["layer_types"]) != kw["num_layers"]:
            raise SpecError(f"layer_types has {len(kw['layer_types'])} entries, num_layers is {kw['num_layers']}")
        return cls(**kw)

    @classmethod
    def from_json(cls, s: str) -> "ModelSpec":
        return cls.from_dict(json.loads(s))

    def spec_hash(self) -> str:
        """Stable hash of the hyperparameters (not of `extra`)."""
        d = self.to_dict()
        d.pop("extra", None)
        return "sha256:" + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    # ---- sources
    @classmethod
    def from_hf_config(cls, cfg: Mapping[str, Any], real_vocab: int | None = None) -> "ModelSpec":
        """From the HF-style config.json FLM ships with a model.

        `real_vocab` is the tokenizer's id count (tokenizer.json); it defaults
        to vocab_size, which for FLM's Qwen3.6 containers is the padded lm_head
        row count -- pass the real one when the tokenizer is at hand."""
        mt = cfg.get("model_type")
        if mt not in HF_FAMILIES:
            raise SpecError(f"model_type {mt!r} has no recipe (known: {sorted(HF_FAMILIES)})")
        return HF_FAMILIES[mt](cfg, real_vocab)

    @classmethod
    def from_gguf_metadata(cls, md: Mapping[str, Any]) -> "ModelSpec":
        """From a GGUF's key/value metadata (llama.cpp's key names)."""
        arch = md.get("general.architecture")
        if arch not in GGUF_FAMILIES:
            raise SpecError(f"general.architecture {arch!r} has no recipe (known: {sorted(GGUF_FAMILIES)})")
        return GGUF_FAMILIES[arch](md)


def _need(d: Mapping[str, Any], key: str, what: str = "config.json"):
    if key not in d:
        raise SpecError(f"{what} lacks {key!r}")
    return d[key]


def _layer_types_hf(cfg: Mapping[str, Any], n: int) -> tuple[str, ...]:
    if "layer_types" in cfg:
        lt = tuple(cfg["layer_types"])
        if len(lt) != n:
            raise SpecError(f"layer_types has {len(lt)} entries, num_hidden_layers is {n}")
        bad = sorted({t for t in lt if t not in (LINEAR, FULL)})
        if bad:
            raise SpecError(f"layer_types: unknown layer type(s) {bad}")
        return lt
    interval = _need(cfg, "full_attention_interval")
    if interval <= 0:
        raise SpecError("full_attention_interval must be positive")
    return tuple(FULL if (l + 1) % interval == 0 else LINEAR for l in range(n))


def _qwen36moe_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    n = _need(cfg, "num_hidden_layers")
    rope = cfg.get("rope_parameters", {})
    theta = rope.get("rope_theta", cfg.get("rope_theta"))
    if theta is None:
        raise SpecError("config.json lacks 'rope_theta' (top level or rope_parameters)")
    prf = rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
    hd = _need(cfg, "head_dim")
    vocab = _need(cfg, "vocab_size")
    return ModelSpec(
        family="qwen36moe",
        hidden=_need(cfg, "hidden_size"),
        num_layers=n,
        layer_types=_layer_types_hf(cfg, n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=_need(cfg, "num_attention_heads"),
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=int(round(hd * prf)),
        rope_theta=float(theta),
        qk_norm=True,
        attn_gate=bool(cfg.get("attn_output_gate", True)),
        lin_key_heads=_need(cfg, "linear_num_key_heads"),
        lin_value_heads=_need(cfg, "linear_num_value_heads"),
        lin_key_dim=_need(cfg, "linear_key_head_dim"),
        lin_value_dim=_need(cfg, "linear_value_head_dim"),
        conv_kernel=_need(cfg, "linear_conv_kernel_dim"),
        num_experts=_need(cfg, "num_experts"),
        experts_per_tok=_need(cfg, "num_experts_per_tok"),
        moe_intermediate=_need(cfg, "moe_intermediate_size"),
        shared_expert_intermediate=cfg.get("shared_expert_intermediate_size", 0),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )


def _qwen36moe_gguf(md: Mapping[str, Any]) -> ModelSpec:
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    n = k("block_count")
    hd = k("attention.key_length")
    vocab = md.get(f"{a}.vocab_size")
    if vocab is None:
        toks = md.get("tokenizer.ggml.tokens")
        if toks is None:
            raise SpecError(f"GGUF metadata lacks '{a}.vocab_size' and 'tokenizer.ggml.tokens'")
        vocab = len(toks)
    interval = k("full_attention_interval")
    if interval <= 0:
        raise SpecError(f"{a}.full_attention_interval must be positive")
    conv = k("ssm.conv_kernel")
    key_heads, val_heads = k("ssm.group_count"), k("ssm.time_step_rank")
    key_dim = k("ssm.state_size")
    inner = k("ssm.inner_size")
    if inner % val_heads:
        raise SpecError(f"{a}.ssm.inner_size {inner} is not a multiple of ssm.time_step_rank {val_heads}")
    return ModelSpec(
        family="qwen36moe",
        hidden=k("embedding_length"),
        num_layers=n,
        layer_types=tuple(FULL if (l + 1) % interval == 0 else LINEAR for l in range(n)),
        vocab=vocab,
        real_vocab=vocab,
        num_heads=k("attention.head_count"),
        num_kv_heads=k("attention.head_count_kv"),
        head_dim=hd,
        rotary_dim=md.get(f"{a}.rope.dimension_count", hd),
        rope_theta=float(k("rope.freq_base")),
        qk_norm=True,
        attn_gate=True,
        lin_key_heads=key_heads,
        lin_value_heads=val_heads,
        lin_key_dim=key_dim,
        lin_value_dim=inner // val_heads,
        conv_kernel=conv,
        num_experts=k("expert_count"),
        experts_per_tok=k("expert_used_count"),
        moe_intermediate=k("expert_feed_forward_length"),
        shared_expert_intermediate=md.get(f"{a}.expert_shared_feed_forward_length", 0),
        norm_eps=float(md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)),
        quant="q4_1",
        extra={"architecture": a, "source": "gguf"},
    )


def _qwen3_hf(cfg: Mapping[str, Any], real_vocab: int | None) -> ModelSpec:
    """Qwen3 dense: GQA with q/k RMSNorm, full RoPE, no attention gate, silu-gated FFN."""
    n = _need(cfg, "num_hidden_layers")
    hd = _need(cfg, "head_dim")
    vocab = _need(cfg, "vocab_size")
    if cfg.get("use_sliding_window"):
        raise SpecError("qwen3: use_sliding_window is not supported by the dense recipe")
    return ModelSpec(
        family="qwen3",
        hidden=_need(cfg, "hidden_size"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=real_vocab if real_vocab is not None else vocab,
        num_heads=_need(cfg, "num_attention_heads"),
        num_kv_heads=_need(cfg, "num_key_value_heads"),
        head_dim=hd,
        rotary_dim=hd,
        rope_theta=float(_need(cfg, "rope_theta")),
        qk_norm=True,
        attn_gate=False,
        intermediate=_need(cfg, "intermediate_size"),
        norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        quant="q4_1",
        extra={"model_type": cfg["model_type"], "source": "hf_config"},
    )


def _qwen3_gguf(md: Mapping[str, Any]) -> ModelSpec:
    a = md["general.architecture"]

    def k(name: str):
        return _need(md, f"{a}.{name}", "GGUF metadata")

    n = k("block_count")
    hd = k("attention.key_length")
    vocab = md.get(f"{a}.vocab_size")
    if vocab is None:
        toks = md.get("tokenizer.ggml.tokens")
        if toks is None:
            raise SpecError(f"GGUF metadata lacks '{a}.vocab_size' and 'tokenizer.ggml.tokens'")
        vocab = len(toks)
    return ModelSpec(
        family="qwen3",
        hidden=k("embedding_length"),
        num_layers=n,
        layer_types=tuple([DENSE] * n),
        vocab=vocab,
        real_vocab=vocab,
        num_heads=k("attention.head_count"),
        num_kv_heads=k("attention.head_count_kv"),
        head_dim=hd,
        rotary_dim=md.get(f"{a}.rope.dimension_count", hd),
        rope_theta=float(k("rope.freq_base")),
        qk_norm=True,
        attn_gate=False,
        intermediate=k("feed_forward_length"),
        norm_eps=float(md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)),
        quant="q4_1",
        extra={"architecture": a, "source": "gguf"},
    )


HF_FAMILIES = {"qwen3_5_moe": _qwen36moe_hf, "qwen3_next": _qwen36moe_hf, "qwen3": _qwen3_hf}
GGUF_FAMILIES = {"qwen35moe": _qwen36moe_gguf, "qwen3next": _qwen36moe_gguf, "qwen3": _qwen3_gguf}
_FAMILY_OF = {_qwen36moe_hf: "qwen36moe", _qwen36moe_gguf: "qwen36moe", _qwen3_hf: "qwen3", _qwen3_gguf: "qwen3"}


def hf_model_types(family: str) -> list[str]:
    """The config.json model_type values a family's recipe accepts."""
    return sorted(k for k, f in HF_FAMILIES.items() if _FAMILY_OF[f] == family)


def gguf_architectures(family: str) -> list[str]:
    return sorted(k for k, f in GGUF_FAMILIES.items() if _FAMILY_OF[f] == family)
