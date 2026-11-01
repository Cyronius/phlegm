"""Qwen Image 2.1 BF16 transformer tensor contract for native NPU packing.

The published pipeline also needs its Qwen3-VL encoder and RGBA autoencoder.
This module validates their identities and maps the DiT matrices; it does not
register an inference backend or claim end-to-end NPU support.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

try:
    from open_kernels.image_safetensors import read_header
except ModuleNotFoundError:  # direct script invocation
    from image_safetensors import read_header

HIDDEN = 4096
MLP = 12288
HEAD_DIM = 128
LAYERS = 32
CHANNELS = 64
MODEL_PARAMS = 7_115_124_736
MODEL_TENSORS = 297


@dataclass(frozen=True)
class GemmSlot:
    """Source matrices are [N, K]; the NPU packer consumes [K, N]."""

    sources: tuple[str, ...]
    k: int
    n: int
    logical_n: int | None = None
    logical_k: int | None = None
    swiglu: bool = False
    gather_input: bool = False


def expected_shapes() -> dict[str, tuple[int, ...]]:
    """The complete 297-key source contract from the pinned transformer."""
    shapes = {
        "img_in.weight": (HIDDEN, CHANNELS),
        "modulation.1.weight": (4 * HIDDEN, HIDDEN),
        "time_text_embed.timestep_embedder.linear_1.weight": (HIDDEN, 256),
        "time_text_embed.timestep_embedder.linear_2.weight": (HIDDEN, HIDDEN),
        "txt_in.in_layer.weight": (HIDDEN, HIDDEN),
        "txt_in.out_layer.weight": (HIDDEN, HIDDEN),
        "txt_in.text_norm.weight": (HIDDEN,),
        "norm_out.linear.weight": (HIDDEN, HIDDEN),
        "proj_out.weight": (CHANNELS, HIDDEN),
    }
    for layer in range(LAYERS):
        prefix = f"transformer_blocks.{layer}"
        for projection in ("to_q", "to_k", "to_v", "to_out.0"):
            shapes[f"{prefix}.attn.{projection}.weight"] = (HIDDEN, HIDDEN)
        for projection in ("norm_q", "norm_k"):
            shapes[f"{prefix}.attn.{projection}.weight"] = (HEAD_DIM,)
        for projection in ("gate_layer", "proj"):
            shapes[f"{prefix}.img_mlp.{projection}.weight"] = (MLP, HIDDEN)
        shapes[f"{prefix}.img_mlp.out.weight"] = (HIDDEN, MLP)
    assert len(shapes) == MODEL_TENSORS
    assert sum(math.prod(shape) for shape in shapes.values()) == MODEL_PARAMS
    return shapes


def gemm_slots() -> dict[str, GemmSlot]:
    slots = {
        "emb.image": GemmSlot(("img_in.weight",), 512, HIDDEN, logical_k=CHANNELS),
        "mod": GemmSlot(("modulation.1.weight",), HIDDEN, 4 * HIDDEN),
        "time.in": GemmSlot(("time_text_embed.timestep_embedder.linear_1.weight",),
                            512, HIDDEN, logical_k=256),
        "time.out": GemmSlot(("time_text_embed.timestep_embedder.linear_2.weight",), HIDDEN, HIDDEN),
        "text.in": GemmSlot(("txt_in.in_layer.weight",), HIDDEN, HIDDEN),
        "text.out": GemmSlot(("txt_in.out_layer.weight",), HIDDEN, HIDDEN),
        "final.mod": GemmSlot(("norm_out.linear.weight",), HIDDEN, HIDDEN),
        # dit_gemm's column tile is 1024; only the first 64 channels are live.
        "final.image": GemmSlot(("proj_out.weight",), HIDDEN, 1024, CHANNELS),
    }
    for layer in range(LAYERS):
        prefix = f"transformer_blocks.{layer}"
        name = f"block.{layer}"
        slots[f"{name}.qkv"] = GemmSlot(
            tuple(f"{prefix}.attn.to_{projection}.weight" for projection in ("q", "k", "v")),
            HIDDEN, 3 * HIDDEN,
        )
        slots[f"{name}.attn_out"] = GemmSlot((f"{prefix}.attn.to_out.0.weight",), HIDDEN, HIDDEN)
        slots[f"{name}.mlp_in"] = GemmSlot(
            (f"{prefix}.img_mlp.gate_layer.weight", f"{prefix}.img_mlp.proj.weight"),
            HIDDEN, 2 * MLP, swiglu=True,
        )
        slots[f"{name}.mlp_out"] = GemmSlot((f"{prefix}.img_mlp.out.weight",), MLP, HIDDEN,
                                            gather_input=True)
    return slots


def _check_slot_coverage(shapes: dict[str, tuple[int, ...]]) -> int:
    slots = gemm_slots()
    used = []
    for name, slot in slots.items():
        source_shapes = [shapes[source] for source in slot.sources]
        if slot.k % 512 or slot.n % 1024:
            raise ValueError(f"{name}: GEMM slot violates dit_gemm tile geometry")
        if any(len(shape) != 2 or shape[1] != (slot.logical_k or slot.k)
               for shape in source_shapes):
            raise ValueError(f"{name}: GEMM source K differs from its NPU slot")
        if sum(shape[0] for shape in source_shapes) != (slot.logical_n or slot.n):
            raise ValueError(f"{name}: GEMM source N differs from its NPU slot")
        used.extend(slot.sources)
    used.append("txt_in.text_norm.weight")
    used.extend(f"transformer_blocks.{layer}.attn.norm_{kind}.weight"
                for layer in range(LAYERS) for kind in ("q", "k"))
    if len(used) != len(set(used)) or set(used) != set(shapes):
        raise ValueError("Qwen Image 2.1 native slots do not cover every source tensor exactly once")
    return len(slots)


def inspect(model_dir: Path) -> dict[str, int | str]:
    """Reject a different checkpoint before a native packer reads its tensors."""
    model_index = json.loads((model_dir / "model_index.json").read_text())
    config = json.loads((model_dir / "transformer" / "config.json").read_text())
    vae = json.loads((model_dir / "vae" / "config.json").read_text())
    encoder = json.loads((model_dir / "text_encoder" / "config.json").read_text())
    if model_index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("expected Qwen Image 2.1 pipeline")
    required = {"_class_name": "QwenImage21Transformer2DModel", "attention_head_dim": HEAD_DIM,
                "context_in_dim": HIDDEN, "in_channels": CHANNELS, "out_channels": CHANNELS,
                "num_attention_heads": 32, "num_layers": LAYERS, "causal_condition": True}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("unsupported Qwen Image 2.1 transformer config")
    if (vae.get("_class_name"), vae.get("z_dim"), vae.get("out_channels"),
        vae.get("scale_factor_spatial")) != ("AutoencoderKLQwenImage21", CHANNELS, 4, 16):
        raise ValueError("unsupported Qwen Image 2.1 RGBA autoencoder config")
    text = encoder.get("text_config", {})
    if (encoder.get("model_type"), text.get("hidden_size"), text.get("num_hidden_layers")) != \
            ("qwen3_vl", HIDDEN, 36):
        raise ValueError("unsupported Qwen Image 2.1 text encoder config")
    index = json.loads((model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    tensors = {}
    for file in files:
        for name, entry in read_header(model_dir / "transformer" / file).items():
            if name in tensors or weight_map.get(name) != file:
                raise ValueError(f"Qwen Image 2.1 shard index differs at {name}")
            tensors[name] = entry
    expected = expected_shapes()
    slot_count = _check_slot_coverage(expected)
    if set(tensors) != set(expected) or set(weight_map) != set(expected):
        raise ValueError("Qwen Image 2.1 transformer keys differ from the native layout")
    if index.get("metadata", {}).get("total_size") != 2 * MODEL_PARAMS:
        raise ValueError("Qwen Image 2.1 transformer byte count differs")
    for name, shape in expected.items():
        entry = tensors[name]
        if entry["dtype"] != "BF16" or tuple(entry["shape"]) != shape:
            raise ValueError(f"Qwen Image 2.1 tensor {name} has {entry['dtype']} {entry['shape']}; expected BF16 {shape}")
    return {"model": "Qwen-Image-2.1", "tensors": len(tensors), "params": MODEL_PARAMS,
            "gemm_slots": slot_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    print(json.dumps(inspect(parser.parse_args().model_dir), indent=2))
