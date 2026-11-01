"""Strict MOSS Image SFT DiT tensor contract and native NPU GEMM slots.

The reference implementation is `pipeline_moss_image.py` from
OpenMOSS-Team/MOSS-Image-SFT. This module describes package and weight layout;
it does not imply that a complete MOSS inference runtime is available.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

try:
    from .image_safetensors import read_header
except ImportError:  # direct CLI execution from open_kernels/
    from image_safetensors import read_header

HIDDEN = 3072
FFN = 8192
TEXT_WIDTH = 4096
HEAD_DIM = 128
HEADS = 24
LAYERS = 42
TEXT_BLOCKS = 2
CHANNELS = 128
FREQ_DIM = 256
MODEL_TENSORS = 463
MODEL_BYTES = 10_164_617_280
ENCODER_TENSORS = 759
ENCODER_BYTES = 16_785_390_048
VAE_TENSORS = 251
MODEL_PARAMS = (MODEL_BYTES - 4 * (HEAD_DIM // 8)) // 2 + HEAD_DIM // 8


@dataclass(frozen=True)
class GemmSlot:
    """Source matrices are [N, K]; the NPU packer consumes padded [K, N]."""

    sources: tuple[str, ...]
    k: int
    n: int
    logical_n: int | None = None
    logical_k: int | None = None
    swiglu: bool = False
    gather_input: bool = False
    split_after: int | None = None


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "video_embedder.weight": (HIDDEN, CHANNELS),
        "video_embedder.bias": (HIDDEN,),
        "text_embedder.proj.weight": (HIDDEN, TEXT_WIDTH),
        "text_embedder.proj.bias": (HIDDEN,),
        "time_embedding.0.weight": (HIDDEN, FREQ_DIM),
        "time_embedding.0.bias": (HIDDEN,),
        "time_embedding.2.weight": (HIDDEN, HIDDEN),
        "time_embedding.2.bias": (HIDDEN,),
        "shared_block_modulation.weight": (6 * HIDDEN, HIDDEN),
        "shared_block_modulation.bias": (6 * HIDDEN,),
        "video_head_modulation.weight": (2 * HIDDEN, HIDDEN),
        "video_head_modulation.bias": (2 * HIDDEN,),
        "final_norm_video.weight": (HIDDEN,),
        "final_linear_video.weight": (CHANNELS, HIDDEN),
        "rope.bands": (HEAD_DIM // 8,),
    }
    for layer in range(TEXT_BLOCKS):
        prefix = f"text_embedder.blocks.{layer}"
        for name, width in (("norm1", HIDDEN), ("norm2", HIDDEN),
                            ("q_norm", HEAD_DIM), ("k_norm", HEAD_DIM)):
            shapes[f"{prefix}.{name}.weight"] = (width,)
        for name, out_width, in_width in (
                ("qkv", 3 * HIDDEN, HIDDEN), ("proj", HIDDEN, HIDDEN),
                ("ffn_up", FFN, HIDDEN), ("ffn_gate", FFN, HIDDEN),
                ("ffn_down", HIDDEN, FFN)):
            shapes[f"{prefix}.{name}.weight"] = (out_width, in_width)
            shapes[f"{prefix}.{name}.bias"] = (out_width,)
    for layer in range(LAYERS):
        prefix = f"layers.{layer}"
        for name, width in (("norm1", HIDDEN), ("norm2", HIDDEN),
                            ("q_norm", HEAD_DIM), ("k_norm", HEAD_DIM)):
            shapes[f"{prefix}.{name}.weight"] = (width,)
        for name, out_width, in_width in (
                ("q", HIDDEN, HIDDEN), ("k", HIDDEN, HIDDEN),
                ("v", HIDDEN, HIDDEN), ("o", HIDDEN, HIDDEN),
                ("ffn_up_gate", 2 * FFN, HIDDEN), ("ffn_down", HIDDEN, FFN)):
            shapes[f"{prefix}.proj.{name}.weight"] = (out_width, in_width)
    return shapes


def gemm_slots() -> dict[str, GemmSlot]:
    slots = {
        "emb.image": GemmSlot(("video_embedder.weight",), 512, HIDDEN,
                               logical_k=CHANNELS),
        "emb.text": GemmSlot(("text_embedder.proj.weight",), TEXT_WIDTH, HIDDEN),
        "time.in": GemmSlot(("time_embedding.0.weight",), 512, HIDDEN,
                            logical_k=FREQ_DIM),
        "time.out": GemmSlot(("time_embedding.2.weight",), HIDDEN, HIDDEN),
        "shared.mod": GemmSlot(("shared_block_modulation.weight",), HIDDEN, 6 * HIDDEN),
        "head.mod": GemmSlot(("video_head_modulation.weight",), HIDDEN, 2 * HIDDEN),
        "final.image": GemmSlot(("final_linear_video.weight",), HIDDEN, 1024,
                                logical_n=CHANNELS),
    }
    for layer in range(TEXT_BLOCKS):
        prefix = f"text_embedder.blocks.{layer}"
        name = f"text.{layer}"
        slots[f"{name}.qkv"] = GemmSlot((f"{prefix}.qkv.weight",), HIDDEN, 3 * HIDDEN)
        slots[f"{name}.attn_out"] = GemmSlot((f"{prefix}.proj.weight",), HIDDEN, HIDDEN)
        # Keep [up | gate] contiguous for MOSS SiTU; SwiGLU interleaving is wrong.
        # 8192 live channels occupy each 9216-wide half. dit_ew's SiTU
        # processes three 3072-wide chunks per token without crossing halves.
        slots[f"{name}.ffn_in"] = GemmSlot((f"{prefix}.ffn_up.weight",
                                             f"{prefix}.ffn_gate.weight"),
                                            HIDDEN, 18_432, logical_n=2 * FFN,
                                            split_after=FFN)
        slots[f"{name}.ffn_out"] = GemmSlot((f"{prefix}.ffn_down.weight",),
                                             9_216, HIDDEN, logical_k=FFN)
    for layer in range(LAYERS):
        prefix = f"layers.{layer}.proj"
        name = f"block.{layer}"
        slots[f"{name}.qkv"] = GemmSlot(tuple(f"{prefix}.{qkv}.weight"
                                               for qkv in ("q", "k", "v")),
                                            HIDDEN, 3 * HIDDEN)
        slots[f"{name}.attn_out"] = GemmSlot((f"{prefix}.o.weight",), HIDDEN, HIDDEN)
        slots[f"{name}.ffn_in"] = GemmSlot((f"{prefix}.ffn_up_gate.weight",),
                                            HIDDEN, 18_432, logical_n=2 * FFN,
                                            split_after=FFN)
        slots[f"{name}.ffn_out"] = GemmSlot((f"{prefix}.ffn_down.weight",),
                                             9_216, HIDDEN, logical_k=FFN)
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
        if slot.swiglu or slot.gather_input:
            raise ValueError(f"{name}: MOSS SiTU cannot use a SwiGLU slot")
        if slot.split_after is not None and (slot.split_after != FFN or
                                             slot.n != 18_432 or
                                             slot.logical_n != 2 * FFN):
            raise ValueError(f"{name}: invalid SiTU half padding")
        used.extend(slot.sources)
    expected_matrices = {name for name, shape in shapes.items() if len(shape) == 2}
    if len(used) != len(set(used)) or set(used) != expected_matrices:
        raise ValueError("MOSS GEMM slots do not cover every matrix exactly once")
    return len(slots)


def inspect(model_dir: Path) -> dict[str, int | str]:
    """Reject any source package whose tensor layout differs from native slots."""
    model = json.loads((model_dir / "model_index.json").read_text())
    config = json.loads((model_dir / "transformer" / "config.json").read_text())
    encoder = json.loads((model_dir / "text_encoder" / "config.json").read_text())
    vae = json.loads((model_dir / "vae" / "config.json").read_text())
    if (model.get("_class_name") != "MOSSImagePipeline" or
            model.get("transformer") != ["pipeline_moss_image", "MOSSImageTransformer"] or
            model.get("text_encoder") != ["transformers", "Qwen3_5Model"] or
            model.get("vae") != ["diffusers", "AutoencoderKLFlux2"]):
        raise ValueError("expected MOSS Image SFT pipeline components")
    required = {"_class_name": "MOSSImageTransformer", "dim": HIDDEN,
                "ffn_dim": FFN, "freq_dim": FREQ_DIM, "latent_channels": CHANNELS,
                "num_heads": HEADS, "num_layers": LAYERS, "rope_base": 10000.0,
                "text_dim": TEXT_WIDTH}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("unsupported MOSS transformer configuration")
    text = encoder.get("text_config", {})
    if (encoder.get("model_type"), text.get("hidden_size"),
        text.get("num_hidden_layers")) != ("qwen3_5", TEXT_WIDTH, 32):
        raise ValueError("unsupported MOSS Qwen3.5 text encoder")
    if (vae.get("_class_name"), vae.get("latent_channels"),
        vae.get("decoder_block_out_channels"), vae.get("patch_size"),
        vae.get("out_channels")) != (
            "AutoencoderKLFlux2", 32, [96, 192, 384, 384], [2, 2], 3):
        raise ValueError("unsupported MOSS FLUX.2 small VAE geometry")
    tokenizer = json.loads((model_dir / "tokenizer" / "tokenizer_config.json").read_text())
    if (tokenizer.get("tokenizer_class") != "Qwen2Tokenizer" or
            not (model_dir / "tokenizer" / "tokenizer.json").is_file() or
            not (model_dir / "tokenizer" / "chat_template.jinja").is_file()):
        raise ValueError("MOSS Qwen2 tokenizer or chat template is missing")

    text_index = json.loads((model_dir / "text_encoder" /
                             "model.safetensors.index.json").read_text())
    text_map = text_index["weight_map"]
    if (len(text_map) != ENCODER_TENSORS or
            text_index.get("metadata", {}).get("total_size") != ENCODER_BYTES or
            len(set(text_map.values())) != 4):
        raise ValueError("MOSS text encoder index differs from the pinned package")
    text_seen = set()
    text_bytes = 0
    for shard in sorted(set(text_map.values())):
        for name, entry in read_header(model_dir / "text_encoder" / shard).items():
            size = math.prod(entry["shape"]) * 2
            if (name in text_seen or text_map.get(name) != shard or
                    entry["dtype"] != "BF16" or
                    entry["data_offsets"][1] - entry["data_offsets"][0] != size):
                raise ValueError(f"MOSS text encoder tensor differs at {name}")
            text_seen.add(name)
            text_bytes += size
    if text_seen != set(text_map) or text_bytes != ENCODER_BYTES:
        raise ValueError("MOSS text encoder payload differs from its index")

    vae_tensors = read_header(model_dir / "vae" / "diffusion_pytorch_model.safetensors")
    if len(vae_tensors) != VAE_TENSORS:
        raise ValueError("MOSS VAE tensor count differs")
    for name, entry in vae_tensors.items():
        if (entry["dtype"] not in ("F32", "I64") or
                entry["data_offsets"][1] - entry["data_offsets"][0] !=
                math.prod(entry["shape"]) * (8 if entry["dtype"] == "I64" else 4)):
            raise ValueError(f"MOSS VAE tensor differs at {name}")
    if (vae_tensors["bn.running_mean"]["shape"] != [CHANNELS] or
            vae_tensors["bn.running_var"]["shape"] != [CHANNELS]):
        raise ValueError("MOSS VAE latent normalization differs")
    index = json.loads((model_dir / "transformer" /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    expected = expected_shapes()
    slot_count = _check_slot_coverage(expected)
    if len(expected) != MODEL_TENSORS or set(weight_map) != set(expected):
        raise ValueError("MOSS tensor index differs from the native layout")
    if index.get("metadata", {}).get("total_size") != MODEL_BYTES:
        raise ValueError("MOSS transformer byte count differs")
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        for name, entry in read_header(model_dir / "transformer" / shard).items():
            if name in tensors or weight_map.get(name) != shard:
                raise ValueError(f"MOSS shard index differs at {name}")
            tensors[name] = entry
    if set(tensors) != set(expected):
        raise ValueError("MOSS transformer payload keys differ from the native layout")
    for name, shape in expected.items():
        entry = tensors[name]
        dtype = "F32" if name == "rope.bands" else "BF16"
        if entry["dtype"] != dtype or tuple(entry["shape"]) != shape:
            raise ValueError(f"MOSS tensor {name} has {entry['dtype']} {entry['shape']}; "
                             f"expected {dtype} {shape}")
    return {"model": "MOSS-Image-SFT", "tensors": len(tensors),
            "params": MODEL_PARAMS, "gemm_slots": slot_count,
            "encoder_tensors": len(text_seen), "vae_tensors": len(vae_tensors)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    print(json.dumps(inspect(parser.parse_args().model_dir), indent=2))
