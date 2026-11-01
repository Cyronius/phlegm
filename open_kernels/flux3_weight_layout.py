"""FLUX 3 Image Klein BF16 DiT tensor contract for native NPU packing.

This module validates the reference checkpoint and names the GEMM weight slots.
It does not register the model or claim end-to-end NPU support. The checkpoint
reader only examines the safetensors header; callers must separately verify the
file hash before exporting weights.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

try:
    from open_kernels.image_safetensors import read_header
except ModuleNotFoundError:  # direct `python open_kernels/flux3_weight_layout.py`
    from image_safetensors import read_header

HIDDEN = 3072
MLP = 9216
HEAD_DIM = 128
DEPTH = 5
DEPTH_SINGLE = 28
CHANNELS = 128
TEXT_WIDTH = 20480
MODEL_PARAMS = 5_548_681_984
MODEL_TENSORS = 359
MODES = ("image", "image_cond", "txt")


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


def block_prefixes() -> tuple[tuple[str, str], ...]:
    early = tuple(
        (f"{mode}.{layer}", f"{family}.{layer}")
        for mode, family in (
            ("image", "content_mode_blocks.image"),
            ("image_cond", "content_mode_blocks.image_cond"),
            ("text", "txt_mode_blocks"),
        )
        for layer in range(DEPTH)
    )
    shared = tuple(
        (f"single.{layer}", f"single_blocks.{layer}")
        for layer in range(DEPTH_SINGLE)
    )
    return early + shared


def expected_shapes() -> dict[str, tuple[int, ...]]:
    """The complete 359-key source contract from the pinned reference DiT."""
    shapes = {
        "emb_in.image.weight": (HIDDEN, CHANNELS),
        "emb_in.image_cond.weight": (HIDDEN, CHANNELS),
        "txt_in.weight": (HIDDEN, TEXT_WIDTH),
        "time_in.in_layer.weight": (HIDDEN, 256),
        "time_in.out_layer.weight": (HIDDEN, HIDDEN),
        "ref_time_in.in_layer.weight": (HIDDEN, 256),
        "ref_time_in.out_layer.weight": (HIDDEN, HIDDEN),
        "final_layer.image.adaLN_modulation.1.weight": (2 * HIDDEN, HIDDEN),
        "final_layer.image.linear.weight": (CHANNELS, HIDDEN),
    }
    for stage in ("early_stream_modulations", "single_stream_modulations"):
        for mode in MODES:
            shapes[f"{stage}.{mode}.lin.weight"] = (3 * HIDDEN, HIDDEN)
    for _, prefix in block_prefixes():
        for projection in ("q_proj", "k_proj", "v_proj", "attn_out"):
            shapes[f"{prefix}.{projection}.weight"] = (HIDDEN, HIDDEN)
        shapes[f"{prefix}.mlp_in.weight"] = (2 * MLP, HIDDEN)
        shapes[f"{prefix}.mlp_out.weight"] = (HIDDEN, MLP)
        for projection in ("query_norm", "key_norm"):
            shapes[f"{prefix}.norm.{projection}.scale"] = (HEAD_DIM,)
    assert len(shapes) == MODEL_TENSORS
    assert sum(math.prod(shape) for shape in shapes.values()) == MODEL_PARAMS
    return shapes


def gemm_slots() -> dict[str, GemmSlot]:
    """NPU GEMM slots; norms and RoPE are handled by elementwise kernels."""
    slots = {
        "emb.image": GemmSlot(("emb_in.image.weight",), 512, HIDDEN, logical_k=CHANNELS),
        "emb.image_cond": GemmSlot(("emb_in.image_cond.weight",), 512, HIDDEN, logical_k=CHANNELS),
        "emb.text": GemmSlot(("txt_in.weight",), TEXT_WIDTH, HIDDEN),
        "time.in": GemmSlot(("time_in.in_layer.weight",), 512, HIDDEN, logical_k=256),
        "time.out": GemmSlot(("time_in.out_layer.weight",), HIDDEN, HIDDEN),
        "ref_time.in": GemmSlot(("ref_time_in.in_layer.weight",), 512, HIDDEN, logical_k=256),
        "ref_time.out": GemmSlot(("ref_time_in.out_layer.weight",), HIDDEN, HIDDEN),
        "final.mod": GemmSlot(("final_layer.image.adaLN_modulation.1.weight",), HIDDEN, 2 * HIDDEN),
        # dit_gemm's column tile is 1024; only the first 128 channels are live.
        "final.image": GemmSlot(("final_layer.image.linear.weight",), HIDDEN, 1024, CHANNELS),
    }
    for stage, tag in (("early_stream_modulations", "early"),
                       ("single_stream_modulations", "single")):
        for mode in MODES:
            slots[f"{tag}.mod.{mode}"] = GemmSlot((f"{stage}.{mode}.lin.weight",), HIDDEN, 3 * HIDDEN)
    for name, prefix in block_prefixes():
        slots[f"{name}.qkv"] = GemmSlot(
            tuple(f"{prefix}.{projection}.weight" for projection in ("q_proj", "k_proj", "v_proj")),
            HIDDEN, 3 * HIDDEN,
        )
        slots[f"{name}.attn_out"] = GemmSlot((f"{prefix}.attn_out.weight",), HIDDEN, HIDDEN)
        slots[f"{name}.mlp_in"] = GemmSlot((f"{prefix}.mlp_in.weight",), HIDDEN, 2 * MLP,
                                           swiglu=True)
        slots[f"{name}.mlp_out"] = GemmSlot((f"{prefix}.mlp_out.weight",), MLP, HIDDEN,
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
    used.extend(f"{prefix}.norm.{kind}.scale"
                for _, prefix in block_prefixes()
                for kind in ("query_norm", "key_norm"))
    if len(used) != len(set(used)) or set(used) != set(shapes):
        raise ValueError("FLUX 3 native slots do not cover each source tensor exactly once")
    return len(slots)


def inspect(model_dir: Path) -> dict[str, int | str]:
    """Reject a different checkpoint before a native packer reads its tensors."""
    config = json.loads((model_dir / "config.json").read_text())
    manifest = json.loads((model_dir / "manifest.json").read_text())
    dit = config.get("dit", {})
    if config.get("model") != "flux-3-image-klein" or not dit.get("flow_map"):
        raise ValueError("expected the distilled FLUX 3 Image Klein flow-map checkpoint")
    for key, value in (("hidden", HIDDEN), ("heads", 24), ("depth", DEPTH),
                       ("depth_single", DEPTH_SINGLE)):
        if dit.get(key) != value:
            raise ValueError(f"unsupported FLUX 3 {key}: {dit.get(key)!r}")
    if dit.get("in_channels") != {"image": CHANNELS, "image_cond": CHANNELS, "text": TEXT_WIDTH}:
        raise ValueError("unsupported FLUX 3 input widths")
    if manifest.get("tensors") != MODEL_TENSORS or manifest.get("params") != MODEL_PARAMS:
        raise ValueError("FLUX 3 checkpoint manifest differs from the native layout")
    tensors = read_header(model_dir / "model.safetensors")
    expected = expected_shapes()
    slot_count = _check_slot_coverage(expected)
    if set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))
        extra = sorted(set(tensors) - set(expected))
        raise ValueError(f"FLUX 3 tensor keys differ: missing={missing[:4]}, extra={extra[:4]}")
    for name, shape in expected.items():
        entry = tensors[name]
        if entry["dtype"] != "BF16" or tuple(entry["shape"]) != shape:
            raise ValueError(f"FLUX 3 tensor {name} has {entry['dtype']} {entry['shape']}; expected BF16 {shape}")
    return {"model": config["model"], "tensors": len(tensors), "params": MODEL_PARAMS,
            "gemm_slots": slot_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    print(json.dumps(inspect(parser.parse_args().model_dir), indent=2))
