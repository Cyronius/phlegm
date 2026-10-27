"""Read FLM Gemma 3 vision BF16 weights and undo its 64×256 matrix tiles.

The tensor names and tiling rule come from FLM's
``utilities/q4nx-build/configs/gemma3.json`` and
``q4nx/models/gemma3.py``. This adapter is intentionally strict: it refuses
shapes that do not match the 4B SigLIP geometry used by the NPU images.
"""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_mm_source import source_matrix

VISION = "vision_tower.vision_model."
PROJECTOR = "multi_modal_projector."


def required_header() -> dict[str, tuple[int, ...]]:
    """Shapes of all 4B vision tensors needed by the complete NPU tower."""
    expected = {
        VISION + "embeddings.patch_embedding.weight": (1152, 3, 14, 14),
        VISION + "embeddings.patch_embedding.bias": (1152,),
        VISION + "embeddings.position_embedding.weight": (4096, 1152),
        VISION + "post_layernorm.weight": (1152,),
        VISION + "post_layernorm.bias": (1152,),
        PROJECTOR + "mm_soft_emb_norm.weight": (1152,),
        PROJECTOR + "mm_input_projection_weight": (40, 5, 16384),
    }
    for index in range(27):
        root = f"{VISION}encoder.layers.{index}."
        for norm in ("layer_norm1", "layer_norm2"):
            for suffix in ("weight", "bias"):
                expected[root + norm + "." + suffix] = (1152,)
        for part in "qkvo":
            stem = root + f"self_attn.{part if part != 'o' else 'out'}_proj."
            expected[stem + "weight"] = (20, 5, 16384)
            expected[stem + "bias"] = (1152,)
        for name, rows, cols in (("fc1", 4304, 1152), ("fc2", 1152, 4304)):
            stem = root + f"mlp.{name}."
            expected[stem + "weight"] = ((rows + 255) // 256 * 4,
                                           (cols + 255) // 256, 16384)
            expected[stem + "bias"] = (rows,)
    return expected


def validate_header(reader) -> int:
    """Reject an incomplete or differently packed vision container before NPU work."""
    expected = required_header()
    extents = []
    for name, shape in expected.items():
        meta = reader.tensors.get(name)
        if meta is None or meta.get("dtype") != "BF16" or tuple(meta.get("shape", ())) != shape:
            raise ValueError(f"{name}: expected BF16{shape}, got {meta}")
        offsets = meta.get("data_offsets")
        size = 2 * int(np.prod(shape))
        if (not isinstance(offsets, (list, tuple)) or len(offsets) != 2 or
                offsets[0] < 0 or offsets[1] - offsets[0] != size or
                reader.data_base + offsets[1] > len(reader.mm)):
            raise ValueError(f"{name}: invalid BF16 byte extent")
        extents.append((offsets[0], offsets[1], name))
    extents.sort()
    for previous, current in zip(extents, extents[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"overlapping vision tensors: {previous[2]} and {current[2]}")
    return len(expected)


def _tensor(reader, name: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    header = reader.tensors.get(name)
    if header is None or header.get("dtype") != dtype or tuple(header.get("shape", ())) != shape:
        raise ValueError(f"{name}: expected {dtype}{shape}, got {header}")
    raw = np.frombuffer(reader.raw(name), np.uint16)
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"{name}: wrong BF16 payload size")
    return raw.reshape(shape).view(bfloat16)


def vector(reader, name: str, width: int) -> np.ndarray:
    return np.ascontiguousarray(_tensor(reader, name, "BF16", (width,)))


def matrix(reader, name: str, rows: int, cols: int) -> np.ndarray:
    padded_rows = (rows + 255) // 256 * 256
    padded_cols = (cols + 255) // 256 * 256
    tiled = _tensor(reader, name, "BF16",
                    (padded_rows // 64, padded_cols // 256, 16384))
    raster = source_matrix(tiled, padded_rows, padded_cols).view(bfloat16)
    if np.any(raster[rows:] != 0) or np.any(raster[:, cols:] != 0):
        raise ValueError(f"{name}: nonzero FLM tile padding")
    return np.ascontiguousarray(raster[:rows, :cols])


def patch_embedding(reader) -> dict[str, np.ndarray]:
    return {
        "weight": np.ascontiguousarray(_tensor(reader, VISION +
            "embeddings.patch_embedding.weight", "BF16", (1152, 3, 14, 14))),
        "bias": vector(reader, VISION + "embeddings.patch_embedding.bias", 1152),
        "position": np.ascontiguousarray(_tensor(reader, VISION +
            "embeddings.position_embedding.weight", "BF16", (4096, 1152))),
    }


def layer(reader, index: int) -> dict[str, np.ndarray]:
    if not 0 <= index < 27:
        raise ValueError("Gemma 3 4B SigLIP has 27 layers")
    root = f"{VISION}encoder.layers.{index}."
    result = {}
    for part, norm in (("ln1", "layer_norm1"), ("ln2", "layer_norm2")):
        result[part + "_gamma"] = vector(reader, root + norm + ".weight", 1152)
        result[part + "_beta"] = vector(reader, root + norm + ".bias", 1152)
    qkv_weights = []
    qkv_biases = []
    for part in "qkv":
        prefix = root + f"self_attn.{part}_proj."
        qkv_weights.append(matrix(reader, prefix + "weight", 1152, 1152))
        qkv_biases.append(vector(reader, prefix + "bias", 1152))
    result["qkv_weight"] = np.concatenate(qkv_weights, axis=0)
    result["qkv_bias"] = np.concatenate(qkv_biases)
    for key, path, rows, cols in (
        ("attention_out", "self_attn.out_proj", 1152, 1152),
        ("mlp_fc1", "mlp.fc1", 4304, 1152),
        ("mlp_fc2", "mlp.fc2", 1152, 4304),
    ):
        result[key + "_weight"] = matrix(reader, root + path + ".weight", rows, cols)
        result[key + "_bias"] = vector(reader, root + path + ".bias", rows)
    return result


def post_layernorm(reader) -> tuple[np.ndarray, np.ndarray]:
    return (vector(reader, VISION + "post_layernorm.weight", 1152),
            vector(reader, VISION + "post_layernorm.bias", 1152))


def projector(reader) -> tuple[np.ndarray, np.ndarray]:
    return (vector(reader, PROJECTOR + "mm_soft_emb_norm.weight", 1152),
            matrix(reader, PROJECTOR + "mm_input_projection_weight", 2560, 1152))
