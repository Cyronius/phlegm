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
