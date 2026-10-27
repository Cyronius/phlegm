"""Bounded reads from a GPT-OSS-20B q4nx container for the checked NPU images.

This is a source adapter, not a model recipe. It reads one projection call or
one expert at a time and rejects unexpected shapes and byte counts before
passing container bytes to the existing kernel-specific pack selectors.
"""
from __future__ import annotations

from math import prod

import numpy as np
from ml_dtypes import bfloat16

from designs.gptoss_expert_gemv.pack import ONE_EXPERT_CHUNKS, projection as expert_projection
from designs.gptoss_o_gemv.pack import projection as o_projection
from designs.gptoss_qkv_gemv.pack import ROLE_ROWS, projection as qkv_projection

CHUNK = 2560
EXPERT_SHAPE = (32, 69, 23, 4, CHUNK)
ATTN_SHAPES = {
    "q": (32, 23, 4, CHUNK),
    "k": (4, 23, 4, CHUNK),
    "v": (4, 23, 4, CHUNK),
    "o": (24, 32, 4, CHUNK),
}
CONSTANTS = {
    "input_norm": ("input_layernorm.weight", (2880,)),
    "post_attention_norm": ("post_attention_layernorm.weight", (2880,)),
    "sinks": ("self_attn.sinks.weight", (64,)),
    "q_bias": ("self_attn.q_proj.bias", (4096,)),
    "k_bias": ("self_attn.k_proj.bias", (512,)),
    "v_bias": ("self_attn.v_proj.bias", (512,)),
    "o_bias": ("self_attn.o_proj.bias", (2880,)),
    "router_bias": ("mlp.router.bias", (32,)),
}


class GptOssSource:
    """Use an open ``model.q4nx`` reader without copying a full expert tensor."""

    def __init__(self, reader):
        self.reader = reader

    @staticmethod
    def _prefix(layer: int) -> str:
        if not isinstance(layer, int) or layer < 0 or layer >= 24:
            raise ValueError("GPT-OSS 20B layer must be in [0,24)")
        return f"model.layers.{layer}."

    def _bytes(self, name: str, dtype: str, shape: tuple[int, ...],
               start: int = 0, count: int | None = None) -> bytes:
        tensor = self.reader.tensors.get(name)
        if tensor is None:
            raise ValueError(f"missing GPT-OSS tensor {name}")
        if tensor.get("dtype") != dtype or tuple(tensor.get("shape", ())) != shape:
            raise ValueError(f"unexpected GPT-OSS tensor layout: {name}: {tensor}")
        size = prod(shape) * (2 if dtype == "BF16" else 1)
        offsets = tensor.get("data_offsets")
        if (not isinstance(offsets, (list, tuple)) or len(offsets) != 2
                or offsets[0] < 0 or offsets[1] - offsets[0] != size):
            raise ValueError(f"invalid GPT-OSS tensor byte extent: {name}")
        if count is None:
            count = size - start
        if start < 0 or count < 0 or start + count > size:
            raise ValueError(f"GPT-OSS tensor slice out of range: {name}")
        begin = self.reader.data_base + offsets[0] + start
        end = begin + count
        if end > len(self.reader.mm):
            raise ValueError(f"truncated GPT-OSS tensor: {name}")
        result = self.reader.mm[begin:end]
        if len(result) != count:
            raise ValueError(f"short GPT-OSS tensor read: {name}")
        return result

    def constant(self, layer: int, role: str) -> np.ndarray:
        if role not in CONSTANTS:
            raise ValueError(f"unknown GPT-OSS constant {role}")
        suffix, shape = CONSTANTS[role]
        name = self._prefix(layer) + suffix
        return np.frombuffer(self._bytes(name, "BF16", shape), bfloat16).copy()

    def router_weight(self, layer: int) -> np.ndarray:
        """Return BF16 [2880,32], the orientation expected by gptoss_router."""
        name = self._prefix(layer) + "mlp.router.weight_prefill"
        raw = self._bytes(name, "BF16", (32, 2944))
        return np.ascontiguousarray(np.frombuffer(raw, bfloat16).reshape(32, 2944)[:, :2880].T)

    def attention(self, layer: int, role: str, start_band: int) -> np.ndarray:
        """Return one aligned 16-band q4_1 projection call in kernel order."""
        if role not in ATTN_SHAPES:
            raise ValueError(f"unknown GPT-OSS attention projection {role}")
        name = self._prefix(layer) + f"self_attn.{role}_proj.weight"
        raw = self._bytes(name, "I8", ATTN_SHAPES[role])
        if role == "o":
            return o_projection(raw, start_band)
        if start_band not in range(0, ROLE_ROWS[role] // 32, 16):
            raise ValueError(f"{role}: expected an aligned 16-band call")
        return qkv_projection(raw, role, start_band, 16)

    def expert(self, layer: int, expert_id: int, role: str) -> np.ndarray:
        """Return one expert's gate/up/down 90-band MXFP4 source stream."""
        if not isinstance(expert_id, int) or expert_id < 0 or expert_id >= 32:
            raise ValueError("GPT-OSS expert id must be in [0,32)")
        one_bytes = ONE_EXPERT_CHUNKS * CHUNK
        name = self._prefix(layer) + "ffn_gate_up_down_exps.weight"
        raw = self._bytes(name, "U8", EXPERT_SHAPE,
                          expert_id * one_bytes, one_bytes)
        return expert_projection(raw, role, 90)

    def expert_bias(self, layer: int, expert_id: int, role: str) -> np.ndarray:
        if role not in ("gate", "up", "down"):
            raise ValueError(f"unknown GPT-OSS expert projection {role}")
        if not isinstance(expert_id, int) or expert_id < 0 or expert_id >= 32:
            raise ValueError("GPT-OSS expert id must be in [0,32)")
        name = self._prefix(layer) + f"mlp.experts.{role}_proj_bias"
        return np.frombuffer(self._bytes(name, "BF16", (32, 2880),
                                         expert_id * 2880 * 2, 2880 * 2), bfloat16).copy()
