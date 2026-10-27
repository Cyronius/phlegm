"""Bounded source reads for the pinned FLM Gemma 4 text containers.

The reader exposes a q4nx mmap and header. Large projection and vocabulary
matrices are read one NPU band at a time, in the kernel's pool order.
"""
from __future__ import annotations

from math import prod

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_source import (q4_projection_pool, q8_head_pool,
                                        source_shape, tensor_name, validate_header)
from open_kernels.gemma4_text_plan import Projection, TextPlan


class Gemma4TextSource:
    def __init__(self, reader, plan: TextPlan):
        if plan.variant not in ("E2B", "E4B", "12B"):
            raise ValueError(f"unsupported Gemma 4 source variant: {plan.variant}")
        self.reader = reader
        self.plan = plan

    def validate(self) -> dict:
        """Check the complete required header and every required byte extent."""
        report = validate_header(self.plan, self.reader.tensors)
        if not report["passed"]:
            raise ValueError("invalid Gemma 4 source header: " + "; ".join(report["errors"][:4]))
        spans = []
        for name, item in self.reader.tensors.items():
            if name == "__metadata__":
                continue
            offsets = item.get("data_offsets")
            if (not isinstance(offsets, (list, tuple)) or len(offsets) != 2 or
                    not all(isinstance(x, int) for x in offsets) or
                    offsets[0] < 0 or offsets[1] < offsets[0] or
                    self.reader.data_base + offsets[1] > len(self.reader.mm)):
                raise ValueError(f"invalid Gemma 4 tensor byte extent: {name}")
            spans.append((offsets[0], offsets[1], name))
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if previous[1] > current[0]:
                raise ValueError(f"overlapping Gemma 4 tensors: {previous[2]} and {current[2]}")
        return report

    def _bytes(self, name: str, dtype: str, shape: tuple[int, ...],
               start: int = 0, count: int | None = None) -> bytes:
        item = self.reader.tensors.get(name)
        if item is None or item.get("dtype") != dtype or tuple(item.get("shape", ())) != shape:
            raise ValueError(f"unexpected Gemma 4 tensor layout: {name}: {item}")
        unit = {"I8": 1, "BF16": 2, "F32": 4}[dtype]
        size = prod(shape) * unit
        offsets = item.get("data_offsets")
        if (not isinstance(offsets, (list, tuple)) or len(offsets) != 2 or
                not all(isinstance(x, int) for x in offsets) or
                offsets[0] < 0 or offsets[1] - offsets[0] != size or
                self.reader.data_base + offsets[1] > len(self.reader.mm)):
            raise ValueError(f"invalid Gemma 4 tensor byte extent: {name}")
        if count is None:
            count = size - start
        if start < 0 or count < 0 or start + count > size:
            raise ValueError(f"Gemma 4 tensor slice out of range: {name}")
        begin = self.reader.data_base + offsets[0] + start
        result = self.reader.mm[begin:begin + count]
        if len(result) != count:
            raise ValueError(f"short Gemma 4 tensor read: {name}")
        return result

    def projection_band(self, projection: Projection, band: int,
                        stripe: int = 0, image_cols: int | None = None) -> np.ndarray:
        """One 64-row q4 band, optionally one K stripe, in NPU pool order."""
        if projection.format != "q4" or projection.rows % 64 or projection.cols % 256:
            raise ValueError("expected a tiled q4 projection")
        cols = projection.cols
        image_cols = cols if image_cols is None else image_cols
        if (not isinstance(band, int) or band < 0 or band >= projection.rows // 64 or
                image_cols <= 0 or image_cols % 256 or cols % image_cols or
                not isinstance(stripe, int) or stripe < 0 or stripe >= cols // image_cols):
            raise ValueError("invalid Gemma 4 projection band or stripe")
        dtype, shape = source_shape(self.plan, projection)
        tile_bytes = 2304 if self.plan.variant == "12B" else 5120
        source_rows = 16 if self.plan.variant == "12B" else 32
        tiles = cols // 256
        selected = image_cols // 256
        name = tensor_name(projection)
        parts = [self._bytes(name, dtype, shape,
                             ((band * 64 // source_rows + row) * tiles + stripe * selected) * tile_bytes,
                             selected * tile_bytes)
                 for row in range(64 // source_rows)]
        return q4_projection_pool(np.frombuffer(b"".join(parts), np.uint8),
                                  64, image_cols, source_tile_bytes=tile_bytes)

    def bf16_projection(self, projection: Projection) -> np.ndarray:
        """The FLM [N/32,K,32] stream used directly by the BF16 GEMV image."""
        if projection.format != "bf16":
            raise ValueError("expected a BF16 per-layer projection")
        dtype, shape = source_shape(self.plan, projection)
        return np.frombuffer(self._bytes(tensor_name(projection), dtype, shape),
                             bfloat16).copy().reshape(shape)

    def vector(self, name: str, width: int) -> np.ndarray:
        if width <= 0:
            raise ValueError("vector width must be positive")
        return np.frombuffer(self._bytes(name, "BF16", (width,)), bfloat16).copy()

    def embedding(self, token: int) -> np.ndarray:
        """Decode one source token row to FP32.

        E2B/E4B q4nx conversion stores the sqrt(hidden) scaling in its
        quantized row. The 12B BF16 source row is returned as stored; its
        model recipe must apply the checkpoint's embedding scale.
        """
        if not isinstance(token, int) or token < 0 or token >= self.plan.vocab:
            raise ValueError("Gemma 4 token id out of range")
        width = self.plan.hidden
        name = "model.embed_tokens.weight"
        if self.plan.variant == "12B":
            raw = self._bytes(name, "BF16", (self.plan.vocab, width),
                              token * width * 2, width * 2)
            return np.frombuffer(raw, bfloat16).astype(np.float32)
        return self._quantized_row(name, token, width)

    def per_layer_embedding(self, token: int, layer: int) -> np.ndarray:
        """Decode one E2B/E4B token's 256-value per-layer input."""
        if self.plan.variant == "12B" or not isinstance(layer, int) or not 0 <= layer < len(self.plan.layers):
            raise ValueError("per-layer embedding is unavailable for this layer")
        width = len(self.plan.layers) * self.plan.layers[0].per_layer_input_width
        row = self._quantized_row("model.per_layer_token_embd.weight", token, width)
        n = self.plan.layers[0].per_layer_input_width
        return row[layer * n:(layer + 1) * n].copy()

    def _quantized_row(self, name: str, token: int, width: int) -> np.ndarray:
        if not isinstance(token, int) or token < 0 or token >= self.plan.vocab:
            raise ValueError("Gemma 4 token id out of range")
        if width % 32:
            raise ValueError("quantized embedding width must be divisible by 32")
        q = np.frombuffer(self._bytes(name, "I8", (self.plan.vocab, width),
                                      token * width, width), np.int8).astype(np.float32)
        scales = np.frombuffer(self._bytes(name + ".scale", "F32",
                                           (self.plan.vocab, width // 32),
                                           token * width // 32 * 4,
                                           width // 32 * 4), np.float32)
        return q * np.repeat(scales, 32)

    def head_band(self, band: int) -> np.ndarray:
        """One 64-row q4 or 128-row q8 vocabulary band in NPU pool order."""
        rows = 128 if self.plan.variant == "12B" else 64
        if not isinstance(band, int) or band < 0 or band >= self.plan.vocab // rows:
            raise ValueError("Gemma 4 head band out of range")
        cols = self.plan.hidden
        if self.plan.variant == "12B":
            shape = (self.plan.vocab // 16, cols // 256, 4352)
            count = rows // 16 * cols // 256 * 4352
            raw = self._bytes("lm_head.weight", "I8", shape, band * count, count)
            return q8_head_pool(np.frombuffer(raw, np.uint8), rows, cols)
        shape = (self.plan.vocab * cols // (32 * 256), 5120)
        count = rows // 32 * cols // 256 * 5120
        raw = self._bytes("lm_head.weight", "I8", shape, band * count, count)
        return q4_projection_pool(np.frombuffer(raw, np.uint8), rows, cols,
                                  source_tile_bytes=5120)
