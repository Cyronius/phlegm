"""Checked 512-position projected conditioning for the experimental Klein input path."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from ml_dtypes import bfloat16

PROTOCOL = "klein-projected-conditioning-v1"
POSITIONS, WIDTH, PAD_ID = 512, 3072, 151643


@dataclass(frozen=True)
class ProjectedConditioning:
    bits: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray
    source_dtype: str
    sha256: str

    def require_prompt(self, ids: np.ndarray, real_tokens: int) -> None:
        mask = (np.arange(POSITIONS) < real_tokens).astype(np.int64)
        if not np.array_equal(ids, self.input_ids) or not np.array_equal(mask, self.attention_mask):
            raise ValueError("projected conditioning token IDs/mask differ from --prompt")

    def provenance(self) -> dict:
        return {"protocol": PROTOCOL, "sha256": self.sha256, "source_dtype": self.source_dtype,
                "runtime_dtype": "BF16", "positions": POSITIONS, "width": WIDTH}


def load_projected_conditioning(path: Path) -> ProjectedConditioning:
    """Accept a teacher/student artifact; round floats once to the native BF16 boundary."""
    from safetensors import deserialize

    path = Path(path)
    # Read one bounded snapshot: the recorded hash identifies the bytes consumed.
    with path.open("rb") as reader:
        data = reader.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("projected conditioning artifact exceeds 16 MiB")
    # The public Rust decoder validates tensor bounds. Decode BF16 directly: the
    # safetensors.numpy byte-loader does not support BF16 in all supported versions.
    tensors = dict(deserialize(data))
    header_bytes = struct.unpack_from("<Q", data)[0]
    metadata = json.loads(data[8:8 + header_bytes]).get("__metadata__", {})
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError("unsupported projected conditioning protocol")
    required = {"projected_conditioning", "text_ids", "input_ids", "attention_mask"}
    if not required.issubset(tensors):
        raise ValueError("projected conditioning requires conditioning, text IDs, input IDs and mask")
    tensor = tensors["projected_conditioning"]
    dtype = tensor["dtype"]
    if dtype not in {"BF16", "F16", "F32"} or tensor["shape"] != [POSITIONS, WIDTH]:
        raise ValueError("projected conditioning must be BF16/F16/F32 [512, 3072]")
    for name, shape in (("text_ids", [POSITIONS, 4]), ("input_ids", [POSITIONS]),
                        ("attention_mask", [POSITIONS])):
        if tensors[name]["dtype"] != "I64" or tensors[name]["shape"] != shape:
            raise ValueError(f"invalid projected conditioning {name} type/shape")
    dt = {"BF16": bfloat16, "F16": "<f2", "F32": "<f4", "I64": "<i8"}

    def array(name):
        t = tensors[name]
        return np.frombuffer(t["data"], dtype=dt[t["dtype"]]).reshape(t["shape"])

    values, positions, ids, mask = (array(name) for name in
                                   ("projected_conditioning", "text_ids", "input_ids", "attention_mask"))
    expected_positions = np.zeros((POSITIONS, 4), np.int64)
    expected_positions[:, 3] = np.arange(POSITIONS)
    if positions.dtype != np.int64 or not np.array_equal(positions, expected_positions):
        raise ValueError("projected conditioning text positions differ from the 512-position control")
    if ids.dtype != np.int64 or mask.dtype != np.int64 or ids.shape != (POSITIONS,) or mask.shape != (POSITIONS,):
        raise ValueError("projected conditioning input IDs and mask must be int64 [512]")
    real = int(mask.sum())
    if not 0 < real <= POSITIONS or not np.array_equal(mask, (np.arange(POSITIONS) < real).astype(np.int64)):
        raise ValueError("projected conditioning mask must describe a nonempty right-padded sequence")
    if np.any(ids < 0) or not np.all(ids[real:] == PAD_ID):
        raise ValueError("invalid projected conditioning input/padding IDs")
    if not np.isfinite(values).all():
        raise ValueError("projected conditioning contains nonfinite values")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.ascontiguousarray(values.astype(bfloat16))
    if not np.isfinite(rounded).all():
        raise ValueError("projected conditioning overflows the native BF16 boundary")
    return ProjectedConditioning(rounded.view(np.uint16), ids.copy(), mask.copy(), dtype,
                                 hashlib.sha256(data).hexdigest())
