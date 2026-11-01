"""Small, dependency-free safetensors header reader for image model exporters."""

from __future__ import annotations

import json
import struct
from pathlib import Path


def read_header(path: Path) -> dict[str, dict]:
    """Read tensor descriptors and reject missing, overlapping or truncated payloads."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > 64 * 1024 * 1024 or header_size + 8 > size:
            raise ValueError(f"{path}: invalid safetensors header length")
        header = json.loads(stream.read(header_size))
    ranges = sorted(tuple(item["data_offsets"]) for name, item in header.items()
                    if name != "__metadata__")
    end = 0
    for start, stop in ranges:
        if start != end or stop < start:
            raise ValueError(f"{path}: overlapping or missing tensor payload")
        end = stop
    if end + 8 + header_size != size:
        raise ValueError(f"{path}: tensor payload length differs from file size")
    return {name: item for name, item in header.items() if name != "__metadata__"}
