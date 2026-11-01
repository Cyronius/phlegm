"""Pack pinned image DiT matrices into the existing dit_gemm B operand format.

This exports weights only. A model needs its own attention, elementwise, VAE,
text-encoder and runtime schedules before it can perform NPU inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import flux3_weight_layout as flux3
import moss_weight_layout as moss
import qwen21_weight_layout as qwen21
from designs.dit_gemm.pack import interleave_swiglu, pack_b
from image_safetensors import read_header

PACK_FORMAT = "bfp16ebs8 pack_b v1"
COLUMN_CHUNK = 1024  # pack_b's tiles are column-major, so chunks concatenate.


class TensorReader:
    def __init__(self, paths: dict[str, Path]):
        self.entries = {}
        for path in set(paths.values()):
            with path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
            for name, entry in read_header(path).items():
                self.entries[name] = (path, 8 + header_size + entry["data_offsets"][0],
                                      tuple(entry["shape"]))
        if set(paths) != set(self.entries):
            raise ValueError("shard index does not match safetensors payload")
        for name, path in paths.items():
            if self.entries[name][0] != path:
                raise ValueError(f"{name}: indexed to a different shard")

    def matrix(self, name: str) -> np.memmap:
        path, offset, shape = self.entries[name]
        if len(shape) != 2:
            raise ValueError(f"{name}: expected a matrix, got {shape}")
        return np.memmap(path, mode="r", dtype=bfloat16, offset=offset, shape=shape)


def source_paths(model: str, model_dir: Path) -> dict[str, Path]:
    if model == "flux3":
        return {name: model_dir / "model.safetensors" for name in flux3.expected_shapes()}
    index = json.loads((model_dir / "transformer" /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())
    return {name: model_dir / "transformer" / shard
            for name, shard in index["weight_map"].items()}


def matrix(reader: TensorReader, slot) -> np.ndarray:
    """Build physical [K, N], padding short input/output widths with zero."""
    out = np.zeros((slot.k, slot.n), dtype=np.float32)
    col = 0
    split_after = getattr(slot, "split_after", None)
    half_pad = (slot.n - slot.logical_n) // 2 if split_after is not None else 0
    for source in slot.sources:
        weight = reader.matrix(source)
        width = weight.shape[0]
        if split_after is None:
            out[:weight.shape[1], col:col + width] = np.asarray(weight, dtype=np.float32).T
        else:
            # MOSS SiTU's 8192-wide halves each occupy three 3072-wide EW
            # elements. This handles both separate text-FFN matrices and the
            # main block's single fused up/gate matrix.
            first = min(width, max(0, split_after - col))
            if first:
                out[:weight.shape[1], col:col + first] = np.asarray(
                    weight[:first], dtype=np.float32).T
            if first < width:
                dest = col + first + half_pad
                out[:weight.shape[1], dest:dest + width - first] = np.asarray(
                    weight[first:], dtype=np.float32).T
        col += width
    if col != (slot.logical_n or slot.n):
        raise ValueError(f"{slot.sources}: column count differs from slot")
    if slot.swiglu:
        out = interleave_swiglu(out)
    return out


def write_slot(path: Path, weight: np.ndarray) -> tuple[int, str]:
    digest = hashlib.sha256()
    nbytes = 0
    with path.open("xb") as stream:
        for column in range(0, weight.shape[1], COLUMN_CHUNK):
            packed = pack_b(weight[:, column:column + COLUMN_CHUNK])
            payload = packed.tobytes()
            stream.write(payload)
            digest.update(payload)
            nbytes += len(payload)
    expected = weight.shape[0] * weight.shape[1] * 9 // 8
    if nbytes != expected:
        raise ValueError(f"{path}: packed length {nbytes} differs from {expected}")
    return nbytes, digest.hexdigest()


def export(model: str, model_dir: Path, out_dir: Path, names: list[str]) -> dict:
    layout = {"flux3": flux3, "qwen21": qwen21, "moss": moss}[model]
    identity = layout.inspect(model_dir)
    slots = layout.gemm_slots()
    selected = names or list(slots)
    if len(selected) != len(set(selected)) or any(name not in slots for name in selected):
        raise ValueError("unknown or repeated GEMM slot")
    reader = TensorReader(source_paths(model, model_dir))
    out_dir.mkdir(parents=True, exist_ok=False)
    records = {}
    for name in selected:
        slot = slots[name]
        weight = matrix(reader, slot)
        path = out_dir / f"{name}.bin"
        nbytes, sha256 = write_slot(path, weight)
        records[name] = {"file": path.name, "k": slot.k, "n": slot.n,
                         "logical_k": slot.logical_k or slot.k,
                         "logical_n": slot.logical_n or slot.n,
                         "swiglu": slot.swiglu, "gather_input": slot.gather_input,
                         "split_after": getattr(slot, "split_after", None),
                         "sources": slot.sources, "bytes": nbytes, "sha256": sha256}
        print(f"{name}: {nbytes} bytes", flush=True)
    manifest = {"format": PACK_FORMAT, "model": identity["model"],
                "source_dir": str(model_dir.resolve()), "full_model": len(selected) == len(slots),
                "slots": records}
    (out_dir / "packed_weights.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("flux3", "qwen21", "moss"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slot", action="append", default=[], help="one GEMM slot; repeat or omit for all")
    args = parser.parse_args()
    export(args.model, args.model_dir, args.out, args.slot)
