"""Export one GPT-OSS-20B layer's checked NPU source inputs from model.q4nx.

The output is an arithmetic-kernel input bundle, not an installable model.
Only explicitly requested experts are exported; each is read separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
from q4nx import Q4NX  # noqa: E402
from recipes.gptoss_source import CONSTANTS, GptOssSource  # noqa: E402


def export(source: GptOssSource, layer: int, out: Path,
           experts: tuple[int, ...] = ()) -> dict:
    """Write a bounded, hash-recorded bundle; leave no partial destination."""
    source._prefix(layer)
    if any(not isinstance(e, int) or e < 0 or e >= 32 for e in experts) or len(set(experts)) != len(experts):
        raise ValueError("experts must be distinct ids in [0,32)")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    files = {}

    def put(relative: str, values: np.ndarray) -> None:
        arr = np.ascontiguousarray(values)
        path = temp / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = arr.tobytes()
        path.write_bytes(data)
        files[relative] = {"dtype": str(arr.dtype), "shape": list(arr.shape),
                           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    try:
        for role, bands in (("q", 128), ("k", 16), ("v", 16), ("o", 96)):
            for band in range(0, bands, 16):
                put(f"attention/{role}-{band:03d}.bin", source.attention(layer, role, band))
        for role in CONSTANTS:
            put(f"constants/{role}.bin", source.constant(layer, role))
        put("constants/router_weight.bin", source.router_weight(layer))
        for expert_id in experts:
            for role in ("gate", "up", "down"):
                put(f"experts/{expert_id:02d}/{role}.bin", source.expert(layer, expert_id, role))
                put(f"experts/{expert_id:02d}/{role}_bias.bin",
                    source.expert_bias(layer, expert_id, role))
                put(f"experts/{expert_id:02d}/{role}_npu.bin",
                    source.expert_input(layer, expert_id, role))
        manifest = {"format": "gptoss-20b-checked-npu-source-v1", "layer": layer,
                    "experts": list(experts), "files": files,
                    "scope": "one-layer source inputs for checked kernels; no model runtime"}
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temp, out)
        return manifest
    except BaseException:
        shutil.rmtree(temp)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="model.q4nx path")
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--experts", type=int, nargs="*", default=[])
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        report = export(GptOssSource(reader), a.layer, a.out, tuple(a.experts))
        print(json.dumps({"layer": report["layer"], "experts": report["experts"],
                          "files": len(report["files"]), "out": str(a.out)}, indent=2))
    finally:
        reader.mm.close()
        reader.f.close()


if __name__ == "__main__":
    main()
