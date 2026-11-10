"""Prepare an isolated, compile-only dense GGUF V0/V1 fit probe (run with python -I).

Build the printed command under Dibs after sourcing target/npu-dev/env.sh.
No downloaded Python or model code is imported or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "open_kernels"))
from gguf_pool import read_gguf
from recipes import dense
from recipes.dense_precision import from_gguf_types, layer_formats
from recipes.load import load_spec
from recipes.spec import ModelSpec


def prepare(spec, tensor_types, out, layer=None):
    data = spec.to_dict()
    data["tensor_quant"] = from_gguf_types(tensor_types, spec.num_layers)
    spec = ModelSpec.from_dict(data)
    if layer is None:
        candidates = [i for i in range(spec.num_layers)
                      if {p for p, f in layer_formats(spec, i).items() if f == "q8"}
                      == {"attn_v", "ffn_down"}]
        if not candidates:
            raise ValueError("GGUF has no V1 layer with exactly q8 attn_v and ffn_down")
        layer = candidates[0]
    os.environ["DENSE_GGUF_LAYER"] = str(layer)
    os.environ["DX_ONECTX"] = "0"
    recipe = dense.recipe(spec)
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "source" / "open_kernels"
    if stage.exists():
        raise FileExistsError(f"use a fresh probe directory: {stage}")
    def ignore(directory, names):
        return [n for n in names if n == "__pycache__" or n.startswith(".staging")
                or (n.startswith("build") and (Path(directory) / n).is_dir())]
    shutil.copytree(ROOT / "open_kernels", stage, ignore=ignore)
    gen_path = stage / "designs/dense/gen_kernels.py"
    module = importlib.util.spec_from_file_location("dense_fit_gen", gen_path)
    gen = importlib.util.module_from_spec(module)
    module.loader.exec_module(gen)
    gen.generate(recipe, gen_path.parent)
    spec_path = out / "spec.json"
    spec_path.write_text(spec.to_json())
    cmd = [sys.executable, str(stage / "build_design.py"),
           str(stage / "designs/dense/dx.py"), str(out / "dx")]
    env = {"OPEN_KERNELS_SPEC": str(spec_path), "DENSE_GGUF_LAYER": str(layer),
           "OFLM_KEEP_FAILED": "1", "DX_ONECTX": "0"}
    (out / "selection.json").write_text(json.dumps({"layer": layer,
        "formats": layer_formats(spec, layer), "spec_hash": spec.spec_hash(),
        "geometry": recipe.geo.__dict__, "layout": recipe.layout.__dict__,
        "env": env, "probe_env": dense.probe_env(), "command": cmd}, indent=2) + "\n")
    print("env " + " ".join(shlex.quote(f"{k}={v}") for k, v in env.items())
          + " " + shlex.join(cmd))
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=ROOT / "open_kernels/recipes/specs/llama32-1b.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int)
    a = ap.parse_args()
    _metadata, tensors, _data_base = read_gguf(str(a.gguf))  # header only, untrusted data
    spec = load_spec(a.spec)
    # Do not silently probe the checked-in geometry against a different model.
    derived = ModelSpec.from_gguf_metadata(_metadata)
    for key in ("family", "hidden", "num_layers", "num_heads", "num_kv_heads", "head_dim", "intermediate"):
        if getattr(spec, key) != getattr(derived, key):
            raise ValueError(f"GGUF/spec geometry mismatch: {key}")
    prepare(spec, {n: t["type"] for n, t in tensors.items()}, a.out.resolve(), a.layer)
    h = hashlib.sha256()
    with a.gguf.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    (a.out / "gguf.json").write_text(json.dumps({"path": str(a.gguf.resolve()),
        "sha256": h.hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
