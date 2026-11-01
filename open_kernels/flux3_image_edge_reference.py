"""Published BF16 FLUX 3 projection oracle for native entry/exit fixtures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    out = args.fixture.resolve()
    meta = json.loads((out / "fixture.json").read_text())
    stage = meta["stage"]
    rows = meta["live_rows"]
    width = (128 if stage == "image-entry" else
             20480 if stage == "text-entry" else 3072)
    raw = np.fromfile(out / "input.bf16", "<u2")[:rows * width]
    x = torch.from_numpy(raw.reshape(rows, width)).view(torch.bfloat16)
    with safe_open(args.model.resolve() / "model.safetensors",
                   framework="pt", device="cpu") as source:
        if stage == "image-entry":
            weight = source.get_tensor("emb_in.image.weight")
        elif stage == "text-entry":
            weight = source.get_tensor("txt_in.weight")
        else:
            weight = source.get_tensor("final_layer.image.linear.weight")
    if stage == "final-exit":
        reference = args.reference.resolve() / "src/flux3/dit.py"
        spec = importlib.util.spec_from_file_location("flux3_edge_dit", reference)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        m = np.fromfile(out / "modulation.bf16", "<u2").reshape(2, 3072)
        mod = torch.from_numpy(m).view(torch.bfloat16)
        x = module.layernorm_modulation(x, mod[1], mod[0])
    expected = torch.nn.functional.linear(x, weight).float().numpy()
    if not np.isfinite(expected).all():
        raise ValueError("FLUX 3 reference projection returned NaN/Inf")
    (out / "reference-output.f32").write_bytes(expected.astype("<f4").tobytes())
    print(f"published FLUX 3 {stage}: {rows} rows, RMS "
          f"{np.sqrt(np.mean(expected*expected)):.6g}")


if __name__ == "__main__":
    main()
