"""CPU oracle for MOSS text refinement; never used for native generation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    model = args.model.resolve()
    fixture = args.fixture.resolve()
    meta = json.loads((fixture / "fixture.json").read_text())
    source = model / "pipeline_moss_image.py"
    spec = importlib.util.spec_from_file_location("moss_reference_pipeline", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    projection = module._TextProjection(4096, 3072, 8192, 24).to(dtype=torch.bfloat16)
    index = json.loads((model / "transformer" /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        names = [name for name, selected in index["weight_map"].items()
                 if selected == shard and name.startswith("text_embedder.")]
        if not names:
            continue
        with safe_open(model / "transformer" / shard, framework="pt", device="cpu") as f:
            for name in names:
                state[name[len("text_embedder."):]] = f.get_tensor(name)
    projection.load_state_dict(state, strict=True)
    projection.eval()
    raw = np.fromfile(fixture / "context.bf16", "<u2")
    if raw.size != meta["tokens"] * 4096:
        raise ValueError("MOSS text context has wrong extent")
    context = torch.from_numpy(raw.reshape(meta["tokens"], 4096)).view(torch.bfloat16)
    context = context[:meta["text_length"]][None]
    rope = module._RoPE(128, 10000.0)(1, 1, meta["text_length"],
                                        torch.device("cpu")).to(torch.bfloat16)[:, 1:]
    result = projection(context, rope).float().numpy()
    if not np.isfinite(result).all():
        raise ValueError("MOSS text refinement reference returned NaN/Inf")
    (fixture / "reference-text.f32").write_bytes(result.astype("<f4").tobytes())
    print(f"MOSS text refinement reference: {meta['text_length']} rows, "
          f"RMS {np.sqrt(np.mean(result ** 2)):.6g}")


if __name__ == "__main__":
    main()
