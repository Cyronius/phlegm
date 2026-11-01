"""CPU oracle for a native MOSS joint-block fixture; never used for generation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def bf16_file(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"{path}: BF16 element count differs")
    return torch.from_numpy(raw.reshape(shape)).view(torch.bfloat16)


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
    start_layer = meta["layer"]
    count = meta.get("layers", 1)
    index = json.loads((model / "transformer" /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())
    rows = meta["valid_tokens"]
    x = bf16_file(fixture / "input-x.bf16", (meta["tokens"], 3072))[:rows][None]
    modulation = torch.from_numpy(np.fromfile(fixture / "modulation.f32", "<f4")
                                  .reshape(1, 6, 3072))
    grid = meta["resolution"] // 16
    rope = module._RoPE(128, 10000.0)(grid, grid, meta["text_length"],
                                        torch.device("cpu")).to(torch.bfloat16)
    for layer in range(start_layer, start_layer + count):
        if layer == meta.get("skip_layer"):
            print(f"reference block {layer} skipped", flush=True)
            continue
        prefix = f"layers.{layer}."
        state = {}
        for shard in sorted(set(index["weight_map"].values())):
            names = [name for name, selected in index["weight_map"].items()
                     if selected == shard and name.startswith(prefix)]
            if not names:
                continue
            with safe_open(model / "transformer" / shard, framework="pt", device="cpu") as f:
                for name in names:
                    state[name[len(prefix):]] = f.get_tensor(name)
        block = module._Block(3072, 8192, 24).to(dtype=torch.bfloat16)
        block.load_state_dict(state, strict=True)
        block.eval()
        x = block(x, rope, modulation)
        print(f"reference block {layer} done", flush=True)
    result = x.float().numpy()
    if not np.isfinite(result).all():
        raise ValueError("MOSS reference joint block returned NaN/Inf")
    (fixture / "reference-x.f32").write_bytes(result.astype("<f4").tobytes())
    print(f"MOSS reference layers {start_layer}..{start_layer + count - 1}: {rows} rows, "
          f"RMS {np.sqrt(np.mean(result ** 2)):.6g} -> {fixture / 'reference-x.f32'}")


if __name__ == "__main__":
    main()
