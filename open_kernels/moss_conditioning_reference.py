"""CPU oracle for MOSS timestep conditioning; never used for generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    model = args.model.resolve()
    fixture = args.fixture.resolve()
    index = json.loads((model / "transformer" /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())
    names = ("time_embedding.0.weight", "time_embedding.0.bias",
             "time_embedding.2.weight", "time_embedding.2.bias",
             "shared_block_modulation.weight", "shared_block_modulation.bias",
             "video_head_modulation.weight", "video_head_modulation.bias")
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        selected = [name for name in names if index["weight_map"][name] == shard]
        if not selected:
            continue
        with safe_open(model / "transformer" / shard, framework="pt", device="cpu") as f:
            for name in selected:
                state[name] = f.get_tensor(name)
    raw = np.fromfile(fixture / "feature.bf16", "<u2")
    if raw.size != 512 * 256 + 512:
        raise ValueError("MOSS timestep feature buffer has wrong extent")
    features = torch.from_numpy(raw[:256].reshape(1, 256)).view(torch.bfloat16)
    time = F.linear(features, state["time_embedding.0.weight"],
                    state["time_embedding.0.bias"])
    time = F.silu(time)
    time = F.linear(time, state["time_embedding.2.weight"],
                    state["time_embedding.2.bias"]).float()
    for name, prefix in (("mod", "shared_block_modulation"),
                         ("head", "video_head_modulation")):
        result = F.linear(time.to(torch.bfloat16), state[prefix + ".weight"],
                          state[prefix + ".bias"]).float().numpy()
        if not np.isfinite(result).all():
            raise ValueError(f"MOSS reference {name} contains NaN/Inf")
        (fixture / f"reference-{name}.f32").write_bytes(result.astype("<f4").tobytes())
        print(f"MOSS reference {name}: RMS {np.sqrt(np.mean(result ** 2)):.6g}")


if __name__ == "__main__":
    main()
