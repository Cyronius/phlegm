"""CPU oracle for MOSS image entry/exit projections; never used for generation."""

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
    meta = json.loads((fixture / "fixture.json").read_text())
    index = json.loads((model / "transformer" /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())
    if meta["stage"] == "image entry":
        names = ("video_embedder.weight", "video_embedder.bias")
    elif meta["stage"] == "image exit":
        names = ("final_norm_video.weight", "final_linear_video.weight")
    else:
        raise ValueError("unknown MOSS image edge fixture stage")
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        selected = [name for name in names if index["weight_map"][name] == shard]
        if not selected:
            continue
        with safe_open(model / "transformer" / shard, framework="pt", device="cpu") as f:
            for name in selected:
                state[name] = f.get_tensor(name)
    tokens = meta["tokens"]
    if meta["stage"] == "image entry":
        raw = np.fromfile(fixture / "latents.bf16", "<u2")
        if raw.size != tokens * 128 + 512:
            raise ValueError("MOSS latent fixture has wrong extent")
        lat = torch.from_numpy(raw[:tokens * 128].reshape(tokens, 128)).view(torch.bfloat16)
        result = F.linear(lat, state["video_embedder.weight"],
                          state["video_embedder.bias"]).float().numpy()
        path = fixture / "reference-image-x.f32"
    else:
        raw = np.fromfile(meta["joint_hidden"], "<u2")
        if raw.size < tokens * 3072:
            raise ValueError("MOSS joint hidden fixture has wrong extent")
        x = torch.from_numpy(raw[:tokens * 3072].reshape(tokens, 3072)).view(torch.bfloat16)
        head_raw = np.fromfile(meta["head_modulation"], "<u2")
        if head_raw.size != 512 * 2 * 3072:
            raise ValueError("MOSS head modulation fixture has wrong extent")
        head = torch.from_numpy(head_raw[:2 * 3072].reshape(2, 3072)).view(torch.bfloat16)
        weight = state["final_norm_video.weight"]
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        y = y.to(torch.bfloat16) * weight
        y = y * (1 + head[1]) + head[0]
        result = F.linear(y, state["final_linear_video.weight"]).float().numpy()
        path = fixture / "reference-velocity.f32"
    if not np.isfinite(result).all():
        raise ValueError("MOSS image edge reference returned NaN/Inf")
    path.write_bytes(result.astype("<f4").tobytes())
    print(f"MOSS reference {meta['stage']}: {tokens} rows, "
          f"RMS {np.sqrt(np.mean(result ** 2)):.6g}")


if __name__ == "__main__":
    main()
