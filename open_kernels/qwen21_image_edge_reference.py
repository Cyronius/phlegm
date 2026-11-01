"""Independent BF16 checkpoint oracle for Qwen Image 2.1 NPU edge fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from open_kernels.qwen21_block_reference import (HIDDEN, IMAGE, TEXT, from_bf16,
                                                  load, read_bf16_f32, stats)


def compare(out: Path, name: str, reference: torch.Tensor) -> None:
    expected = reference.float().numpy().astype("<f4")
    (out / f"reference-{name}.f32").write_bytes(expected.tobytes())
    native = read_bf16_f32(out / f"native-{name}.bf16").reshape(expected.shape)
    print(f"{name}: {stats(native, expected)}", flush=True)


@torch.no_grad()
def entry(model: Path, out: Path) -> None:
    meta = json.loads((out / "fixture.json").read_text())
    time = float(meta["timestep"])
    text = from_bf16(out / "input-text.bf16", (TEXT, HIDDEN))
    latent = from_bf16(out / "input-latent.bf16", (IMAGE, 64))
    weight = load(model, "txt_in.text_norm.weight")
    rrms = torch.rsqrt(text.float().square().mean(-1, keepdim=True) + 1e-6)
    norm = (text.float() * rrms * (weight.float() + 1)).to(torch.bfloat16)
    x = F.linear(norm, load(model, "txt_in.in_layer.weight"))
    x = F.gelu(x, approximate="tanh")
    projected_text = F.linear(x, load(model, "txt_in.out_layer.weight"))
    compare(out, "text", projected_text)
    compare(out, "image", F.linear(latent, load(model, "img_in.weight")))

    frequencies = torch.exp(-np.log(10000.0) * torch.arange(128, dtype=torch.float32) / 128)
    phase = 1000 * torch.tensor([time, 0.0], dtype=torch.float32)[:, None] * frequencies[None]
    encoded = torch.cat((torch.cos(phase), torch.sin(phase)), -1).to(torch.bfloat16)
    temb = F.linear(encoded, load(model, "time_text_embed.timestep_embedder.linear_1.weight"))
    temb = F.silu(temb)
    temb = F.linear(temb, load(model, "time_text_embed.timestep_embedder.linear_2.weight"))
    activated = F.silu(temb)
    modulation = F.linear(activated, load(model, "modulation.1.weight"))
    final_scale = F.linear(activated, load(model, "norm_out.linear.weight"))
    compare(out, "image-mod", modulation[0].reshape(4, HIDDEN))
    compare(out, "text-mod", modulation[1].reshape(4, HIDDEN))
    compare(out, "image-final-scale", final_scale[0])
    compare(out, "text-final-scale", final_scale[1])


@torch.no_grad()
def final(model: Path, out: Path) -> None:
    image = from_bf16(out / "input-image.bf16", (IMAGE, HIDDEN))
    scale = from_bf16(out / "input-scale.bf16", (HIDDEN,))
    x = F.layer_norm(image, (HIDDEN,), eps=1e-6) * (1 + scale)
    velocity = F.linear(x, load(model, "proj_out.weight"))
    compare(out, "velocity", velocity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    out = args.fixture.resolve()
    stage = json.loads((out / "fixture.json").read_text())["stage"]
    entry(args.model.resolve(), out) if stage == "entry" else final(args.model.resolve(), out)


if __name__ == "__main__":
    main()
