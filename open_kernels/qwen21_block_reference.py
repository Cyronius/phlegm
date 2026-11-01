"""Independent BF16 oracle for a Qwen Image 2.1 native block fixture.

Implements the published diffusers transformer block equations directly from
the checkpoint, including its three-axis RoPE and segmented block-causal
attention. It reads the fixture's exact BF16 inputs and modulation rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from open_kernels.qwen21_weight_layout import HIDDEN

TEXT, IMAGE = 15, 1024


def load(model: Path, name: str) -> torch.Tensor:
    index = json.loads((model / "transformer" /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())
    path = model / "transformer" / index["weight_map"][name]
    with safe_open(path, framework="pt", device="cpu") as source:
        value = source.get_tensor(name)
    if value.dtype != torch.bfloat16:
        raise ValueError(f"{name}: expected BF16")
    return value


def from_bf16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    value = np.fromfile(path, dtype="<u2").copy()
    if value.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected {shape}")
    return torch.from_numpy(value).view(torch.bfloat16).reshape(shape)


def read_bf16_f32(path: Path) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2").astype("<u4")
    return np.left_shift(words, 16).view("<f4")


def rope(x: torch.Tensor) -> torch.Tensor:
    """Published axes [16,56,56], text all axes advancing, image centred."""
    n = x.shape[0]
    if n != TEXT + IMAGE:
        raise ValueError("reference is for the 15+1024 token fixture")
    pos = torch.arange(TEXT, dtype=torch.float32)
    spatial = torch.arange(IMAGE)
    frame = torch.cat((pos, torch.full((IMAGE,), float(TEXT))))
    height = torch.cat((pos, (spatial // 32 - 16).float()))
    width = torch.cat((pos, (spatial % 32 - 16).float()))
    freqs = []
    for coord, dim in zip((frame, height, width), (16, 56, 56)):
        inv = 10000.0 ** (-torch.arange(0, dim, 2).float() / dim)
        freqs.append(torch.polar(torch.ones_like(coord[:, None] * inv[None]),
                                  coord[:, None] * inv[None]))
    phase = torch.cat(freqs, -1)[:, None, :]
    complex_x = torch.view_as_complex(x.float().reshape(n, 32, 64, 2))
    return torch.view_as_real(complex_x * phase).reshape(n, HIDDEN).to(x.dtype)


def rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    heads = x.reshape(-1, 32, 128).float()
    normalized = heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + 1e-6)
    return (normalized * weight.float()).to(x.dtype).reshape(x.shape)


def mod_row(text_mod: torch.Tensor, image_mod: torch.Tensor, index: int) -> torch.Tensor:
    return torch.cat((text_mod[index].expand(TEXT, -1),
                      image_mod[index].expand(IMAGE, -1)), 0)


def stats(native: np.ndarray, reference: np.ndarray) -> str:
    a, b = native.astype(np.float64).reshape(-1), reference.astype(np.float64).reshape(-1)
    return (f"relative L2 {np.linalg.norm(a-b)/np.linalg.norm(b):.6g}, "
            f"cosine {np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)):.9g}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    meta = json.loads((fixture / "fixture.json").read_text())
    layer = meta["layer"]
    x = torch.cat((from_bf16(fixture / "input-text.bf16", (TEXT, HIDDEN)),
                   from_bf16(fixture / "input-image.bf16", (IMAGE, HIDDEN))), 0)
    text_mod = from_bf16(fixture / "text-mod.bf16", (4, HIDDEN))
    image_mod = from_bf16(fixture / "image-mod.bf16", (4, HIDDEN))
    prefix = f"transformer_blocks.{layer}."
    names = {key: prefix + value for key, value in {
        "q": "attn.to_q.weight", "k": "attn.to_k.weight", "v": "attn.to_v.weight",
        "qout": "attn.to_out.0.weight", "nq": "attn.norm_q.weight",
        "nk": "attn.norm_k.weight", "gate": "img_mlp.gate_layer.weight",
        "up": "img_mlp.proj.weight", "down": "img_mlp.out.weight"}.items()}
    weights = {key: load(args.model.resolve(), name) for key, name in names.items()}
    norm1 = F.layer_norm(x, (HIDDEN,), eps=1e-6)
    xmod1 = norm1 * (1 + mod_row(text_mod, image_mod, 0))
    q = rope(rms(F.linear(xmod1, weights["q"]), weights["nq"]))
    k = rope(rms(F.linear(xmod1, weights["k"]), weights["nk"]))
    v = F.linear(xmod1, weights["v"])
    split = lambda value: value.reshape(-1, 32, 128).transpose(0, 1).unsqueeze(0)
    text_attn = F.scaled_dot_product_attention(split(q[:TEXT]), split(k[:TEXT]),
                                                split(v[:TEXT]), is_causal=True)
    image_attn = F.scaled_dot_product_attention(split(q[TEXT:]), split(k), split(v))
    attention = torch.cat((text_attn, image_attn), 2).squeeze(0).transpose(0, 1).reshape(
        TEXT + IMAGE, HIDDEN)
    aout = F.linear(attention, weights["qout"])
    x1 = x + mod_row(text_mod, image_mod, 1).tanh() * aout
    norm2 = F.layer_norm(x1, (HIDDEN,), eps=1e-6)
    xmod2 = norm2 * (1 + mod_row(text_mod, image_mod, 2))
    mlp = F.silu(F.linear(xmod2, weights["gate"])) * F.linear(xmod2, weights["up"])
    mout = F.linear(mlp, weights["down"])
    result = x1 + mod_row(text_mod, image_mod, 3).tanh() * mout
    for name, value in (("q", q), ("k", k), ("attention", attention), ("x", result)):
        ref = value.float().numpy().astype("<f4")
        (fixture / f"reference-{name}.f32").write_bytes(ref.tobytes())
        native_name = "native-image.bf16" if name == "x" else f"native-{name}.bf16"
        if name == "x":
            native = np.concatenate((read_bf16_f32(fixture / "native-text.bf16"),
                                     read_bf16_f32(fixture / native_name)))
        else:
            native = read_bf16_f32(fixture / native_name)
        native = native.reshape(ref.shape)
        print(f"{name}: {stats(native.astype(np.float32), ref)}", flush=True)


if __name__ == "__main__":
    main()
