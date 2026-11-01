"""Run the published BF16 Qwen Image 2.1 block stack on an NPU fixture.

Each layer uses its own checkpoint tensors. This oracle begins with the native
runner's embedded token rows, so its comparisons isolate the block stack from
text/latent entry error. It reports every layer, including late residual drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from open_kernels.qwen21_block_reference import (HIDDEN, IMAGE, TEXT, from_bf16,
                                                  load, mod_row, read_bf16_f32,
                                                  rms, rope, stats)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    model, fixture = args.model.resolve(), args.fixture.resolve()
    meta = json.loads((fixture / "run.json").read_text())
    x = torch.cat((from_bf16(fixture / "input-text.bf16", (TEXT, HIDDEN)),
                   from_bf16(fixture / "input-image.bf16", (IMAGE, HIDDEN))), 0)
    text_mod = from_bf16(fixture / "text-mod.bf16", (4, HIDDEN))
    image_mod = from_bf16(fixture / "image-mod.bf16", (4, HIDDEN))
    scale1, gate1, scale2, gate2 = (mod_row(text_mod, image_mod, i) for i in range(4))
    split = lambda value: value.reshape(-1, 32, 128).transpose(0, 1).unsqueeze(0)
    for layer in range(meta["layers"]):
        prefix = f"transformer_blocks.{layer}."
        weights = {key: load(model, prefix + name) for key, name in {
            "q": "attn.to_q.weight", "k": "attn.to_k.weight",
            "v": "attn.to_v.weight", "qout": "attn.to_out.0.weight",
            "nq": "attn.norm_q.weight", "nk": "attn.norm_k.weight",
            "gate": "img_mlp.gate_layer.weight", "up": "img_mlp.proj.weight",
            "down": "img_mlp.out.weight"}.items()}
        xmod = F.layer_norm(x, (HIDDEN,), eps=1e-6) * (1 + scale1)
        q = rope(rms(F.linear(xmod, weights["q"]), weights["nq"]))
        k = rope(rms(F.linear(xmod, weights["k"]), weights["nk"]))
        v = F.linear(xmod, weights["v"])
        text_attn = F.scaled_dot_product_attention(split(q[:TEXT]), split(k[:TEXT]),
                                                    split(v[:TEXT]), is_causal=True)
        image_attn = F.scaled_dot_product_attention(split(q[TEXT:]), split(k), split(v))
        attended = torch.cat((text_attn, image_attn), 2).squeeze(0).transpose(0, 1).reshape(
            TEXT + IMAGE, HIDDEN)
        aout = F.linear(attended, weights["qout"])
        x1 = x + gate1.tanh() * aout
        xmod2 = F.layer_norm(x1, (HIDDEN,), eps=1e-6) * (1 + scale2)
        mlp = F.silu(F.linear(xmod2, weights["gate"])) * F.linear(xmod2, weights["up"])
        x = x1 + gate2.tanh() * F.linear(mlp, weights["down"])
        layer_dir = fixture / f"layer-{layer:02d}"
        native = np.concatenate((read_bf16_f32(layer_dir / "text.bf16"),
                                 read_bf16_f32(layer_dir / "image.bf16"))).reshape(TEXT + IMAGE, HIDDEN)
        reference = x.float().numpy()
        (layer_dir / "reference-state.f32").write_bytes(reference.astype("<f4").tobytes())
        print(f"layer {layer:02d}: {stats(native, reference)}", flush=True)


if __name__ == "__main__":
    main()
