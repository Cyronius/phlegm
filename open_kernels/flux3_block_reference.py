"""Published FLUX 3 BF16 Block oracle for the native image-mode fixture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    model, fixture = args.model.resolve(), args.fixture.resolve()
    meta = json.loads((fixture / "fixture.json").read_text())
    reference = args.reference.resolve() / "src/flux3/dit.py"
    spec = importlib.util.spec_from_file_location("flux3_pinned_dit", reference)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    first, count = meta["layer"], meta.get("layers", 1)
    tokens, valid = meta["tokens"], meta.get("valid_tokens", meta["tokens"])
    grid = meta["resolution"] // 16
    mode = meta.get("mode", "image")
    shared = meta.get("stage") == "shared"
    x = np.fromfile(fixture / "input-x.bf16", "<u2").reshape(tokens, module.HIDDEN)[:valid]
    x = torch.from_numpy(x).view(torch.bfloat16)
    if shared:
        n_text = meta["text_length"]
        pieces = []
        for name, mode_rows in (("text", n_text), ("image", valid - n_text)):
            raw = np.fromfile(fixture / f"modulation_{name}.bf16", "<u2")
            value = torch.from_numpy(raw.reshape(3, module.HIDDEN)).view(torch.bfloat16)
            pieces.append(value[:, None, :].expand(3, mode_rows, module.HIDDEN))
        mod = torch.cat(pieces, dim=1)
    else:
        m = np.fromfile(fixture / "modulation.bf16", "<u2").reshape(3, module.HIDDEN)
        mod = torch.from_numpy(m).view(torch.bfloat16)
    positions = torch.arange(valid, dtype=torch.float32)
    zero = torch.zeros_like(positions)
    if shared:
        n_text = meta["text_length"]
        image_pos = positions - n_text
        ids = torch.stack((zero, torch.where(positions < n_text, zero,
                                             (image_pos // grid) * 0.5),
                           torch.where(positions < n_text, zero,
                                       (image_pos % grid) * 0.5),
                           torch.where(positions < n_text, positions, zero)), -1)
    else:
        ids = (torch.stack((zero, zero, zero, positions), -1) if mode == "text" else
               torch.stack((zero, (positions // grid) * 0.5,
                            (positions % grid) * 0.5, zero), -1))
    freqs = module.rope_embed(ids[None])
    for layer in range(first, first + count):
        prefix = (f"single_blocks.{layer}." if shared else
                  f"content_mode_blocks.image.{layer}." if mode == "image" else
                  f"txt_mode_blocks.{layer}.")
        state = {}
        with safe_open(model / "model.safetensors", framework="pt", device="cpu") as source:
            for name in source.keys():
                if name.startswith(prefix):
                    state[name[len(prefix):]] = source.get_tensor(name)
        block = module.Block(module.HIDDEN, module.HEADS, module.MLP_RATIO).to(torch.bfloat16)
        block.load_state_dict(state, strict=True)
        block.eval()
        previous = x
        x = block(x, freqs, [(0, valid)], tuple(mod))
        print(f"published FLUX 3 block {layer} done", flush=True)
    y = x
    x_mod = module.layernorm_modulation(previous, mod[1], mod[0])
    split = lambda t: t.reshape(1, valid, module.HEADS, -1).transpose(1, 2)
    q, k = block.norm(split(block.q_proj(x_mod)),
                      split(block.k_proj(x_mod)), freqs)
    v = split(block.v_proj(x_mod))
    attention = module.packed_attention(q, k, v.to(q.dtype), [(0, valid)])
    attn_out = block.attn_out(attention)
    gate_in, up = block.mlp_in(x_mod).chunk(2, dim=-1)
    mlp_out = block.mlp_out(module.F.silu(gate_in) * up)
    combined = attn_out + mlp_out
    traced = previous + mod[2] * combined
    if not torch.equal(traced, y):
        raise ValueError("FLUX 3 trace differs from pinned Block.forward")
    for name, value in {"qkv": torch.cat((q, k, v), 1).transpose(1, 2).reshape(valid, -1),
                        "attention": attention, "attn_out": attn_out,
                        "mlp_out": mlp_out, "sum": combined}.items():
        (fixture / f"reference-{name}.f32").write_bytes(
            value.float().numpy().astype("<f4").tobytes())
    result = y.float().numpy()
    if not np.isfinite(result).all():
        raise ValueError("FLUX 3 reference block returned NaN/Inf")
    (fixture / "reference-x.f32").write_bytes(result.astype("<f4").tobytes())
    print(f"published FLUX 3 {mode} blocks {first}..{first+count-1}: {valid} rows, RMS "
          f"{np.sqrt(np.mean(result*result)):.6g}")


if __name__ == "__main__":
    main()
