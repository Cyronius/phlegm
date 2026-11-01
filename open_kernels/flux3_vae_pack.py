"""Pack the pinned FLUX 3 FP32 autoencoder decoder for Phlegm's native VAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from open_kernels.image_safetensors import read_header
from open_kernels.vae_decoder import BLOCK_OUT, pack_weights, plan


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decoder_name(name: str) -> str:
    if name.startswith("decoder.post_quant_conv."):
        return "post_quant_conv." + name.split("decoder.post_quant_conv.", 1)[1]
    if name.startswith("decoder.norm_out."):
        return "decoder.conv_norm_out." + name.split("decoder.norm_out.", 1)[1]
    for old, new in (("decoder.mid.block_1.", "decoder.mid_block.resnets.0."),
                     ("decoder.mid.block_2.", "decoder.mid_block.resnets.1.")):
        if name.startswith(old):
            return new + name[len(old):].replace(".nin_shortcut.", ".conv_shortcut.")
    if name.startswith("decoder.mid.attn_1."):
        tail = name[len("decoder.mid.attn_1."):]
        for old, new in (("norm.", "group_norm."), ("q.", "to_q."),
                         ("k.", "to_k."), ("v.", "to_v."),
                         ("proj_out.", "to_out.0.")):
            if tail.startswith(old):
                return "decoder.mid_block.attentions.0." + new + tail[len(old):]
        raise ValueError(f"unknown FLUX 3 VAE attention tensor: {name}")
    for level in range(4):
        prefix = f"decoder.up.{level}."
        if name.startswith(prefix):
            tail = name[len(prefix):]
            target = f"decoder.up_blocks.{3 - level}."
            if tail.startswith("block."):
                return target + tail.replace("block.", "resnets.", 1).replace(
                    ".nin_shortcut.", ".conv_shortcut.")
            if tail.startswith("upsample."):
                return target + tail.replace("upsample.", "upsamplers.0.", 1)
            raise ValueError(f"unknown FLUX 3 VAE up block tensor: {name}")
    if name.startswith(("decoder.", "bn.")):
        return name
    raise ValueError(f"unexpected FLUX 3 VAE decoder key: {name}")


def tensors(path: Path) -> dict[str, np.ndarray]:
    header = read_header(path)
    with path.open("rb") as stream:
        base = 8 + struct.unpack("<Q", stream.read(8))[0]
    wanted = {k: v for k, v in header.items()
              if k.startswith(("decoder.", "bn.")) and k != "bn.num_batches_tracked"}
    if any(v["dtype"] != "F32" for v in wanted.values()):
        raise ValueError("FLUX 3 VAE decoder tensors must be FP32")
    out = {}
    for key, entry in wanted.items():
        name = decoder_name(key)
        if name in out:
            raise ValueError(f"duplicate FLUX 3 VAE mapped key {name}")
        shape = tuple(entry["shape"])
        value = np.memmap(path, dtype="<f4", mode="r",
                          offset=base + entry["data_offsets"][0], shape=shape)
        if name.startswith("decoder.mid_block.attentions.0.") and shape[-2:] == (1, 1):
            value = value.reshape(shape[:2])
        out[name] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, out = args.model.resolve(), args.out.resolve()
    source = model / "ae.safetensors"
    if out.exists():
        raise FileExistsError(out)
    sd = tensors(source)
    packed, table, blocks, norms = pack_weights(sd)
    scheduled = plan(512, BLOCK_OUT)
    used_weights = {name for op in scheduled.ops for kind, name in op["args"] if kind == "w"}
    used_norms = {name for op in scheduled.ops for kind, name in op["args"] if kind == "gn"}
    if used_weights != set(table) or used_norms != set(norms):
        raise ValueError(f"FLUX 3 VAE schedule mismatch: "
                         f"weights={used_weights ^ set(table)}, norms={used_norms ^ set(norms)}")
    out.mkdir(parents=True)
    weights = out / "weights.bin"
    gnorms = out / "gn_blocks.bf16"
    packed.tofile(weights)
    blocks.tofile(gnorms)
    manifest = {
        "model": "black-forest-labs/flux-3-image-klein",
        "profile": list(BLOCK_OUT), "groupnorm_element": 4096,
        "attention_input_columns": 1024, "attention_bias_column": 512,
        "attention_padded_heads": 4,
        "source": {"weights_sha256": sha256(source)},
        "outputs": {"weights.bin": {"bytes": weights.stat().st_size,
                                       "sha256": sha256(weights)},
                    "gn_blocks.bf16": {"bytes": gnorms.stat().st_size,
                                        "sha256": sha256(gnorms)}},
        "weight_slices": {key: {"offset": offset, "bytes": size}
                          for key, (offset, size) in table.items()},
        "groupnorm_blocks": norms,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"FLUX 3 VAE packed: {len(table)} weights, {len(norms)} norms, "
          f"{packed.nbytes:,} bytes -> {out}")


if __name__ == "__main__":
    main()
