"""Pack the pinned MOSS Image SFT VAE decoder for Phlegm's native NPU layout.

The output is a headless decoder package: packed convolution/GEMM weights,
GroupNorm parameter blocks, and a manifest of their byte slices. No model
inference or network access occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from open_kernels.image_safetensors import read_header
from open_kernels.vae_decoder import MOSS_BLOCK_OUT, pack_weights, plan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensors(path: Path) -> dict[str, np.ndarray]:
    header = read_header(path)
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
    base = 8 + header_size
    wanted = {k: v for k, v in header.items()
              if k.startswith(("decoder.", "bn.", "post_quant_conv."))
              and k != "bn.num_batches_tracked"}
    if not wanted or any(v["dtype"] != "F32" for v in wanted.values()):
        raise ValueError("MOSS VAE decoder tensors must be FP32")
    return {key: np.memmap(path, dtype="<f4", mode="r", offset=base + item["data_offsets"][0],
                           shape=tuple(item["shape"]))
            for key, item in wanted.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint_dir.resolve()
    config_path = source / "config.json"
    weight_path = source / "diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text())
    required = {"_class_name": "AutoencoderKLFlux2", "latent_channels": 32,
                "decoder_block_out_channels": list(MOSS_BLOCK_OUT), "patch_size": [2, 2],
                "norm_num_groups": 32}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"unsupported MOSS VAE {key}: {config.get(key)!r}")
    if args.out.exists():
        raise FileExistsError(args.out)
    sd = tensors(weight_path)
    packed, table, blocks, norms = pack_weights(sd)
    scheduled = plan(512, MOSS_BLOCK_OUT)
    used_weights = {key for op in scheduled.ops for kind, key in op["args"] if kind == "w"}
    used_norms = {key for op in scheduled.ops for kind, key in op["args"] if kind == "gn"}
    if used_weights != set(table) or used_norms != set(norms):
        raise ValueError(f"VAE schedule/weight mismatch: weights={used_weights ^ set(table)}, "
                         f"norms={used_norms ^ set(norms)}")
    if any(offset % 4096 or offset + length > packed.nbytes
           for offset, length in table.values()):
        raise ValueError("unaligned or out-of-range packed VAE weight slice")
    args.out.mkdir(parents=True)
    weight_out = args.out / "weights.bin"
    norm_out = args.out / "gn_blocks.bf16"
    packed.tofile(weight_out)
    blocks.tofile(norm_out)
    manifest = {
        "model": "OpenMOSS-Team/MOSS-Image-SFT",
        "profile": list(MOSS_BLOCK_OUT),
        "groupnorm_element": 3072,
        "attention_input_columns": 512,
        "attention_bias_column": 384,
        "attention_padded_heads": 4,
        "source": {"config_sha256": sha256(config_path),
                   "weights_sha256": sha256(weight_path)},
        "outputs": {"weights.bin": {"bytes": weight_out.stat().st_size,
                                    "sha256": sha256(weight_out)},
                    "gn_blocks.bf16": {"bytes": norm_out.stat().st_size,
                                        "sha256": sha256(norm_out)}},
        "weight_slices": {key: {"offset": offset, "bytes": length}
                          for key, (offset, length) in table.items()},
        "groupnorm_blocks": norms,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"MOSS VAE packed: {len(table)} weights, {len(norms)} norms, "
          f"{packed.nbytes:,} weight bytes -> {args.out}")


if __name__ == "__main__":
    main()
