"""Pack the Qwen Image 2.1 first-frame VAE decoder's active checkpoint tensors.

The output is a set of kernel-ready convolution weight streams and padded
RMSNorm gamma vectors, plus a manifest. It is not a runnable decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.designs.dit_conv.conv_pack import pack_conv
from open_kernels.designs.dit_conv.dit_conv import resolve_spec
from open_kernels.image_safetensors import read_header
from open_kernels.qwen21_vae_plan import plan


def tensor(path: Path, header: dict, name: str) -> np.ndarray:
    item = header[name]
    if item["dtype"] != "F32":
        raise ValueError(f"{name} is not FP32")
    with path.open("rb") as stream:
        data = 8 + struct.unpack("<Q", stream.read(8))[0]
    return np.memmap(path, mode="r", dtype="<f4",
                     offset=data + item["data_offsets"][0], shape=tuple(item["shape"]))


def physical(channels: int, input_side: bool) -> int:
    tile = 64 if input_side and channels == 64 else 128
    return (channels + tile - 1) // tile * tile


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(model: Path, out: Path, limit: int | None = None) -> dict:
    topology = plan(model)
    file = model / "vae/diffusion_pytorch_model.safetensors"
    header = read_header(file)
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    wanted = [op for op in topology["operations"] if op["kind"] in ("conv", "rmsnorm")]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        wanted = wanted[:limit]
    for index, op in enumerate(wanted):
        name = op["name"]
        if op["kind"] == "conv":
            ci, co, taps = op["input_channels"], op["channels"], op["taps"]
            ci_physical, co_physical = physical(ci, True), physical(co, False)
            up = ".upsampler.resample.1" in name
            kernel_spec = {"H": op["height"] // (2 if up else 1),
                           "W": op["width"] // (2 if up else 1),
                           "Cin": ci_physical, "Cout": co_physical, "up": up}
            resolved = resolve_spec(kernel_spec, taps * taps)
            w = tensor(file, header, name + ".weight")
            b = tensor(file, header, name + ".bias")
            latent_affine = False
            if name == "post_quant_conv":
                config = json.loads((model / "vae/config.json").read_text())
                mean = np.asarray(config["latents_mean"], np.float32)
                std = np.asarray(config["latents_std"], np.float32)
                if mean.shape != (64,) or std.shape != (64,) or not np.all(std > 0):
                    raise ValueError("unsupported Qwen VAE latent affine parameters")
                original = np.asarray(w, np.float32)
                b = np.asarray(b, np.float32) + original[:, :, 0, 0] @ mean
                w = original * std[None, :, None, None]
                latent_affine = True
            blob = pack_conv(w, b, taps=taps * taps, up=up, cin=ci_physical)
            expected = resolved["sizes"]["W"] * 9
            if blob.nbytes != expected:
                raise ValueError(f"packed size mismatch for {name}: {blob.nbytes} != {expected}")
            dtype = "dit_conv_v8bfp16ebs8"
            meta = {"input_channels": ci, "output_channels": co,
                    "physical_input_channels": ci_physical,
                    "physical_output_channels": co_physical, "taps": taps,
                    "kernel_spec": kernel_spec,
                    "latents_mean_std_fused": latent_affine}
        else:
            co = op["channels"]
            co_physical = physical(co, False)
            gamma = np.asarray(tensor(file, header, name + ".gamma"), np.float32).reshape(-1)
            padded = np.zeros(co_physical, bfloat16)
            padded[:co] = (gamma * np.sqrt(co / co_physical)).astype(bfloat16)
            blob = padded.view(np.uint8)
            dtype = "bf16_rms_gamma_padded"
            meta = {"channels": co, "physical_channels": co_physical,
                    "scale": "sqrt(channels / physical_channels)"}
        target = out / f"{index:03d}.bin"
        blob.tofile(target)
        records[name] = {"file": target.name, "kind": op["kind"], "format": dtype,
                         "bytes": target.stat().st_size,
                         "sha256": digest(target), **meta}
        print(f"{index + 1}/{len(wanted)} {name}: {target.stat().st_size} bytes", flush=True)
    result = {"model": topology["model"], "scope": "first-frame decoder parameters only",
              "resolution": topology["resolution"], "complete": limit is None,
              "records": records}
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = pack(args.model, args.out, args.limit)
    print(json.dumps({"parameters": len(result["records"]), "complete": result["complete"]}))


if __name__ == "__main__":
    main()
