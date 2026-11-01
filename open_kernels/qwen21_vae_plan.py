"""Checkpoint-validated operation plan for a Qwen Image 2.1 still-image VAE decode.

This is an architecture contract, not an executable NPU decoder.  It records
the first-frame path used by AutoencoderKLQwenImage21._decode and rejects a
checkpoint whose tensors do not match that path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_kernels.image_safetensors import read_header


def plan(model: Path, resolution: int = 512) -> dict:
    vae = model / "vae"
    config = json.loads((vae / "config.json").read_text())
    if resolution % 16 or resolution < 256:
        raise ValueError("Qwen VAE image resolution must be a multiple of 16 and at least 256")
    expected = {
        "decoder_base_dim": 144, "dim_mult": [1, 2, 4, 8, 8],
        "num_res_blocks": 2, "z_dim": 64, "is_residual": True,
        "scale_factor_spatial": 16, "out_channels": 4,
        "temperal_downsample": [False, True, True, True],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported Qwen VAE {key}: {config.get(key)!r}")
    header = read_header(vae / "diffusion_pytorch_model.safetensors")
    operations: list[dict] = []
    used: set[str] = set()

    def tensor(key: str, shape: tuple[int, ...]) -> None:
        item = header.get(key)
        if item is None or tuple(item["shape"]) != shape or item["dtype"] != "F32":
            raise ValueError(f"Qwen VAE checkpoint tensor mismatch: {key}, expected {shape} F32")
        used.add(key)

    def op(kind: str, name: str, h: int, channels: int, **extra: object) -> None:
        operations.append({"kind": kind, "name": name, "height": h,
                           "width": h, "channels": channels, **extra})

    def conv(base: str, h: int, ci: int, co: int, taps: int) -> None:
        tensor(base + ".weight", (co, ci, taps, taps))
        tensor(base + ".bias", (co,))
        op("conv", base, h, co, input_channels=ci, taps=taps)

    def rms(base: str, h: int, channels: int, attention: bool = False) -> None:
        tensor(base + ".gamma", (channels, 1, 1) if attention else (channels, 1, 1, 1))
        op("rmsnorm", base, h, channels,
           physical_channels=(channels + 127) // 128 * 128)

    def resnet(base: str, h: int, ci: int, co: int) -> None:
        if ci != co:
            conv(base + ".conv_shortcut", h, ci, co, 1)
        rms(base + ".norm1", h, ci)
        op("silu", base + ".silu1", h, ci)
        conv(base + ".conv1", h, ci, co, 3)
        rms(base + ".norm2", h, co)
        op("silu", base + ".silu2", h, co)
        conv(base + ".conv2", h, co, co, 3)
        op("add", base + ".residual", h, co)

    h = resolution // 16
    op("latent_denormalize", "latents_mean_std", h, 64)
    conv("post_quant_conv", h, 64, 64, 1)
    conv("decoder.conv_in", h, 64, 1152, 3)
    resnet("decoder.mid_block.resnets.0", h, 1152, 1152)
    attention = "decoder.mid_block.attentions.0"
    rms(attention + ".norm", h, 1152, attention=True)
    conv(attention + ".to_qkv", h, 1152, 3456, 1)
    op("attention", attention, h, 1152, heads=1, tokens=h * h)
    conv(attention + ".proj", h, 1152, 1152, 1)
    op("add", attention + ".residual", h, 1152)
    resnet("decoder.mid_block.resnets.1", h, 1152, 1152)

    widths = [1152, 1152, 1152, 576, 288, 144]
    temporal = list(reversed(config["temperal_downsample"]))
    for stage, (ci, co) in enumerate(zip(widths[:-1], widths[1:])):
        base = f"decoder.up_blocks.{stage}"
        for index in range(config["num_res_blocks"] + 1):
            resnet(f"{base}.resnets.{index}", h, ci if index == 0 else co, co)
        if stage < 4:
            op("nearest2x", base + ".upsampler.resample.0", h * 2, co)
            conv(base + ".upsampler.resample.1", h * 2, co, co, 3)
            # The published first-frame path passes feat_cache to the decoder.
            # Its resampler marks the empty cache "Rep" and skips time_conv.
            # The residual up block still adds DupUp3D(first_chunk=True).
            op("dup_up3d_first_frame", base + ".avg_shortcut", h * 2, co,
               input_channels=ci, temporal_factor=2 if temporal[stage] else 1)
            op("add", base + ".upsample_residual", h * 2, co)
            h *= 2
    rms("decoder.norm_out", h, 144)
    op("silu", "decoder.out_silu", h, 144)
    conv("decoder.conv_out", h, 144, 4, 3)
    op("clamp", "decoder.clamp", h, 4, minimum=-1, maximum=1)

    expected_used = {key for key in header if key.startswith("decoder.") or
                     key.startswith("post_quant_conv.")}
    inactive = sorted(expected_used - used)
    if not all(".upsampler.time_conv." in key for key in inactive):
        raise ValueError(f"unplanned VAE decoder tensors: {inactive}")
    return {"model": "Qwen/Qwen-Image-2.1", "resolution": resolution,
            "scope": "first-frame still-image decoder topology; no NPU execution",
            "operations": operations, "active_checkpoint_tensors": len(used),
            "inactive_first_frame_tensors": inactive}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = plan(args.model, args.resolution)
    body = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(body)
    print(json.dumps({"operations": len(result["operations"]),
                      "active_checkpoint_tensors": result["active_checkpoint_tensors"],
                      "inactive_first_frame_tensors": len(result["inactive_first_frame_tensors"])}))


if __name__ == "__main__":
    main()
