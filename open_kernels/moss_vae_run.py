"""Prepare/check a standalone MOSS VAE decoder pass for Phlegm's C++ XRT harness.

Prepare creates fixed random latents and a complete native kernel schedule.
Run `run_kernel run.cfg` under Dibs on the NPU, then call check. This is a
decoder validation fixture, not an image-generation API or DiT fallback.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.vae_decoder import BLOCK, MOSS_BLOCK_OUT, plan


def write_png(path: Path, rgba: np.ndarray) -> None:
    height, width, channels = rgba.shape
    if channels != 4 or rgba.dtype != np.uint8:
        raise ValueError("PNG writer needs uint8 RGBA pixels")

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    scanlines = b"".join(b"\0" + rgba[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(scanlines, 6))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def prepare(args, *, profile=MOSS_BLOCK_OUT,
            model="OpenMOSS-Team/MOSS-Image-SFT") -> None:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    kernels = args.kernels.resolve()
    weights = args.weights.resolve()
    km = json.loads((kernels / "manifest.json").read_text())
    wm = json.loads((weights / "manifest.json").read_text())
    if not km.get("complete") or args.resolution not in km["resolutions"]:
        raise ValueError("VAE kernel set is missing this resolution")
    if (km["profile"] != list(profile) or wm["profile"] != list(profile)
            or km["family"] != model or wm["model"] != model):
        raise ValueError("VAE kernel and weight model profiles differ")
    vp = plan(args.resolution, profile)
    stream_sets = dict(vp.streams)
    stream_sets["rgba"] = {name: spec for name, spec in vp.streams["vew"].items()
                           if spec["op"] == "rgba"}
    stream_sets["vew"] = {name: spec for name, spec in vp.streams["vew"].items()
                          if spec["op"] != "rgba"}
    entries = []
    for kset, streams in stream_sets.items():
        marker = json.loads((kernels / kset / "set.json").read_text())
        if marker["streams"] != streams or not marker.get("complete"):
            raise ValueError(f"VAE {kset} streams differ from the current plan")
        entries.append(f"xclbin {kset} {kernels / kset / 'final.xclbin'}")
        entries += [f"kernelx {name} {kset} {kernels / kset / f'insts_{name}.bin'}"
                    for name in streams]
    for name, buf in vp.buffers.items():
        initial = ""
        if name == "LAT":
            lat = np.zeros(buf.nelems, bfloat16)
            n = (args.resolution // 16) ** 2 * 128
            if args.latents_f32 is None:
                lat[:n] = np.random.default_rng(args.seed).standard_normal(n).astype(bfloat16)
            else:
                values = np.fromfile(args.latents_f32, dtype="<f4")
                if values.size != n:
                    raise ValueError("VAE latents must be FP32 [128,H/16,W/16]")
                lat[:n] = values.reshape(128, n // 128).T.reshape(-1).astype(bfloat16)
            path = out / "latents.bf16"
            lat.tofile(path)
            initial = f" {path}"
        elif name == "QIN":
            qin = np.zeros(buf.nelems, bfloat16)
            qin.reshape(-1, buf.C)[:, wm["attention_bias_column"]] = bfloat16(1)
            path = out / "attention-ones.bf16"
            qin.tofile(path)
            initial = f" {path}"
        entries.append(f"buf v_{name} {buf.nelems * 2}{initial}")
    packed = np.memmap(weights / "weights.bin", np.uint8, mode="r")
    for key, slot in wm["weight_slices"].items():
        path = out / f"w_{key}.bin"
        packed[slot["offset"]:slot["offset"] + slot["bytes"]].tofile(path)
        entries.append(f"buf w_{key} {slot['bytes']} {path}")
    gn = np.memmap(weights / "gn_blocks.bf16", np.uint8, mode="r")
    block_bytes = BLOCK * wm["groupnorm_element"] * 2
    for key, index in wm["groupnorm_blocks"].items():
        path = out / f"g_{key}.bf16"
        gn[index * block_bytes:(index + 1) * block_bytes].tofile(path)
        entries.append(f"buf g_{key} {block_bytes} {path}")
    for op in vp.ops:
        names = []
        for kind, key in op["args"]:
            names.append({"buf": "v_", "w": "w_", "gn": "g_"}[kind] + key)
        entries.append(f"# {op['what']}")
        entries.append(f"run {op['stream']} {' '.join(names)}")
    out_size = vp.buffers["RGBA"].nelems * 2
    entries.append(f"dump v_RGBA {out / 'output-rgba-blocks.bin'} {out_size}")
    (out / "run.cfg").write_text("\n".join(["device", *entries, ""]))
    (out / "fixture.json").write_text(json.dumps({
        "model": model,
        "resolution": args.resolution,
        "seed": args.seed,
        "operations": len(vp.ops),
        "kernel_manifest": str(kernels / "manifest.json"),
        "weight_manifest": str(weights / "manifest.json"),
        "scope": "native VAE decoder only",
    }, indent=2) + "\n")
    print(f"prepared {len(vp.ops)} native VAE operations -> {out / 'run.cfg'}")


def check(args) -> None:
    out = args.out.resolve()
    fixture = json.loads((out / "fixture.json").read_text())
    R = fixture["resolution"]
    raw = np.fromfile(out / "output-rgba-blocks.bin", np.uint8)
    if raw.size != R * R * 8:
        raise ValueError(f"wrong native RGBA output size: {raw.size}")
    rgba = raw.reshape(-1, 8192)[:, :4096].reshape(R, R, 4)
    if not np.all(rgba[..., 3] == 255):
        raise ValueError("native VAE output alpha is not opaque")
    path = out / "native-vae.png"
    write_png(path, rgba)
    print(f"native VAE RGBA output: {R}x{R}, RGB range "
          f"{int(rgba[..., :3].min())}..{int(rgba[..., :3].max())} -> {path}")


def main(*, profile=MOSS_BLOCK_OUT,
         model="OpenMOSS-Team/MOSS-Image-SFT") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--latents-f32", type=Path,
                   help="FP32 channel-first [128,H/16,W/16] sampler latents")
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args, profile=profile, model=model) if args.command == "prepare" else check(args)


if __name__ == "__main__":
    main()
