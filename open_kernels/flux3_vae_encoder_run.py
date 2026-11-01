"""Prepare/check a complete native FLUX 3 VAE image encoder pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.vae_encoder import BLOCK, BLOCK_OUT, plan


MODEL = "black-forest-labs/flux-3-image-klein"


def image_input(args) -> np.ndarray:
    resolution = args.resolution
    if args.image_f32:
        rgb = np.fromfile(args.image_f32.resolve(), "<f4")
        if rgb.size != resolution * resolution * 3:
            raise ValueError("FLUX 3 VAE encoder FP32 input has wrong RGB extent")
        rgb = rgb.reshape(resolution, resolution, 3)
        if not np.isfinite(rgb).all() or np.max(np.abs(rgb)) > 1:
            raise ValueError("FLUX 3 VAE encoder FP32 input must be finite RGB in [-1,1]")
        return rgb
    if args.image:
        from PIL import Image
        with Image.open(args.image.resolve()) as source:
            if source.size != (resolution, resolution):
                raise ValueError("FLUX 3 VAE encoder input must match --resolution")
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    else:
        y, x = np.indices((resolution, resolution))
        rgb = np.stack(((x * 255 // resolution), (y * 255 // resolution),
                        (((x // 32 + y // 32) % 2) * 255)), axis=-1).astype(np.float32)
    return rgb / np.float32(127.5) - np.float32(1)


def prepare(args) -> None:
    out, kernels, weights = (p.resolve() for p in (args.out, args.kernels, args.weights))
    km = json.loads((kernels / "manifest.json").read_text())
    wm = json.loads((weights / "manifest.json").read_text())
    if (not km.get("complete") or km.get("stage") != "encoder" or
            args.resolution not in km["resolutions"] or km["profile"] != list(BLOCK_OUT) or
            km["family"] != MODEL or wm.get("stage") != "encoder" or
            wm["profile"] != list(BLOCK_OUT) or wm["model"] != MODEL):
        raise ValueError("FLUX 3 VAE encoder kernel and weight manifests differ")
    vp = plan(args.resolution)
    for kset, streams in vp.streams.items():
        marker = json.loads((kernels / kset / "set.json").read_text())
        if not marker.get("complete") or marker["streams"] != streams:
            raise ValueError(f"FLUX 3 VAE encoder {kset} streams differ from current plan")
    out.mkdir(parents=True, exist_ok=False)
    entries = ["device"]
    for kset, streams in vp.streams.items():
        entries.append(f"xclbin {kset} {kernels / kset / 'final.xclbin'}")
        entries += [f"kernelx {name} {kset} {kernels / kset / f'insts_{name}.bin'}"
                    for name in streams]
    for name, buf in vp.buffers.items():
        initial = ""
        if name == "IN":
            rgb = image_input(args)
            rgb.astype("<f4").tofile(out / "image.f32")
            data = np.zeros((buf.H + 2, buf.pitch, buf.C), bfloat16)
            data[1:1 + buf.H, 1:1 + buf.W, :3] = rgb.astype(bfloat16)
            path = out / "image-input.bf16"
            data.tofile(path)
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
    traces = {0: "conv_in", 17: "down0", 35: "down1", 53: "down2",
              72: "mid0", 76: "attention", 82: "mid1"}
    for index, op in enumerate(vp.ops):
        names = [{"buf": "v_", "w": "w_", "gn": "g_"}[kind] + key
                 for kind, key in op["args"]]
        entries.append(f"# {op['what']}")
        entries.append(f"run {op['stream']} {' '.join(names)}")
        if args.trace and index in traces:
            name = "X0" if index == 0 else "X1" if index == 17 else \
                "X2" if index == 35 else "X3"
            entries.append(f"dump v_{name} {out / f'native-{traces[index]}.bf16'} "
                           f"{vp.buffers[name].nelems * 2}")
    entries.append(f"dump v_REF {out / 'native-tokens.bf16'} {vp.buffers['REF'].nelems * 2}")
    (out / "run.cfg").write_text("\n".join([*entries, ""]))
    (out / "fixture.json").write_text(json.dumps({
        "model": MODEL, "stage": "encoder", "resolution": args.resolution,
        "operations": len(vp.ops), "source_image": str(args.image) if args.image else None,
        "trace": bool(args.trace),
        "scope": "native VAE encoder only",
    }, indent=2) + "\n")
    print(f"prepared {len(vp.ops)} native FLUX 3 VAE encoder operations -> {out / 'run.cfg'}")


def check(args) -> None:
    out = args.out.resolve()
    meta = json.loads((out / "fixture.json").read_text())
    h = meta["resolution"] // 16
    tokens = np.fromfile(out / "native-tokens.bf16", bfloat16)
    if tokens.size != h * h * 128:
        raise ValueError("FLUX 3 VAE encoder produced wrong token extent")
    tokens = tokens.reshape(h * h, 128).astype(np.float32)
    if not np.isfinite(tokens).all():
        raise ValueError("FLUX 3 VAE encoder produced nonfinite tokens")
    print(f"native FLUX 3 VAE encoder: {h*h} tokens, RMS "
          f"{np.sqrt(np.mean(tokens*tokens)):.6g}")
    ref = out / "reference-tokens.f32"
    if ref.is_file():
        expected = np.fromfile(ref, "<f4").reshape(tokens.shape)
        delta = tokens - expected
        print(f"vs published FP32 encoder: relative L2 "
              f"{np.linalg.norm(delta)/np.linalg.norm(expected):.6g}, "
              f"cosine {np.vdot(tokens, expected)/(np.linalg.norm(tokens)*np.linalg.norm(expected)):.9g}, "
              f"max abs {np.max(np.abs(delta)):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--image", type=Path)
    group.add_argument("--image-f32", type=Path)
    p.add_argument("--trace", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else check(args)


if __name__ == "__main__":
    main()
