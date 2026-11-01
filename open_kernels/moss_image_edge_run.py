"""Prepare/check native MOSS latent embedding and final velocity projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.moss_dit_block import EL, plan as block_plan
from open_kernels.moss_image_edges import entry_bias, final_norm_params, specs


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def verify(edges: Path, weights: Path, resolution: int) -> tuple[dict, dict]:
    km = json.loads((edges / "manifest.json").read_text())
    wm = json.loads((weights / "packed_weights.json").read_text())
    if (not km.get("complete") or km["resolution"] != resolution or
            not wm.get("full_model") or wm.get("model") != "MOSS-Image-SFT"):
        raise ValueError("MOSS edge kernels or packed weights differ")
    for kset, streams in specs(resolution).items():
        marker = json.loads((edges / kset / "set.json").read_text())
        if not marker.get("complete") or marker["streams"] != streams:
            raise ValueError(f"MOSS edge {kset} streams differ from source")
    return km, wm


def weight_entry(entries: list[str], weights: Path, wm: dict, name: str) -> None:
    slot = wm["slots"][name]
    path = weights / slot["file"]
    if path.stat().st_size != slot["bytes"] or digest(path) != slot["sha256"]:
        raise ValueError(f"MOSS edge weight failed integrity check: {name}")
    entries.append(f"buf w_{name} {slot['bytes']} {path}")


def prepare_entry(args) -> None:
    edges, weights, model, out = (p.resolve() for p in
                                  (args.kernels, args.weights, args.model, args.out))
    _, wm = verify(edges, weights, args.resolution)
    tokens = (args.resolution // 16) ** 2
    out.mkdir(parents=True, exist_ok=False)
    lat = np.zeros(tokens * 128 + 512, bfloat16)
    if args.latents_f32 is None:
        lat[:tokens * 128] = np.random.default_rng(args.seed).standard_normal(
            tokens * 128).astype(bfloat16)
    else:
        values = np.fromfile(args.latents_f32, dtype="<f4")
        if values.size != 128 * tokens:
            raise ValueError("MOSS image latents must be FP32 [128,H/16,W/16]")
        lat[:tokens * 128] = values.reshape(128, tokens).T.reshape(-1).astype(bfloat16)
    lat.tofile(out / "latents.bf16")
    bias = entry_bias(model)
    bias.tofile(out / "bias.bf16")
    entries = ["device", f"xclbin gemm {edges / 'gemm' / 'final.xclbin'}",
               f"kernelx emb.image gemm {edges / 'gemm' / 'insts_emb.image.bin'}",
               f"xclbin bias {edges / 'bias' / 'final.xclbin'}",
               f"kernelx bias_image bias {edges / 'bias' / 'insts_bias_image.bin'}",
               f"buf LAT {lat.nbytes} {out / 'latents.bf16'}",
               f"buf X {tokens * EL * 2}",
               f"buf P {bias.nbytes} {out / 'bias.bf16'}",
               f"buf DUMMY {EL * 2}"]
    weight_entry(entries, weights, wm, "emb.image")
    entries += ["run emb.image LAT w_emb.image X",
                "run bias_image X DUMMY P X DUMMY",
                f"dump X {out / 'native-image-x.bf16'} {tokens * EL * 2}", ""]
    (out / "run.cfg").write_text("\n".join(entries))
    (out / "fixture.json").write_text(json.dumps({
        "model": "OpenMOSS-Team/MOSS-Image-SFT", "stage": "image entry",
        "resolution": args.resolution, "seed": args.seed, "tokens": tokens,
    }, indent=2) + "\n")
    print(f"prepared MOSS image entry -> {out / 'run.cfg'}")


def prepare_exit(args) -> None:
    edges, block, weights, model, out = (p.resolve() for p in
                                           (args.kernels, args.block_kernels,
                                            args.weights, args.model, args.out))
    _, wm = verify(edges, weights, args.resolution)
    bp = block_plan(args.resolution, args.text_length)
    bm = json.loads((block / "manifest.json").read_text())
    marker = json.loads((block / "ew" / "set.json").read_text())
    if (not bm.get("complete") or bm["resolution"] != args.resolution or
            bm["text_length"] != args.text_length or
            marker["streams"]["norm1"] != bp.streams["ew"]["norm1"]):
        raise ValueError("MOSS block norm stream differs from final-head contract")
    source = args.hidden.resolve()
    if source.stat().st_size != bp.buffers["X"] * 2:
        raise ValueError("MOSS joint hidden buffer has wrong extent")
    head = np.fromfile(args.head.resolve(), bfloat16)
    if head.size != 512 * 2 * EL:
        raise ValueError("MOSS head modulation buffer has wrong extent")
    out.mkdir(parents=True, exist_ok=False)
    p = final_norm_params(model, head[:2 * EL].astype(np.float32).reshape(2, EL))
    p.tofile(out / "final-norm.bf16")
    tokens = (args.resolution // 16) ** 2
    entries = ["device", f"xclbin ew {block / 'ew' / 'final.xclbin'}",
               f"kernelx norm1 ew {block / 'ew' / 'insts_norm1.bin'}",
               f"xclbin gemm {edges / 'gemm' / 'final.xclbin'}",
               f"kernelx final.image gemm {edges / 'gemm' / 'insts_final.image.bin'}",
               f"buf X {source.stat().st_size} {source}",
               f"buf Z {bp.buffers['Z'] * 2}",
               f"buf P {p.nbytes} {out / 'final-norm.bf16'}",
               f"buf DUMMY {EL * 2}",
               f"buf V {tokens * 1024 * 2}"]
    weight_entry(entries, weights, wm, "final.image")
    entries += ["run norm1 X DUMMY P Z DUMMY",
                "run final.image Z w_final.image V",
                f"dump V {out / 'native-velocity.bf16'} {tokens * 1024 * 2}", ""]
    (out / "run.cfg").write_text("\n".join(entries))
    (out / "fixture.json").write_text(json.dumps({
        "model": "OpenMOSS-Team/MOSS-Image-SFT", "stage": "image exit",
        "resolution": args.resolution, "text_length": args.text_length,
        "tokens": tokens, "joint_hidden": str(source),
        "head_modulation": str(args.head.resolve()),
    }, indent=2) + "\n")
    print(f"prepared MOSS image exit -> {out / 'run.cfg'}")


def check(args) -> None:
    out = args.out.resolve()
    meta = json.loads((out / "fixture.json").read_text())
    if meta["stage"] == "image entry":
        shape, native_path, ref_path = ((meta["tokens"], EL),
                                        "native-image-x.bf16", "reference-image-x.f32")
        live = EL
    else:
        shape, native_path, ref_path = ((meta["tokens"], 1024),
                                        "native-velocity.bf16", "reference-velocity.f32")
        live = 128
    native = np.fromfile(out / native_path, bfloat16)
    if native.size != int(np.prod(shape)):
        raise ValueError("MOSS image edge native output has wrong extent")
    native = native.reshape(shape)[:, :live].astype(np.float32)
    if not np.isfinite(native).all():
        raise ValueError("MOSS image edge native output contains NaN/Inf")
    print(f"native {meta['stage']}: RMS {np.sqrt(np.mean(native ** 2)):.6g}")
    ref = out / ref_path
    if ref.is_file():
        expected = np.fromfile(ref, "<f4")
        if expected.size != native.size:
            raise ValueError("MOSS image edge reference has wrong extent")
        expected = expected.reshape(native.shape)
        diff = native - expected
        rel = np.linalg.norm(diff.ravel()) / np.linalg.norm(expected.ravel())
        cosine = np.dot(native.ravel(), expected.ravel()) / (
            np.linalg.norm(native.ravel()) * np.linalg.norm(expected.ravel()))
        print(f"vs BF16 checkpoint reference: relative L2 {rel:.6g}, "
              f"cosine {cosine:.9g}, max abs {np.max(np.abs(diff)):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-entry")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--latents-f32", type=Path,
                   help="FP32 channel-first [128,H/16,W/16] sampler latents")
    p = sub.add_parser("prepare-exit")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--block-kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--hidden", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--text-length", type=int, default=17)
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prepare-entry": prepare_entry, "prepare-exit": prepare_exit,
     "check": check}[args.command](args)


if __name__ == "__main__":
    main()
