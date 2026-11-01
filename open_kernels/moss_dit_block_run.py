"""Prepare and check a real-weight MOSS joint-block replay on the AMD NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.moss_dit_block import EL, params, plan


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def prepare(args) -> None:
    kernels = args.kernels.resolve()
    weights = args.weights.resolve()
    model = args.model.resolve()
    out = args.out.resolve()
    p = plan(args.resolution, args.text_length, args.layer)
    if args.layers < 1 or args.layer + args.layers > 42:
        raise ValueError("MOSS replay layers must lie within 0..41")
    if args.skip_layer is not None and not (args.layer <= args.skip_layer < args.layer + args.layers):
        raise ValueError("MOSS skipped layer must lie within the selected range")
    km = json.loads((kernels / "manifest.json").read_text())
    wm = json.loads((weights / "packed_weights.json").read_text())
    if (not km.get("complete") or km["resolution"] != args.resolution or
            km["text_length"] != args.text_length or km["layer"] != args.layer):
        raise ValueError("MOSS joint-block kernel set differs from the fixture")
    sets = {"gemm": p.streams["gemm"],
            "ew": {k: v for k, v in p.streams["ew"].items()
                   if k not in ("qk", "situ")},
            "qk": {"qk": p.streams["ew"]["qk"]},
            "situ": {"situ": p.streams["ew"]["situ"]},
            "fa": p.streams["fa"]}
    for kset, specs in sets.items():
        marker = json.loads((kernels / kset / "set.json").read_text())
        if marker["streams"] != specs or not marker.get("complete"):
            raise ValueError(f"MOSS {kset} stream set differs from current source")
    if not wm.get("full_model") or wm.get("model") != "MOSS-Image-SFT":
        raise ValueError("expected the complete MOSS DiT weight pack")
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    x = np.zeros((p.tokens, EL), bfloat16)
    if args.image_x or args.text_x:
        if not args.image_x or not args.text_x:
            raise ValueError("supply both native image and text hidden buffers")
        image_tokens = (args.resolution // 16) ** 2
        image = np.fromfile(args.image_x.resolve(), bfloat16)
        text = np.fromfile(args.text_x.resolve(), bfloat16)
        if image.size != image_tokens * EL or text.size < args.text_length * EL:
            raise ValueError("native MOSS image/text hidden buffer has wrong extent")
        x[:image_tokens] = image.reshape(image_tokens, EL)
        x[image_tokens:p.valid_tokens] = text[:args.text_length * EL].reshape(args.text_length, EL)
    else:
        x[:p.valid_tokens] = rng.normal(0, 0.6, (p.valid_tokens, EL)).astype(bfloat16)
    if args.modulation:
        mod = np.fromfile(args.modulation.resolve(), bfloat16)
        if mod.size < 6 * EL:
            raise ValueError("native MOSS modulation buffer has wrong extent")
        modulation = mod[:6 * EL].astype(np.float32).reshape(6, EL)
    else:
        modulation = rng.normal(0, 0.1, (6, EL)).astype(np.float32)
        modulation[[2, 5]] += 0.8  # make both gated residual paths observable
    initial = {"X": x}
    x.tofile(out / "input-x.bf16")
    modulation.tofile(out / "modulation.f32")
    entries = ["device"]
    for kset, specs in sets.items():
        entries.append(f"xclbin {kset} {kernels / kset / 'final.xclbin'}")
        entries.extend(f"kernelx {name} {kset} {kernels / kset / f'insts_{name}.bin'}"
                       for name in specs)
    scratch = {key: value for key, value in p.buffers.items()
               if not key.startswith("P_")}
    for name, elements in scratch.items():
        source = ""
        if name in initial:
            path = out / f"{name}.bf16"
            initial[name].tofile(path)
            if path.stat().st_size != elements * 2:
                raise ValueError(f"{name} buffer size differs from NPU plan")
            source = f" {path}"
        entries.append(f"buf b_{name} {elements * 2}{source}")
    for layer in range(args.layer, args.layer + args.layers):
        if layer == args.skip_layer:
            continue
        for name, data in params(model, layer, args.resolution, modulation).items():
            path = out / f"{name}_L{layer}.bf16"
            data.tofile(path)
            entries.append(f"buf b_{name}_L{layer} {data.nbytes} {path}")
        for name in ("qkv", "attn_out", "ffn_in", "ffn_out"):
            key = f"block.{layer}.{name}"
            slot = wm["slots"][key]
            path = weights / slot["file"]
            if path.stat().st_size != slot["bytes"] or digest(path) != slot["sha256"]:
                raise ValueError(f"MOSS packed weight failed integrity check: {key}")
            entries.append(f"buf w_{name}_L{layer} {slot['bytes']} {path}")
        for op in plan(args.resolution, args.text_length, layer).ops:
            names = [("w_" + key.rsplit(".", 1)[1] + f"_L{layer}")
                     if key.startswith("block.") else
                     ("b_" + key + f"_L{layer}") if key.startswith("P_") else
                     ("b_" + key) for key in op["args"]]
            entries.append(f"run {op['stream']} {' '.join(names)}")
    entries.append(f"dump b_X {out / 'native-x.bf16'} {p.buffers['X'] * 2}")
    (out / "run.cfg").write_text("\n".join(entries + [""]))
    (out / "fixture.json").write_text(json.dumps({
        "model": "OpenMOSS-Team/MOSS-Image-SFT",
        "scope": "native joint transformer block range",
        "resolution": args.resolution,
        "text_length": args.text_length,
        "layer": args.layer,
        "layers": args.layers,
        "tokens": p.tokens,
        "valid_tokens": p.valid_tokens,
        "seed": args.seed,
        "input_sources": {"image": str(args.image_x.resolve()) if args.image_x else None,
                          "text": str(args.text_x.resolve()) if args.text_x else None,
                          "modulation": str(args.modulation.resolve()) if args.modulation else None},
        "skip_layer": args.skip_layer,
        "operations": len(p.ops) * (args.layers - int(args.skip_layer is not None)),
        "kernel_manifest": str(kernels / "manifest.json"),
        "weight_manifest": str(weights / "packed_weights.json"),
    }, indent=2) + "\n")
    print(f"prepared {len(p.ops) * (args.layers - int(args.skip_layer is not None))} "
          f"real-weight NPU operations -> {out / 'run.cfg'}")


def check(args) -> None:
    out = args.out.resolve()
    fixture = json.loads((out / "fixture.json").read_text())
    p = plan(fixture["resolution"], fixture["text_length"], fixture["layer"])
    x = np.fromfile(out / "native-x.bf16", dtype=bfloat16)
    if x.size != p.buffers["X"]:
        raise ValueError("NPU joint-block output has the wrong size")
    x = x.reshape(p.tokens, EL).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("NPU joint-block output contains NaN/Inf")
    print(f"native MOSS block: {p.valid_tokens}/{p.tokens} rows, "
          f"range {x[:p.valid_tokens].min():.4g}..{x[:p.valid_tokens].max():.4g}, "
          f"RMS {np.sqrt(np.mean(x[:p.valid_tokens] ** 2)):.4g}")
    if (out / "reference-x.f32").is_file():
        ref = np.fromfile(out / "reference-x.f32", dtype="<f4")
        if ref.size != p.valid_tokens * EL:
            raise ValueError("reference MOSS block output has the wrong size")
        ref = ref.reshape(p.valid_tokens, EL)
        err = x[:p.valid_tokens] - ref
        rel = np.linalg.norm(err.ravel()) / np.linalg.norm(ref.ravel())
        cosine = np.dot(x[:p.valid_tokens].ravel(), ref.ravel()) / (
            np.linalg.norm(x[:p.valid_tokens].ravel()) * np.linalg.norm(ref.ravel()))
        print(f"vs BF16 checkpoint reference: relative L2 {rel:.6g}, cosine {cosine:.9g}, "
              f"max abs {np.max(np.abs(err)):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--text-length", type=int, default=17)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--skip-layer", type=int,
                   help="zero-based block to omit, as in the negative CFG branch")
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--image-x", type=Path)
    p.add_argument("--text-x", type=Path)
    p.add_argument("--modulation", type=Path)
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else check(args)


if __name__ == "__main__":
    main()
