"""Prepare/check one real-weight FLUX 3 per-mode image block on the AMD NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.flux3_block import EL, params, plan
from open_kernels.flux3_shared_block import plan as shared_plan, params as shared_params


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(args) -> None:
    kernels, weights, model, out = (Path(v).resolve() for v in
                                    (args.kernels, args.weights, args.model, args.out))
    p = (shared_plan(args.resolution, args.text_length, args.layer, args.references)
         if args.stage == "shared" else
         plan(args.resolution, args.layer, mode=args.mode, text_length=args.text_length))
    if args.layers < 1 or args.layer + args.layers > (28 if args.stage == "shared" else 5):
        raise ValueError("FLUX 3 block range exceeds its stage")
    km = json.loads((kernels / "manifest.json").read_text())
    wm = json.loads((weights / "packed_weights.json").read_text())
    if (not km.get("complete") or km["resolution"] != args.resolution or
            km["layer"] != args.layer or km.get("stage", "early") != args.stage or
            km.get("mode", "image") != ("joint" if args.stage == "shared" else args.mode) or
            km.get("text_length", 0) != (args.text_length if args.stage == "shared" or args.mode == "text" else 0) or
            km.get("references", 0) != (args.references if args.stage == "shared" else 0)):
        raise ValueError("FLUX 3 block kernel set differs from the requested fixture")
    if not wm.get("full_model") or wm["model"] != "flux-3-image-klein":
        raise ValueError("expected complete FLUX 3 Image Klein weight pack")
    sets = {"gemm": p.streams["gemm"],
            "ew": {k: v for k, v in p.streams["ew"].items() if k != "qk"},
            "qk": {"qk": p.streams["ew"]["qk"]},
            "fa": p.streams["fa"]}
    for kset, specs in sets.items():
        marker = json.loads((kernels / kset / "set.json").read_text())
        if not marker.get("complete") or marker["streams"] != specs:
            raise ValueError(f"FLUX 3 {kset} stream set differs from source")
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    x = np.zeros((p.tokens, EL), bfloat16)
    if args.input_bf16:
        source = np.fromfile(args.input_bf16.resolve(), bfloat16)
        if source.size < p.valid_tokens * EL:
            raise ValueError("FLUX 3 block input is shorter than its live token extent")
        x[:p.valid_tokens] = source[:p.valid_tokens * EL].reshape(p.valid_tokens, EL)
    else:
        x[:p.valid_tokens] = rng.normal(0, 0.6, (p.valid_tokens, EL)).astype(bfloat16)
    if args.stage == "shared":
        mods = {}
        for mode in p.modes:
            if args.modulation_dir:
                value = np.fromfile(args.modulation_dir.resolve() /
                                    f"modulation-single_{mode}.bf16", bfloat16)
                if value.size != 3 * EL:
                    raise ValueError(f"FLUX 3 {mode} shared modulation has wrong extent")
                value = value.reshape(3, EL)
            else:
                value = rng.normal(0, 0.1, (3, EL)).astype(bfloat16)
                value[2] = (value[2].astype(np.float32) + 0.8).astype(bfloat16)
            mods[mode] = value
            value.tofile(out / f"modulation_{mode}.bf16")
    else:
        if args.modulation_dir:
            mod = np.fromfile(args.modulation_dir.resolve() /
                              f"modulation-early_{args.mode}.bf16", bfloat16)
            if mod.size != 3 * EL:
                raise ValueError("FLUX 3 early modulation has wrong extent")
            mod = mod.reshape(3, EL)
        else:
            mod = rng.normal(0, 0.1, (3, EL)).astype(bfloat16)
            mod[2] = (mod[2].astype(np.float32) + 0.8).astype(bfloat16)
        mod.tofile(out / "modulation.bf16")
    x.tofile(out / "input-x.bf16")
    entries = ["device"]
    for kset, specs in sets.items():
        entries.append(f"xclbin {kset} {kernels / kset / 'final.xclbin'}")
        entries.extend(f"kernelx {name} {kset} {kernels / kset / f'insts_{name}.bin'}"
                       for name in specs)
    for name, size in p.buffers.items():
        source = f" {out / 'input-x.bf16'}" if name == "X" else ""
        entries.append(f"buf b_{name} {size * 2}{source}")
    for layer in range(args.layer, args.layer + args.layers):
        suffix = f"_L{layer}"
        packed_params = (shared_params(model, layer,
                                       {k: v.astype(np.float32) for k, v in mods.items()})
                         if args.stage == "shared" else
                         params(model, layer, mod.astype(np.float32), mode=args.mode))
        for name, data in packed_params.items():
            path = out / f"{name}{suffix}.bf16"
            data.tofile(path)
            entries.append(f"buf b_{name}{suffix} {data.nbytes} {path}")
        for name in p.streams["gemm"]:
            key = f"{('single' if args.stage == 'shared' else args.mode)}.{layer}.{name}"
            slot = wm["slots"][key]
            path = weights / slot["file"]
            if path.stat().st_size != slot["bytes"] or digest(path) != slot["sha256"]:
                raise ValueError(f"FLUX 3 packed weight failed integrity check: {key}")
            entries.append(f"buf w_{name}{suffix} {slot['bytes']} {path}")
        for op in p.ops:
            names = [(v + suffix) if v.startswith("w_") else
                     (f"b_{v}{suffix}" if v.startswith("P_") else f"b_{v}")
                     for v in op["args"]]
            entries.append(f"run {op['stream']} {' '.join(names)}")
    for name in ("QKV", "ATTN", "AOUT", "MOUT", "SUM"):
        entries.append(f"dump b_{name} {out / f'native-{name.lower()}.bf16'} "
                       f"{p.buffers[name] * 2}")
    entries.append(f"dump b_X {out / 'native-x.bf16'} {p.buffers['X'] * 2}")
    (out / "run.cfg").write_text("\n".join(entries + [""]))
    (out / "fixture.json").write_text(json.dumps({
        "model": "black-forest-labs/flux-3-image-klein",
        "scope": "native checkpoint-weighted per-mode block",
        "resolution": args.resolution, "layer": args.layer,
        "layers": args.layers, "stage": args.stage,
        "mode": "joint" if args.stage == "shared" else args.mode,
        "text_length": args.text_length if args.stage == "shared" or args.mode == "text" else 0,
        "tokens": p.tokens, "valid_tokens": p.valid_tokens, "seed": args.seed,
        "references": args.references if args.stage == "shared" else 0,
        "operations": len(p.ops) * args.layers,
        "kernel_manifest": str(kernels / "manifest.json"),
        "weight_manifest": str(weights / "packed_weights.json"),
    }, indent=2) + "\n")
    print(f"prepared {len(p.ops) * args.layers} FLUX 3 block NPU operations -> {out / 'run.cfg'}")


def check(args) -> None:
    out = Path(args.out).resolve()
    fixture = json.loads((out / "fixture.json").read_text())
    p = (shared_plan(fixture["resolution"], fixture["text_length"], fixture["layer"],
                     fixture.get("references", 0))
         if fixture.get("stage") == "shared" else
         plan(fixture["resolution"], fixture["layer"],
              mode=fixture.get("mode", "image"),
              text_length=fixture.get("text_length", 80) or 80))
    x = np.fromfile(out / "native-x.bf16", dtype=bfloat16)
    if x.size != p.buffers["X"]:
        raise ValueError("FLUX 3 block output has wrong extent")
    x = x.reshape(p.tokens, EL).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("FLUX 3 block output contains NaN/Inf")
    print(f"native FLUX 3 {fixture.get('mode', 'image')} block: {p.valid_tokens}/{p.tokens} rows, RMS "
          f"{np.sqrt(np.mean(x[:p.valid_tokens]**2)):.6g}")
    if (out / "reference-x.f32").is_file():
        y = np.fromfile(out / "reference-x.f32", "<f4").reshape(p.valid_tokens, EL)
        x = x[:p.valid_tokens]
        diff = x - y
        print(f"vs published BF16 block: relative L2 "
              f"{np.linalg.norm(diff)/np.linalg.norm(y):.6g}, cosine "
              f"{np.vdot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)):.9g}, "
              f"max abs {np.max(np.abs(diff)):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--mode", choices=("image", "image_cond", "text"), default="image")
    p.add_argument("--stage", choices=("early", "shared"), default="early")
    p.add_argument("--text-length", type=int, default=80)
    p.add_argument("--references", type=int, default=0,
                   help="shared stage: reference images joined after the generated image")
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--modulation-dir", type=Path)
    p.add_argument("--input-bf16", type=Path)
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else check(args)


if __name__ == "__main__":
    main()
