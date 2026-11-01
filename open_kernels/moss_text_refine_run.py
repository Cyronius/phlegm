"""Run MOSS text refinement on the NPU from saved native Qwen3.5 hidden rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.moss_text_refine import params, plan


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def prepare(args) -> None:
    kernels, weights, model = (p.resolve() for p in
                               (args.kernels, args.weights, args.model))
    out = args.out.resolve()
    p = plan(args.text_length)
    km = json.loads((kernels / "manifest.json").read_text())
    wm = json.loads((weights / "packed_weights.json").read_text())
    if not km.get("complete") or km["text_length"] != args.text_length:
        raise ValueError("MOSS text kernel manifest differs from fixture")
    if not wm.get("full_model") or wm.get("model") != "MOSS-Image-SFT":
        raise ValueError("expected complete MOSS DiT weight pack")
    sets = {"gemm": p.streams["gemm"],
            "ew": {k: v for k, v in p.streams["ew"].items()
                   if k in ("norm", "residual")},
            "bias": {k: v for k, v in p.streams["ew"].items()
                     if k.startswith("bias")},
            "qk": {"qk": p.streams["ew"]["qk"]},
            "situ": {"situ": p.streams["ew"]["situ"]},
            "fa": p.streams["fa"]}
    for kset, specs in sets.items():
        marker = json.loads((kernels / kset / "set.json").read_text())
        if not marker.get("complete") or marker["streams"] != specs:
            raise ValueError(f"MOSS text {kset} streams differ from source")
    hidden = np.fromfile(args.hidden.resolve(), dtype="<f4")
    if hidden.size != args.text_length * 4096 or not np.isfinite(hidden).all():
        raise ValueError("native Qwen3.5 hidden rows differ from expected shape")
    out.mkdir(parents=True, exist_ok=False)
    context = np.zeros((p.tokens, 4096), bfloat16)
    context[:args.text_length] = hidden.reshape(args.text_length, 4096).astype(bfloat16)
    context.tofile(out / "context.bf16")
    ps = params(model)
    entries = ["device"]
    for kset, specs in sets.items():
        entries.append(f"xclbin {kset} {kernels / kset / 'final.xclbin'}")
        entries.extend(f"kernelx {name} {kset} {kernels / kset / f'insts_{name}.bin'}"
                       for name in specs)
    for name, elements in p.buffers.items():
        source = f" {out / 'context.bf16'}" if name == "CTX" else ""
        entries.append(f"buf b_{name} {elements * 2}{source}")
    for name, data in ps.items():
        path = out / f"{name}.bf16"
        data.tofile(path)
        entries.append(f"buf p_{name} {data.nbytes} {path}")
    for name in p.streams["gemm"]:
        slot = wm["slots"][name]
        path = weights / slot["file"]
        if path.stat().st_size != slot["bytes"] or digest(path) != slot["sha256"]:
            raise ValueError(f"packed MOSS text weight failed integrity check: {name}")
        entries.append(f"buf w_{name} {slot['bytes']} {path}")
    for op in p.ops:
        names = [("w_" + key) if key in p.streams["gemm"] else
                 ("p_" + key) if key in ps else ("b_" + key)
                 for key in op["args"]]
        entries.append(f"run {op['stream']} {' '.join(names)}")
    entries.append(f"dump b_X {out / 'native-text.bf16'} {p.buffers['X'] * 2}")
    (out / "run.cfg").write_text("\n".join(entries + [""]))
    (out / "fixture.json").write_text(json.dumps({
        "model": "OpenMOSS-Team/MOSS-Image-SFT",
        "scope": "native text projection and refinement",
        "text_length": args.text_length,
        "tokens": p.tokens,
        "operations": len(p.ops),
        "encoder_hidden": str(args.hidden.resolve()),
        "kernel_manifest": str(kernels / "manifest.json"),
        "weight_manifest": str(weights / "packed_weights.json"),
    }, indent=2) + "\n")
    print(f"prepared {len(p.ops)} MOSS text NPU operations -> {out / 'run.cfg'}")


def check(args) -> None:
    out = args.out.resolve()
    meta = json.loads((out / "fixture.json").read_text())
    p = plan(meta["text_length"])
    native = np.fromfile(out / "native-text.bf16", bfloat16)
    if native.size != p.buffers["X"]:
        raise ValueError("native MOSS text output has wrong extent")
    native = native.reshape(p.tokens, 3072).astype(np.float32)[:p.text_length]
    if not np.isfinite(native).all():
        raise ValueError("native MOSS text output contains NaN/Inf")
    print(f"native MOSS text refinement: {p.text_length} rows, "
          f"RMS {np.sqrt(np.mean(native ** 2)):.6g}")
    reference = out / "reference-text.f32"
    if reference.is_file():
        expected = np.fromfile(reference, "<f4")
        if expected.size != native.size:
            raise ValueError("MOSS text reference has wrong extent")
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
    p = sub.add_parser("prepare")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--hidden", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--text-length", type=int, default=17)
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else check(args)


if __name__ == "__main__":
    main()
