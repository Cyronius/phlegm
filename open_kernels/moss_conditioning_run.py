"""Prepare/check one real-weight native MOSS timestep-conditioning pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.moss_conditioning import params, plan
from open_kernels.moss_schedule import timestep_features


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def prepare(args) -> None:
    if not np.isfinite(args.sigma) or not 0 <= args.sigma <= 1:
        raise ValueError("MOSS sigma must be finite and in [0,1]")
    kernels, weights, model = (p.resolve() for p in
                               (args.kernels, args.weights, args.model))
    out = args.out.resolve()
    p = plan()
    km = json.loads((kernels / "manifest.json").read_text())
    wm = json.loads((weights / "packed_weights.json").read_text())
    if not km.get("complete") or not wm.get("full_model") or wm.get("model") != "MOSS-Image-SFT":
        raise ValueError("MOSS conditioning kernels or weight pack incomplete")
    sets = {"gemm": p.streams["gemm"],
            "bias": {k: v for k, v in p.streams["ew"].items() if k.startswith("bias")},
            "silu": {"silu": p.streams["ew"]["silu"]}}
    for kset, specs in sets.items():
        marker = json.loads((kernels / kset / "set.json").read_text())
        if not marker.get("complete") or marker["streams"] != specs:
            raise ValueError(f"MOSS conditioning {kset} streams differ from source")
    out.mkdir(parents=True, exist_ok=False)
    feature = np.zeros(p.buffers["FEATURE"], bfloat16)
    feature[:256] = timestep_features(args.sigma).astype(bfloat16)
    feature.tofile(out / "feature.bf16")
    ps = params(model)
    entries = ["device"]
    for kset, specs in sets.items():
        entries.append(f"xclbin {kset} {kernels / kset / 'final.xclbin'}")
        entries.extend(f"kernelx {name} {kset} {kernels / kset / f'insts_{name}.bin'}"
                       for name in specs)
    for name, elements in p.buffers.items():
        source = f" {out / 'feature.bf16'}" if name == "FEATURE" else ""
        entries.append(f"buf b_{name} {elements * 2}{source}")
    for name, data in ps.items():
        path = out / f"{name}.bf16"
        data.tofile(path)
        entries.append(f"buf p_{name} {data.nbytes} {path}")
    for name in p.streams["gemm"]:
        slot = wm["slots"][name]
        path = weights / slot["file"]
        if path.stat().st_size != slot["bytes"] or digest(path) != slot["sha256"]:
            raise ValueError(f"packed MOSS conditioning weight failed integrity check: {name}")
        entries.append(f"buf w_{name} {slot['bytes']} {path}")
    for op in p.ops:
        names = [("w_" + key) if key in p.streams["gemm"] else
                 ("p_" + key) if key in ps else ("b_" + key)
                 for key in op["args"]]
        entries.append(f"run {op['stream']} {' '.join(names)}")
    for name in ("MOD", "HEAD"):
        entries.append(f"dump b_{name} {out / f'native-{name.lower()}.bf16'} {p.buffers[name] * 2}")
    (out / "run.cfg").write_text("\n".join(entries + [""]))
    (out / "fixture.json").write_text(json.dumps({
        "model": "OpenMOSS-Team/MOSS-Image-SFT",
        "scope": "native timestep conditioning",
        "sigma": args.sigma,
        "operations": len(p.ops),
        "kernel_manifest": str(kernels / "manifest.json"),
        "weight_manifest": str(weights / "packed_weights.json"),
    }, indent=2) + "\n")
    print(f"prepared {len(p.ops)} conditioning NPU operations -> {out / 'run.cfg'}")


def check(args) -> None:
    out = args.out.resolve()
    p = plan()
    for name, width in (("mod", 6 * 3072), ("head", 2 * 3072)):
        native = np.fromfile(out / f"native-{name}.bf16", bfloat16)
        if native.size != p.buffers[name.upper()]:
            raise ValueError(f"native MOSS {name} has wrong extent")
        native = native.reshape(512, width)[0].astype(np.float32)
        if not np.isfinite(native).all():
            raise ValueError(f"native MOSS {name} contains NaN/Inf")
        print(f"native {name}: RMS {np.sqrt(np.mean(native ** 2)):.6g}")
        ref = out / f"reference-{name}.f32"
        if ref.is_file():
            expected = np.fromfile(ref, "<f4")
            if expected.size != width:
                raise ValueError(f"MOSS {name} reference has wrong extent")
            diff = native - expected
            rel = np.linalg.norm(diff) / np.linalg.norm(expected)
            cosine = np.dot(native, expected) / (np.linalg.norm(native) * np.linalg.norm(expected))
            print(f"vs BF16 checkpoint reference: relative L2 {rel:.6g}, "
                  f"cosine {cosine:.9g}, max abs {np.max(np.abs(diff)):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sigma", type=float, default=1.0)
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else check(args)


if __name__ == "__main__":
    main()
