"""Prepare/check native FLUX 3 flow-map timestep and mode modulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.flux3_conditioning import EL, GEMMS, N, add_params, input_rows, specs


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(args) -> None:
    kernels, weights, out = (p.resolve() for p in
                             (args.kernels, args.weights, args.out))
    km = json.loads((kernels / "manifest.json").read_text())
    wm = json.loads((weights / "packed_weights.json").read_text())
    if (not km.get("complete") or km["family"] != "black-forest-labs/flux-3-image-klein" or
            not wm.get("full_model") or wm["model"] != "flux-3-image-klein"):
        raise ValueError("FLUX 3 conditioning kernel/weight manifests differ")
    sets = specs()
    for kset, streams in sets.items():
        marker = json.loads((kernels / kset / "set.json").read_text())
        if not marker.get("complete") or marker["streams"] != streams:
            raise ValueError(f"FLUX 3 {kset} conditioning streams differ from source")
    out.mkdir(parents=True, exist_ok=False)
    current, target = input_rows(args.time, args.target)
    current.tofile(out / "current.bf16")
    target.tofile(out / "target.bf16")
    params = add_params()
    params.tofile(out / "add-params.bf16")
    entries = ["device"]
    for kset, streams in sets.items():
        entries.append(f"xclbin {kset} {kernels / kset / 'final.xclbin'}")
        entries.extend(f"kernelx {name} {kset} {kernels / kset / f'insts_{name}.bin'}"
                       for name in streams)
    entries += [f"buf b_CURRENT {current.nbytes} {out / 'current.bf16'}",
                f"buf b_TARGET {target.nbytes} {out / 'target.bf16'}",
                f"buf b_P {params.nbytes} {out / 'add-params.bf16'}",
                f"buf b_DUMMY {EL * 2}"]
    sizes = {"TIME_MID": N * EL, "TIME_ACT": N * EL,
             "TIME_VEC": N * EL, "REF_MID": N * EL,
             "REF_ACT": N * EL, "REF_VEC": N * EL,
             "VEC": N * EL, "MOD_IN": N * EL,
             "EARLY_IMAGE": N * 3 * EL, "EARLY_TEXT": N * 3 * EL,
             "SINGLE_IMAGE": N * 3 * EL, "SINGLE_TEXT": N * 3 * EL,
             "FINAL": N * 2 * EL}
    if args.reference:
        sizes |= {"EARLY_IMAGE_COND": N * 3 * EL, "SINGLE_IMAGE_COND": N * 3 * EL}
    entries += [f"buf b_{name} {size * 2}" for name, size in sizes.items()]
    # Reference-image modulation has the same shape as the generated image's;
    # its compiled streams run with the checkpoint's image_cond weights.
    reference_slots = ("early.mod.image_cond", "single.mod.image_cond") if args.reference else ()
    for name in GEMMS + reference_slots:
        slot = wm["slots"][name]
        path = weights / slot["file"]
        if path.stat().st_size != slot["bytes"] or digest(path) != slot["sha256"]:
            raise ValueError(f"FLUX 3 conditioning weight failed integrity: {name}")
        entries.append(f"buf w_{name} {slot['bytes']} {path}")
    entries += [
        "run time.in b_CURRENT w_time.in b_TIME_MID",
        "run silu b_TIME_MID b_DUMMY b_DUMMY b_TIME_ACT b_DUMMY",
        "run time.out b_TIME_ACT w_time.out b_TIME_VEC",
        "run ref_time.in b_TARGET w_ref_time.in b_REF_MID",
        "run silu b_REF_MID b_DUMMY b_DUMMY b_REF_ACT b_DUMMY",
        "run ref_time.out b_REF_ACT w_ref_time.out b_REF_VEC",
        "run add b_TIME_VEC b_REF_VEC b_P b_VEC b_DUMMY",
        "run silu b_VEC b_DUMMY b_DUMMY b_MOD_IN b_DUMMY",
    ]
    for slot, buffer in (("early.mod.image", "EARLY_IMAGE"),
                         ("early.mod.txt", "EARLY_TEXT"),
                         ("single.mod.image", "SINGLE_IMAGE"),
                         ("single.mod.txt", "SINGLE_TEXT"),
                         ("final.mod", "FINAL")):
        entries.append(f"run {slot} b_MOD_IN w_{slot} b_{buffer}")
    for slot, buffer in (("early.mod.image_cond", "EARLY_IMAGE_COND"),
                         ("single.mod.image_cond", "SINGLE_IMAGE_COND"))[:len(reference_slots)]:
        entries.append(f"run {slot.removesuffix('_cond')} b_MOD_IN w_{slot} b_{buffer}")
    for name in ("VEC", "EARLY_IMAGE", "EARLY_TEXT", "SINGLE_IMAGE",
                 "SINGLE_TEXT", "FINAL", *(("EARLY_IMAGE_COND", "SINGLE_IMAGE_COND")
                                           if args.reference else ())):
        width = EL if name == "VEC" else 2 * EL if name == "FINAL" else 3 * EL
        entries.append(f"dump b_{name} {out / f'native-{name.lower()}.bf16'} {2 * width * 2}")
    (out / "run.cfg").write_text("\n".join(entries + [""]))
    (out / "fixture.json").write_text(json.dumps({
        "model": "black-forest-labs/flux-3-image-klein",
        "scope": "native flow-map timestep and modulation",
        "image_time": args.time, "target_time": args.target,
        "reference": bool(args.reference),
        "operations": 15 if args.reference else 13,
    }, indent=2) + "\n")
    print(f"prepared FLUX 3 conditioning -> {out / 'run.cfg'}")


def check(args) -> None:
    out = args.out.resolve()
    reference = json.loads((out / "fixture.json").read_text()).get("reference", False)
    for name in ("vec", "early_image", "early_text", "single_image",
                 "single_text", "final",
                 *(("early_image_cond", "single_image_cond") if reference else ())):
        width = EL if name == "vec" else 2 * EL if name == "final" else 3 * EL
        native = np.fromfile(out / f"native-{name}.bf16", bfloat16)
        if native.size != 2 * width:
            raise ValueError(f"FLUX 3 {name} native output has wrong extent")
        native = native.reshape(2, width).astype(np.float32)
        if not np.isfinite(native).all():
            raise ValueError(f"FLUX 3 {name} native output contains NaN/Inf")
        # Row 0 carries the generated image's time; row 1 is t=0 (text and
        # reference images) plus the shared flow-map target time.
        mode_row = 1 if name.endswith(("text", "image_cond")) else 0
        print(f"native {name}: RMS {np.sqrt(np.mean(native[mode_row]**2)):.6g}")
        ref = out / f"reference-{name}.f32"
        if ref.is_file():
            expected = np.fromfile(ref, "<f4").reshape(2, width)
            diff = native[mode_row] - expected[mode_row]
            print(f"  vs published BF16: relative L2 "
                  f"{np.linalg.norm(diff)/np.linalg.norm(expected[mode_row]):.6g}, "
                  f"cosine {np.vdot(native[mode_row],expected[mode_row])/(np.linalg.norm(native[mode_row])*np.linalg.norm(expected[mode_row])):.9g}")
        if name in ("early_image", "early_text", "single_image", "single_text",
                    "early_image_cond", "single_image_cond"):
            native[mode_row].astype(bfloat16).tofile(out / f"modulation-{name}.bf16")
        elif name == "final":
            native[0].astype(bfloat16).tofile(out / "modulation-final.bf16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--time", type=float, default=0.8)
    p.add_argument("--target", type=float, default=0.7)
    p.add_argument("--reference", action="store_true",
                   help="also project the reference-image (image_cond) modulation")
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else check(args)


if __name__ == "__main__":
    main()
