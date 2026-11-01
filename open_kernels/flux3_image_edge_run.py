"""Prepare/check FLUX 3 native image/text entry and final projection fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.flux3_image_edges import final_norm_params, specs
from open_kernels.flux3_weight_layout import HIDDEN


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
    if (not km.get("complete") or km["resolution"] != args.resolution or
            km["text_length"] != args.text_length or
            not wm.get("full_model") or wm["model"] != "flux-3-image-klein"):
        raise ValueError("FLUX 3 edge kernel/weight manifests differ")
    sets = specs(args.resolution, args.text_length)
    for kset, streams in sets.items():
        marker = json.loads((kernels / kset / "set.json").read_text())
        if not marker.get("complete") or marker["streams"] != streams:
            raise ValueError(f"FLUX 3 {kset} edge streams differ from source")
    image = (args.resolution // 16) ** 2
    text = (args.text_length + 511) // 512 * 512
    stage = args.stage
    if stage in ("image-entry", "cond-entry"):
        # A reference image's entry has the generated image's shape; it runs the
        # same compiled stream with the checkpoint's emb_in.image_cond weight.
        source_size, result_size, live = image * 128 + 512, image * HIDDEN, HIDDEN
        slot = "emb.image" if stage == "image-entry" else "emb.image_cond"
    elif stage == "text-entry":
        source_size, result_size, live, slot = text * 20480, text * HIDDEN, HIDDEN, "emb.text"
    else:
        source_size, result_size, live, slot = image * HIDDEN, image * 1024, 128, "final.image"
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    source = np.zeros(source_size, bfloat16)
    active = ((image * 128) if stage in ("image-entry", "cond-entry") else
              (args.text_length * 20480) if stage == "text-entry" else
              source_size)
    if args.input_bf16:
        data = np.fromfile(args.input_bf16.resolve(), bfloat16)
        if data.size < active:
            raise ValueError("FLUX 3 edge input is shorter than the live extent")
        source[:active] = data[:active]
    else:
        source[:active] = rng.normal(0, 0.6, active).astype(bfloat16)
    source.tofile(out / "input.bf16")
    entry = wm["slots"][slot]
    packed = weights / entry["file"]
    if packed.stat().st_size != entry["bytes"] or digest(packed) != entry["sha256"]:
        raise ValueError(f"FLUX 3 packed edge weight failed integrity: {slot}")
    stream = "emb.image" if stage == "cond-entry" else slot
    cfg = ["device", f"xclbin gemm {kernels / 'gemm' / 'final.xclbin'}",
           f"kernelx {stream} gemm {kernels / 'gemm' / f'insts_{stream}.bin'}",
           f"buf IN {source.nbytes} {out / 'input.bf16'}",
           f"buf W {entry['bytes']} {packed}",
           f"buf OUT {result_size * 2}"]
    if stage == "final-exit":
        if args.modulation_bf16:
            mod = np.fromfile(args.modulation_bf16.resolve(), bfloat16)
            if mod.size != 2 * HIDDEN:
                raise ValueError("FLUX 3 final modulation has wrong extent")
            mod = mod.reshape(2, HIDDEN)
        else:
            mod = rng.normal(0, 0.1, (2, HIDDEN)).astype(bfloat16)
        mod.tofile(out / "modulation.bf16")
        params = final_norm_params(mod[0], mod[1])
        params.tofile(out / "norm-params.bf16")
        cfg += [f"xclbin ew {kernels / 'ew' / 'final.xclbin'}",
                f"kernelx final_norm ew {kernels / 'ew' / 'insts_final_norm.bin'}",
                f"buf P {params.nbytes} {out / 'norm-params.bf16'}",
                f"buf DUMMY {HIDDEN * 2}",
                f"buf NORM {source.nbytes}",
                "run final_norm IN DUMMY P NORM DUMMY",
                f"run {slot} NORM W OUT"]
    else:
        cfg.append(f"run {stream} IN W OUT")
    cfg += [f"dump OUT {out / 'native-output.bf16'} {result_size * 2}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    (out / "fixture.json").write_text(json.dumps({
        "model": "black-forest-labs/flux-3-image-klein", "stage": stage,
        "resolution": args.resolution, "text_length": args.text_length,
        "seed": args.seed, "live_rows": args.text_length if stage == "text-entry" else image,
        "padded_rows": text if stage == "text-entry" else image,
        "live_channels": live, "slot": slot,
        "source_extent": active, "output_extent": result_size,
    }, indent=2) + "\n")
    print(f"prepared FLUX 3 {stage} -> {out / 'run.cfg'}")


def check(args) -> None:
    out = args.out.resolve()
    meta = json.loads((out / "fixture.json").read_text())
    raw = np.fromfile(out / "native-output.bf16", bfloat16)
    if raw.size != meta["output_extent"]:
        raise ValueError("FLUX 3 edge output has wrong extent")
    n = raw.reshape(meta["padded_rows"], -1)[:meta["live_rows"], :meta["live_channels"]]
    native = n.astype(np.float32)
    if not np.isfinite(native).all():
        raise ValueError("FLUX 3 edge output contains NaN/Inf")
    print(f"native FLUX 3 {meta['stage']}: RMS {np.sqrt(np.mean(native*native)):.6g}")
    ref = out / "reference-output.f32"
    if ref.is_file():
        expected = np.fromfile(ref, "<f4")
        if expected.size != native.size:
            raise ValueError("FLUX 3 edge reference has wrong extent")
        expected = expected.reshape(native.shape)
        diff = native - expected
        print(f"vs pinned BF16 reference: relative L2 "
              f"{np.linalg.norm(diff)/np.linalg.norm(expected):.6g}, cosine "
              f"{np.vdot(native,expected)/(np.linalg.norm(native)*np.linalg.norm(expected)):.9g}, "
              f"max abs {np.max(np.abs(diff)):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=("image-entry", "cond-entry", "text-entry", "final-exit"),
                   required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--text-length", type=int, default=80)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--input-bf16", type=Path)
    p.add_argument("--modulation-bf16", type=Path)
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else check(args)


if __name__ == "__main__":
    main()
