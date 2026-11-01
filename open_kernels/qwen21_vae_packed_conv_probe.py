"""Numerically check a packed Qwen VAE convolution against the native NPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.designs.dit_conv.conv_pack import unpack_conv
from open_kernels.designs.dit_conv.dit_conv import resolve_spec


def metadata(weights: Path, kernels: Path, name: str) -> tuple[dict, dict, Path]:
    wm = json.loads((weights / "manifest.json").read_text())
    km = json.loads((kernels / "manifest.json").read_text())
    if not wm.get("complete") or not km.get("complete"):
        raise ValueError("probe needs complete weight and kernel packages")
    record = wm["records"][name]
    stream = km["operations"][name]
    item = km["streams"][stream]
    if record["kernel_spec"] != item["spec"] or record["taps"] ** 2 != item["taps"]:
        raise ValueError("packed weight and compiled convolution geometry differ")
    return record, item, kernels / stream


def prepare(weights: Path, kernels: Path, name: str, out: Path, seed: int) -> None:
    record, item, kernel = metadata(weights, kernels, name)
    out.mkdir(parents=True, exist_ok=False)
    spec = resolve_spec(item["spec"], item["taps"])
    h, w = spec["H"], spec["W"]
    ci, co = spec["Cin"], spec["Cout"]
    xbuf = np.zeros(spec["sizes"]["X"], bfloat16)
    x = xbuf[:(h + 2) * spec["x"]["pitch"] * ci].reshape(h + 2, spec["x"]["pitch"], ci)
    rng = np.random.default_rng(seed)
    x[1:h + 1, 1:w + 1, :record["input_channels"]] = rng.standard_normal(
        (h, w, record["input_channels"]), dtype=np.float32).astype(bfloat16)
    xbuf.tofile(out / "X.bin")
    wfile = weights / record["file"]
    lines = ["device", f"xclbin G {kernel / 'final.xclbin'}",
             f"kernelx k G {kernel / 'insts.bin'}",
             f"buf X {xbuf.nbytes} {out / 'X.bin'}",
             f"buf W {record['bytes']} {wfile}",
             f"buf Y {spec['sizes']['Y'] * 2}", "run k X W Y",
             f"dump Y {out / 'Y.out'} {spec['sizes']['Y'] * 2}", ""]
    (out / "run.cfg").write_text("\n".join(lines))
    (out / "fixture.json").write_text(json.dumps({
        "name": name, "record": record, "stream": item, "seed": seed}, indent=2) + "\n")


def check(weights: Path, out: Path) -> None:
    meta = json.loads((out / "fixture.json").read_text())
    record, item = meta["record"], meta["stream"]
    spec = resolve_spec(item["spec"], item["taps"])
    h, w, ho, wo = (spec[key] for key in ("H", "W", "Ho", "Wo"))
    ci, co = spec["Cin"], spec["Cout"]
    packed = np.fromfile(weights / record["file"], np.uint8)
    ws, bs = unpack_conv(packed, ci, co, taps=item["taps"], up=spec["up"])
    x = np.fromfile(out / "X.bin", bfloat16)[:(h + 2) * spec["x"]["pitch"] * ci]
    x = x.reshape(h + 2, spec["x"]["pitch"], ci).astype(np.float32)
    y = np.fromfile(out / "Y.out", bfloat16)[:(ho + 2) * spec["y"]["pitch"] * co]
    y = y.reshape(ho + 2, spec["y"]["pitch"], co).astype(np.float32)
    sample_y = sorted({0, 1, 2, ho // 2, ho - 3, ho - 2, ho - 1})
    sample_x = sorted({0, 1, 2, wo // 2, wo - 3, wo - 2, wo - 1})
    errors, refs = [], []
    for row in sample_y:
        for col in sample_x:
            if spec["up"]:
                phase = 2 * (row % 2) + (col % 2)
                sy, sx = row // 2, col // 2
            else:
                phase, sy, sx = 0, row, col
            ref = bs[phase].copy()
            taps = 3 if item["taps"] == 9 else 1
            for ky in range(taps):
                for kx in range(taps):
                    iy, ix = (sy + ky, sx + kx) if taps == 3 else (sy + 1, sx + 1)
                    ref += ws[phase, :, :, ky, kx].dot(x[iy, ix])
            errors.append(y[row + 1, col + 1] - ref)
            refs.append(ref)
    diff, ref = np.concatenate(errors), np.concatenate(refs)
    error = float(np.linalg.norm(diff) / np.linalg.norm(ref))
    border = max(float(np.max(np.abs(y[0]))), float(np.max(np.abs(y[-1]))),
                 float(np.max(np.abs(y[:, 0]))), float(np.max(np.abs(y[:, wo + 1]))))
    print(f"{meta['name']}: relative L2 {error:.6g}, border max {border:.6g}")
    if not np.isfinite(y).all() or error > 0.03 or border != 0:
        raise SystemExit("Qwen VAE packed convolution failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--weights", type=Path, required=True)
    prep.add_argument("--kernels", type=Path, required=True)
    prep.add_argument("--name", required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=37)
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("--weights", type=Path, required=True)
    check_cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.weights, args.kernels, args.name, args.out, args.seed)
    else:
        check(args.weights, args.out)


if __name__ == "__main__":
    main()
