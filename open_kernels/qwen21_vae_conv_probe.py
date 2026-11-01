"""Probe the first Qwen Image 2.1 VAE decoder convolution on the native NPU."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.image_safetensors import read_header
from open_kernels.designs.dit_conv.conv_pack import pack_conv, unpack_conv
from open_kernels.designs.dit_conv.dit_conv import resolve_spec

HEIGHT = WIDTH = 32
CIN, COUT = 64, 1152


def tensor(path: Path, name: str) -> np.ndarray:
    header = read_header(path)
    item = header[name]
    if item["dtype"] != "F32":
        raise ValueError(f"{name}: expected FP32 VAE tensor")
    with path.open("rb") as stream:
        start = 8 + struct.unpack("<Q", stream.read(8))[0] + item["data_offsets"][0]
    return np.memmap(path, mode="r", dtype="<f4", offset=start,
                     shape=tuple(item["shape"]))


def prepare(model: Path, kernel: Path, out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=False)
    spec = {"H": HEIGHT, "W": WIDTH, "Cin": CIN, "Cout": COUT, "up": False}
    shape = resolve_spec(spec, 9)
    weight_file = model / "vae/diffusion_pytorch_model.safetensors"
    w = tensor(weight_file, "decoder.conv_in.weight")
    b = tensor(weight_file, "decoder.conv_in.bias")
    if w.shape != (COUT, CIN, 3, 3) or b.shape != (COUT,):
        raise ValueError("Qwen VAE first decoder convolution shape changed")
    packed = pack_conv(w, b, taps=9, up=False, cin=CIN)
    if packed.size != shape["sizes"]["W"] * 9:
        raise ValueError("packed convolution byte count differs from NPU stream")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((HEIGHT, WIDTH, CIN), dtype=np.float32).astype(bfloat16)
    xbuf = np.zeros(shape["sizes"]["X"], bfloat16)
    view = xbuf[:(HEIGHT + 2) * shape["x"]["pitch"] * CIN].reshape(
        HEIGHT + 2, shape["x"]["pitch"], CIN)
    view[1:HEIGHT + 1, 1:WIDTH + 1] = x
    xbuf.tofile(out / "X.bin")
    packed.tofile(out / "W.bin")
    (out / "spec.json").write_text(json.dumps(spec) + "\n")
    cfg = ["device", f"xclbin G {kernel / 'final.xclbin'}",
           f"kernelx k G {kernel / 'insts.bin'}",
           f"buf X {xbuf.nbytes} X.bin", f"buf W {packed.nbytes} W.bin",
           f"buf Y {shape['sizes']['Y'] * 2}", "run k X W Y",
           f"dump Y Y.out {shape['sizes']['Y'] * 2}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    (out / "fixture.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "stage": "decoder.conv_in",
        "seed": seed, "resolution": 512}, indent=2) + "\n")


def check(out: Path) -> None:
    spec = json.loads((out / "spec.json").read_text())
    shape = resolve_spec(spec, 9)
    packed = np.fromfile(out / "W.bin", np.uint8)
    weights, biases = unpack_conv(packed, CIN, COUT, taps=9, up=False)
    w, b = weights[0].astype(np.float32), biases[0].astype(np.float32)
    x = np.fromfile(out / "X.bin", bfloat16)[:(HEIGHT + 2) * shape["x"]["pitch"] * CIN].reshape(
        HEIGHT + 2, shape["x"]["pitch"], CIN).astype(np.float32)
    y = np.fromfile(out / "Y.out", bfloat16)[:(HEIGHT + 2) * shape["y"]["pitch"] * COUT].reshape(
        HEIGHT + 2, shape["y"]["pitch"], COUT).astype(np.float32)
    rows = np.array([0, 1, 15, 16, 30, 31])
    ref = np.broadcast_to(b, (len(rows), WIDTH, COUT)).copy()
    for ky in range(3):
        for kx in range(3):
            ref += x[rows[:, None] + ky, np.arange(WIDTH)[None] + kx].reshape(
                len(rows) * WIDTH, CIN).dot(w[:, :, ky, kx].T).reshape(ref.shape)
    got = y[rows + 1, 1:WIDTH + 1]
    error = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    border = max(float(np.max(np.abs(y[0]))), float(np.max(np.abs(y[-1]))),
                 float(np.max(np.abs(y[:, 0]))),
                 float(np.max(np.abs(y[:, WIDTH + 1]))))
    print(f"Qwen VAE decoder.conv_in: relative L2 {error:.6g}, border max {border:.6g}")
    if not np.isfinite(got).all() or error > 0.03 or border != 0:
        raise SystemExit("Qwen VAE decoder first convolution failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model", type=Path, required=True)
    prep.add_argument("--kernel", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=37)
    validate = sub.add_parser("check")
    validate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.kernel, args.out, args.seed) if args.stage == "prepare" else check(args.out)


if __name__ == "__main__":
    main()
