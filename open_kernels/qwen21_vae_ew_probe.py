"""Check the Qwen VAE native pixelwise RMSNorm/SiLU and residual-add stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def prepare(kernel: Path, out: Path, op: str, h: int, channels: int,
            logical: int, el: int, seed: int, gamma_range=(0.7, 1.3)) -> None:
    out.mkdir(parents=True, exist_ok=False)
    if logical > channels or el % channels or el % 256:
        raise ValueError("invalid physical/logical channels or element size")
    rng = np.random.default_rng(seed)
    shape = (h + 2, h + 2, channels)
    x = np.zeros(shape, bfloat16)
    b = np.zeros(shape, bfloat16)
    x[1:-1, 1:-1, :logical] = rng.standard_normal((h, h, logical)).astype(bfloat16)
    b[1:-1, 1:-1, :logical] = rng.standard_normal((h, h, logical)).astype(bfloat16)
    gamma = rng.uniform(*gamma_range, logical).astype(np.float32)
    p = np.zeros((2, el), bfloat16)
    p[:, :logical] = (gamma * np.sqrt(logical / channels)).astype(bfloat16)
    x.tofile(out / "X.bin")
    b.tofile(out / "B.bin")
    p.tofile(out / "P.bin")
    gamma.tofile(out / "gamma.f32")
    cfg = ["device", f"xclbin G {kernel / 'final.xclbin'}",
           f"kernelx k G {kernel / 'insts.bin'}",
           f"buf X {x.nbytes} {out / 'X.bin'}",
           f"buf B {b.nbytes} {out / 'B.bin'}",
           f"buf P {p.nbytes} {out / 'P.bin'}",
           f"buf Y {x.nbytes}", "run k X B P Y",
           f"dump Y {out / 'Y.out'} {x.nbytes}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    (out / "fixture.json").write_text(json.dumps({
        "op": op, "H": h, "C": channels, "logical": logical, "el": el,
        "seed": seed}, indent=2) + "\n")


def check(out: Path) -> None:
    m = json.loads((out / "fixture.json").read_text())
    h, c, logical, op = m["H"], m["C"], m["logical"], m["op"]
    shape = (h + 2, h + 2, c)
    x = np.fromfile(out / "X.bin", bfloat16).reshape(shape).astype(np.float32)
    b = np.fromfile(out / "B.bin", bfloat16).reshape(shape).astype(np.float32)
    got = np.fromfile(out / "Y.out", bfloat16).reshape(shape).astype(np.float32)
    if op == "add":
        ref = x[1:-1, 1:-1, :logical] + b[1:-1, 1:-1, :logical]
    else:
        a = x[1:-1, 1:-1, :logical]
        gamma = np.fromfile(out / "gamma.f32", "<f4")
        ref = a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + 1e-12)
        ref *= gamma
        if op == "rms_silu":
            ref *= 1 / (1 + np.exp(-ref))
    actual = got[1:-1, 1:-1, :logical]
    error = float(np.linalg.norm(actual - ref) / np.linalg.norm(ref))
    border = max(float(np.max(np.abs(got[0]))), float(np.max(np.abs(got[-1]))),
                 float(np.max(np.abs(got[:, 0]))), float(np.max(np.abs(got[:, -1]))))
    tail = float(np.max(np.abs(got[1:-1, 1:-1, logical:]))) if logical < c else 0.0
    print(f"Qwen VAE {op} C={logical}/{c} H={h}: relL2={error:.6g}, "
          f"border={border:.6g}, tail={tail:.6g}")
    if not np.isfinite(got).all() or error > 0.03 or border or tail:
        raise SystemExit("Qwen VAE elementwise numeric gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--kernel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--op", choices=("rms", "rms_silu", "add"), default="rms_silu")
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--channels", type=int, default=1152)
    p.add_argument("--logical", type=int, default=1152)
    p.add_argument("--el", type=int, default=2304)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--gamma", type=float, nargs=2, default=(0.7, 1.3),
                   help="uniform gamma range (the encoder's are mostly 0..0.5)")
    p = sub.add_parser("check")
    p.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "prepare":
        prepare(a.kernel, a.out, a.op, a.height, a.channels, a.logical, a.el, a.seed,
                tuple(a.gamma))
    else:
        check(a.out)


if __name__ == "__main__":
    main()
