"""Test vectors for ln: random x/add, real input_layernorm weight from the captured
L0 pack (C:/caps/m0d/000118.bo @0, bf16[2048]); fp64 reference."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
PACK = Path("/mnt/c/caps/m0d/000118.bo")
N = 2048


def main() -> int:
    w = np.fromfile(PACK, np.uint8)[:N * 2].view(bfloat16).copy()
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(N) * 0.5).astype(np.float32)
    add = (rng.standard_normal(N) * 0.5).astype(np.float32)
    y = (x.astype(np.float64) + add.astype(np.float64))
    xn = (y / np.sqrt((y ** 2).mean() + 1e-6) * w.astype(np.float64)).astype(np.float32).astype(bfloat16)
    (HERE / "x.bin").write_bytes(x.tobytes())
    (HERE / "add.bin").write_bytes(add.tobytes())
    (HERE / "w.bin").write_bytes(w.tobytes())
    (HERE / "ref_y.bin").write_bytes(y.astype(np.float32).tobytes())
    (HERE / "ref_xn.bin").write_bytes(xn.tobytes())
    d = "C:/code/phlegm/tools/open-kernels/designs/ln"
    cfg = "\n".join([
        "device",
        f"xclbin G {d}/build/final.xclbin",
        f"kernelx k G {d}/build/insts.bin",
        f"buf x {x.nbytes} {d}/x.bin",
        f"buf add {add.nbytes} {d}/add.bin",
        f"buf w {w.nbytes} {d}/w.bin",
        f"buf y {x.nbytes}",
        f"buf xn {w.nbytes}",
        "run k x add w y xn",
        "run k x add w y xn",
        f"dump y {d}/y_y.bin {x.nbytes}",
        f"dump xn {d}/y_xn.bin {w.nbytes}",
        "",
    ])
    (HERE / "run.cfg").write_text(cfg, newline="\n")
    print(f"xn[:4]={xn[:4].astype(np.float32)} rms={np.sqrt((y**2).mean()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
