r"""Test vectors for gemv_q4: slice a projection out of a captured layer pool,
make an activation, compute the fp64 reference from the SAME pool bytes.

    python make_test.py [--region qkv|z|share_up|share_gate|share_down] [--x random|ones|onehot:K|act:FILE]

Writes w.bin (pool-order chunks), x.bin (bf16[K]), ref.bin (f32[N]) next to
this file. Pool offsets: npu-engine/src/pools.rs. The pool blob is FLM's own
layer-0 pool as captured (verified byte-exact by pools.rs tests).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
POOL = Path("/mnt/c/caps/m0d/blob_536870912_836fd8e49f35a0b6.bin")
CH = 5120

REGIONS = {                       # name: (offset, N, K)
    "qkv":        (505_282_560, 8192, 2048),
    "z":          (515_768_320, 4096, 2048),
    "share_up":   (503_316_480, 512, 2048),
    "share_gate": (503_971_840, 512, 2048),
    "share_down": (504_627_200, 2048, 512),
}


def dequant_chunk(b: np.ndarray) -> np.ndarray:
    """5120 B pool chunk -> f32 [32 rows, 256 k] (q4nx.rs dequant_q4_bytes)."""
    d = b[0:512].view(bfloat16).astype(np.float32)        # index bc*32 + r
    m = b[512:1024].view(bfloat16).astype(np.float32)
    q = b[1024:]
    nib = np.empty(8192, np.uint8)
    nib[0::2] = q & 0xF
    nib[1::2] = q >> 4
    # nib index p = (r/16)*4096 + k*16 + (r%16)  ->  [rb, k, r16]
    n3 = nib.reshape(2, 256, 16)
    w = np.empty((32, 256), np.float32)
    kb = np.arange(256) // 32                             # k block per column
    for rb in range(2):
        r = rb * 16 + np.arange(16)                       # rows of this block
        codes = n3[rb].T.astype(np.float32)               # [16 rows, 256 k]
        dd = d[kb[None, :] * 32 + r[:, None]]             # [16, 256]
        mm = m[kb[None, :] * 32 + r[:, None]]
        w[r] = codes * dd + mm
    return w


def reference(w_bytes: np.ndarray, x: np.ndarray, n: int, k: int) -> np.ndarray:
    per_band = k // 128
    y = np.zeros(n, np.float64)
    xf = x.astype(np.float64)
    nch = len(w_bytes) // CH
    for c in range(nch):
        band, ci = divmod(c, per_band)
        rows0 = 64 * band + 32 * (ci % 2)
        cols0 = 256 * (ci // 2)
        w = dequant_chunk(w_bytes[c * CH:(c + 1) * CH]).astype(np.float64)
        y[rows0:rows0 + 32] += w @ xf[cols0:cols0 + 256]
    return y.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="qkv")
    ap.add_argument("--x", default="random")
    ap.add_argument("--bands", type=int, default=None, help="cap bands (64 rows each)")
    a = ap.parse_args()
    off, n, k = REGIONS[a.region]
    if a.bands:
        n = a.bands * 64
    nbytes = n * k * CH // (32 * 256)
    with POOL.open("rb") as f:
        f.seek(off)
        w = np.frombuffer(f.read(nbytes), np.uint8)
    assert len(w) == nbytes

    if a.x == "ones":
        x = np.ones(k, np.float32).astype(bfloat16)
    elif a.x.startswith("onehot"):
        x = np.zeros(k, np.float32)
        x[int(a.x.split(":")[1])] = 1.0
        x = x.astype(bfloat16)
    elif a.x.startswith("act:"):
        x = np.fromfile(a.x[4:], np.uint16)[:k].view(bfloat16)
    else:
        x = np.random.default_rng(0).standard_normal(k).astype(np.float32).astype(bfloat16)

    ref = reference(w, x, n, k)
    r = a.region
    (HERE / f"w_{r}.bin").write_bytes(w.tobytes())
    (HERE / f"x_{r}.bin").write_bytes(x.tobytes())
    (HERE / f"ref_{r}.bin").write_bytes(ref.tobytes())
    # Driver config for this region (Windows paths). Two runs: the second is
    # the steady-state timing (first includes nothing extra on this driver, but
    # keeps the comparison honest).
    d = "C:/code/phlegm/tools/open-kernels/designs/gemv_q4"
    cfg = "\n".join([
        "device",
        f"xclbin G {d}/build_{r}/final.xclbin",
        f"kernelx k G {d}/build_{r}/insts.bin",
        f"buf w {nbytes} {d}/w_{r}.bin",
        f"buf x {x.nbytes} {d}/x_{r}.bin",
        f"buf y {ref.nbytes}",
        "run k w x y",
        "run k w x y",
        f"dump y {d}/y_{r}.bin {ref.nbytes}",
        "",
    ])
    (HERE / f"run_{r}.cfg").write_text(cfg, newline="\n")
    cores = min(8, n // 64)
    print(f"{r}: N={n} K={k} w={nbytes} B x={x.nbytes} B ref={ref.nbytes} B "
          f"ref[:4]={ref[:4]} absmax={np.abs(ref).max():.4g}")
    print(f"build: GEMV_N={n} GEMV_K={k} GEMV_CORES={cores} python build_design.py "
          f"designs/gemv_q4/gemv_q4.py designs/gemv_q4/build_{r}")
    print(f"run:   open-qwen-npu npu designs/gemv_q4/run_{r}.cfg ; python compare.py {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
