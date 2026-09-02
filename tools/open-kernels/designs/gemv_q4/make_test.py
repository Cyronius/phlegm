r"""Test vectors for gemv_q4: slice a projection out of a captured layer pool,
make an activation, compute the fp64 reference from the SAME pool bytes.

    python make_test.py [--region R] [--expert E] [--x random|ones|onehot:K|act:FILE]

Regions (npu-engine/src/pools.rs offsets; RS = band row split, see gemv_q4.h):
    qkv, z, share_up, share_gate, share_down          standard layout, RS=2
    exp_up, exp_gate     expert E's 4 stripes (128 rows x 2048 each), RS=4
    exp_down             expert E's down [2048, 512], RS=4

Writes w_<tag>.bin (pool-order chunks), x_<tag>.bin (bf16[K]), ref_<tag>.bin
(f32[N]) and run_<tag>.cfg next to this file. The pool blob is FLM's own
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
S = 32 * CH                       # one expert stripe (128 rows x 2048)

REGIONS = {                       # name: (offset, N, K, RS)
    "qkv":        (505_282_560, 8192, 2048, 2),
    "z":          (515_768_320, 4096, 2048, 2),
    "share_up":   (503_316_480, 512, 2048, 2),
    "share_gate": (503_971_840, 512, 2048, 2),
    "share_down": (504_627_200, 2048, 512, 2),
}


def region_bytes(f, region: str, expert: int, bands_cap: int | None):
    """-> (w_bytes, N, K, RS). Expert up/gate stripes are interleaved
    [up0 gate0 up1 gate1 ...] in the pool; concatenate the 4 of one kind."""
    if region in ("exp_up", "exp_gate"):
        parts = []
        for k in range(4):
            f.seek((8 * expert + 2 * k + (1 if region == "exp_gate" else 0)) * S)
            parts.append(f.read(S))
        w = np.frombuffer(b"".join(parts), np.uint8)
        return w, 512, 2048, 4
    if region == "exp_down":
        f.seek(335_544_320 + expert * 655_360)
        return np.frombuffer(f.read(655_360), np.uint8), 2048, 512, 4
    off, n, k, rs = REGIONS[region]
    if bands_cap:
        n = bands_cap * 32 * rs
    nbytes = n * k * CH // (32 * 256)
    f.seek(off)
    w = np.frombuffer(f.read(nbytes), np.uint8)
    assert len(w) == nbytes
    return w, n, k, rs


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


def reference(w_bytes: np.ndarray, x: np.ndarray, n: int, k: int, rs: int) -> np.ndarray:
    per_band = rs * k // 256
    y = np.zeros(n, np.float64)
    xf = x.astype(np.float64)
    nch = len(w_bytes) // CH
    for c in range(nch):
        band, ci = divmod(c, per_band)
        rows0 = 32 * rs * band + 32 * (ci % rs)
        cols0 = 256 * (ci // rs)
        w = dequant_chunk(w_bytes[c * CH:(c + 1) * CH]).astype(np.float64)
        y[rows0:rows0 + 32] += w @ xf[cols0:cols0 + 256]
    return y.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="qkv")
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--x", default="random")
    ap.add_argument("--bands", type=int, default=None, help="cap bands (standard regions)")
    a = ap.parse_args()
    with POOL.open("rb") as f:
        w, n, k, rs = region_bytes(f, a.region, a.expert, a.bands)
    nbytes = len(w)

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

    ref = reference(w, x, n, k, rs)
    tag = a.region if not a.region.startswith("exp_") else f"{a.region}{a.expert}"
    (HERE / f"w_{tag}.bin").write_bytes(w.tobytes())
    (HERE / f"x_{tag}.bin").write_bytes(x.tobytes())
    (HERE / f"ref_{tag}.bin").write_bytes(ref.tobytes())
    d = "C:/code/phlegm/tools/open-kernels/designs/gemv_q4"
    build = a.region  # builds are per shape, shared across experts
    cfg = "\n".join([
        "device",
        f"xclbin G {d}/build_{build}/final.xclbin",
        f"kernelx k G {d}/build_{build}/insts.bin",
        f"buf w {nbytes} {d}/w_{tag}.bin",
        f"buf x {x.nbytes} {d}/x_{tag}.bin",
        f"buf y {ref.nbytes}",
        "run k w x y",
        "run k w x y",
        f"dump y {d}/y_{tag}.bin {ref.nbytes}",
        "",
    ])
    (HERE / f"run_{tag}.cfg").write_text(cfg, newline="\n")
    cores = min(8, n // (32 * rs))
    print(f"{tag}: N={n} K={k} RS={rs} w={nbytes} B x={x.nbytes} B ref={ref.nbytes} B "
          f"ref[:4]={ref[:4]} absmax={np.abs(ref).max():.4g}")
    print(f"build: GEMV_N={n} GEMV_K={k} GEMV_RS={rs} GEMV_CORES={cores} python build_design.py "
          f"designs/gemv_q4/gemv_q4.py designs/gemv_q4/build_{build}")
    print(f"run:   open-qwen-npu npu designs/gemv_q4/run_{tag}.cfg ; python compare.py {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
