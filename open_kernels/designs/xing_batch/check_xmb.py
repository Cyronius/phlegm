r"""Unit fixtures for Xing's block route kernels (OPEN-FAMILY-XING "Block prefill"): every gemm_q4_prefill shape the
route runs (GQP_FMT q4_0) and the xmb streams, on synthetic weights in the recipe's Q4_0-native band law
(designs/xing/check_xg.py make_bands), through the run_kernel harness, against NumPy.

  gemm_n<N>_k<K>   Y [N, T] = W bf16(X)^T, W at byte 0 of the weight buffer, X [T, K] tiled as core.cpp tile_x
  xmb_s<S>         per slot s (compiled at pool expert s, no patch): h = silu(W_g x) * (W_u x), y = W_d bf16(h) for
                   8 tokens a slot; the pool is the moe layer pool (experts from POOL_EXP, down slices from
                   POOL_DOWN_M), eight distinct experts repeated over the slots

    python designs/xing_batch/check_xmb.py --kernels <kernel set dir> --spec-from <model dir> --out <dir>
        [--driver target/npu-dev/harness-build/run_kernel] [--parts gemm,xmb] [--slots 8,16,64]
Run on the NPU through target/npu-dev/npu-run.sh. Exit 1 if any part misses its tolerance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent / "xing"))
from check_xg import bf, make_bands, silu  # noqa: E402
from recipes import xing as XR  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402

T = XR.GEMM_T
NT = XR.MB_NT


def tile_x(x: np.ndarray) -> np.ndarray:
    """[T, K] f32 -> bf16 tiles [K/64][T/32][8 si][4 ti][8 s][8 t] (block_host.cpp tile_x)."""
    t, k = x.shape
    v = x.astype(np.float32).astype(bfloat16).reshape(t // 32, 4, 8, k // 64, 8, 8)    # [nb, ti, t, kb, si, s]
    return np.ascontiguousarray(v.transpose(3, 0, 4, 1, 5, 2))


def x_to_dev(x: np.ndarray) -> np.ndarray:
    """x [slots, NT, K] bf16 -> the A tiles [slots][K/8][NT][8] (lfm2moe_batch/layout.py)."""
    s, n, k = x.shape
    return np.ascontiguousarray(x.reshape(s, n, k // 8, 8).transpose(0, 2, 1, 3))


def y_from_dev(dev: np.ndarray, rows: int) -> np.ndarray:
    """lfm2moe_batch/layout.py y_from_dev at NT 8: [slots][rbg][half][r][gg][p][t][j] -> [slots, NT, rows]."""
    s = dev.size // (rows * NT)
    nb = rows // 64
    d = dev.reshape(s, nb // 4, 2, 4, 2, 2, NT // 8, 8, 8)
    d = d.transpose(0, 1, 3, 2, 4, 5, 6, 7, 8).reshape(s, nb, 4, 2, NT // 8, 8, 8)
    y = d.transpose(0, 4, 5, 1, 2, 6, 3)
    return np.ascontiguousarray(y.reshape(s, NT, rows))


def h_from_dev(dev: np.ndarray, ff: int) -> np.ndarray:
    s = dev.size // (ff * NT)
    d = dev.reshape(s, ff // 8, NT // 8, 8, 8)
    return np.ascontiguousarray(d.transpose(0, 2, 3, 1, 4).reshape(s, NT, ff))


def run(driver: str, out: Path, name: str, cfg: list[str]) -> tuple[int, str]:
    (out / f"{name}.cfg").write_text("\n".join(cfg) + "\n")
    r = subprocess.run([driver, str(out / f"{name}.cfg")], capture_output=True, text=True)
    (out / f"{name}.log").write_text(r.stdout + r.stderr)
    ms = [l.split("(")[-1].rstrip(")") for l in r.stdout.splitlines() if l.startswith("run ")]
    return r.returncode, " ".join(ms[-3:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kernels", required=True, type=Path)
    ap.add_argument("--spec-from", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--driver", default="/home/cyrus/code/phlegm/target/npu-dev/harness-build/run_kernel")
    ap.add_argument("--parts", default="gemm,xmb")
    ap.add_argument("--slots", default="8,16,64")
    a = ap.parse_args()
    spec = spec_from_model_dir(a.spec_from)
    G, L = XR.geometry(spec), XR.layout(spec)
    man = json.loads((a.kernels / "manifest.json").read_text())
    rng = np.random.default_rng(1109)
    a.out.mkdir(parents=True, exist_ok=True)
    ok = True
    results = {}
    if "gemm" in a.parts.split(","):
        ctx = a.kernels / man["contexts"]["gemm"]
        names = sorted(k for k in man["kernels"] if k.startswith("gemm_n"))
        for name in names:
            N, K = (int(v) for v in name[len("gemm_n"):].split("_k"))
            wb, W = make_bands(rng, N, K, 0.02)
            x = rng.standard_normal((T, K)).astype(np.float32)
            wb.tofile(a.out / f"{name}-w.bin")
            tile_x(x).tofile(a.out / f"{name}-x.bin")
            cfg = ["device", f"xclbin g {ctx}", f"kernelx k g {a.kernels / name / 'insts.bin'}",
                   f"buf w {wb.size} {name}-w.bin", f"buf x {T * K * 2} {name}-x.bin", f"buf y {N * T * 4}",
                   "run k w x y", "run k w x y", "run k w x y", f"dump y {name}-y.bin"]
            rc, ms = run(a.driver, a.out, name, cfg)
            if rc:
                print(f"{name}: run_kernel rc {rc}")
                ok = False
                continue
            got = np.fromfile(a.out / f"{name}-y.bin", np.float32).reshape(N, T).astype(np.float64)
            want = W @ bf(x).T
            rel = np.linalg.norm(got - want) / np.linalg.norm(want)
            worst = float(np.max(np.linalg.norm(got - want, axis=0) / np.linalg.norm(want, axis=0)))
            print(f"{name}: rel L2 {rel:.2e}, worst token {worst:.2e}  ({ms})")
            results[name] = {"rel": rel, "worst_token": worst, "ms": ms}
            ok &= bool(rel < 5e-3 and worst < 1e-2)
    if "xmb" in a.parts.split(","):
        ctx = a.kernels / man["contexts"]["xmb"]
        bb = XR.band_bytes(G.HID)
        distinct = 8
        pool = np.zeros(L.POOL_MOE_BYTES, np.uint8)
        Wg, Wu, Wd, ug_bytes, dn_bytes = [], [], [], [], []
        for e in range(distinct):
            gb, g = make_bands(rng, G.FF, G.HID, 0.02)
            ub, u = make_bands(rng, G.FF, G.HID, 0.02)
            db, d = make_bands(rng, G.HID, G.FF, 0.02)
            blob = np.zeros(L.UPS, np.uint8)
            for band in range(G.FF // 64):                         # gate band b at 2 b bb, up band at (2 b + 1) bb
                blob[2 * band * bb:(2 * band + 1) * bb] = gb[band * bb:(band + 1) * bb]
                blob[(2 * band + 1) * bb:(2 * band + 2) * bb] = ub[band * bb:(band + 1) * bb]
            Wg.append(g)
            Wu.append(u)
            Wd.append(d)
            ug_bytes.append(blob)
            dn_bytes.append(db)
        for e in range(G.EXPERTS):
            pool[L.POOL_EXP + e * L.UPS:L.POOL_EXP + (e + 1) * L.UPS] = ug_bytes[e % distinct]
            pool[L.POOL_DOWN_M + e * L.DNS:L.POOL_DOWN_M + e * L.DNS + dn_bytes[e % distinct].size] = dn_bytes[e % distinct]
        pool.tofile(a.out / "xmb-pool.bin")
        for S in (int(v) for v in a.slots.split(",")):
            name = f"xmb_s{S}"
            if name not in man["kernels"]:
                print(f"{name}: not in the kernel set")
                ok = False
                continue
            xs = rng.standard_normal((S, NT, G.HID)).astype(np.float32).astype(bfloat16)
            x_to_dev(xs).tofile(a.out / f"{name}-x.bin")
            cfg = ["device", f"xclbin m {ctx}", f"kernelx k m {a.kernels / name / 'insts.bin'}",
                   f"buf pool {L.POOL_MOE_BYTES} xmb-pool.bin", f"buf x {S * G.HID * NT * 2} {name}-x.bin",
                   f"buf h {S * G.FF * NT * 2}", f"buf y {S * G.HID * NT * 4}", "run k pool x h y", "run k pool x h y",
                   "run k pool x h y", f"dump h {name}-h.bin", f"dump y {name}-y.bin"]
            rc, ms = run(a.driver, a.out, name, cfg)
            if rc:
                print(f"{name}: run_kernel rc {rc}")
                ok = False
                continue
            hd = h_from_dev(np.fromfile(a.out / f"{name}-h.bin", np.uint16).view(bfloat16), G.FF).astype(np.float64)
            yd = y_from_dev(np.fromfile(a.out / f"{name}-y.bin", np.float32), G.HID).astype(np.float64)
            herr, yerr = [], []
            for s in range(S):
                e = s % distinct
                xt = xs[s].astype(np.float64)
                h = silu(xt @ Wg[e].T) * (xt @ Wu[e].T)
                herr.append(np.linalg.norm(hd[s] - h) / np.linalg.norm(h))
                want = hd[s] @ Wd[e].T                              # the down over the kernel's own bf16 h
                yerr.append(np.linalg.norm(yd[s] - want) / np.linalg.norm(want))
            print(f"{name}: h rel L2 max {max(herr):.2e}, y rel L2 max {max(yerr):.2e}  ({ms})")
            results[name] = {"h": max(herr), "y": max(yerr), "ms": ms}
            ok &= bool(max(herr) < 1e-2 and max(yerr) < 5e-3)
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    print("XMB-UNIT OK" if ok else "XMB-UNIT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
