r"""Unit fixtures for the xg one-image program (OPEN-FAMILY-XING): every stream of the image on synthetic Q4_0-native
weights, through the run_kernel harness, against NumPy.

  qkva / qb / o   one GEMV: y = W bf16(x), x the f32 activation at AD_XF (prepf rounds it to bf16), y at AD_Y / AD_O
  moe (5 slots)   per slot e (compiled at pool expert e, no patch): g / u = W_g,u bf16(xm), h_e = w_e silu(g) u at
  dense (9 slots) AD_H + e H_STRIDE; m = sum_e W_down,e bf16(h_e) at AD_MOE; xm bf16 at AD_XM, the weights f32[16]
                  at AD_W (moe: 4 routed weights then 1; dense: 1 each)

The pool is the layer pool of the moe kind (POOL_BYTES), every region the streams read filled with random chunks in
the recipe's band law (gate | up bands interleaved). Nothing else of the model is needed: the expected values come
from decoding the same chunks (the 4608-B Q4_0-native law of gemv_q4.h / xing_pack_check.cpp).

    python designs/xing/check_xg.py --kernels <kernel set dir> --spec-from <model dir> --out <dir>
        [--driver target/npu-dev/harness-build/run_kernel] [--parts qkva,qb,o,moe,dense]
Run on the NPU through target/npu-dev/npu-run.sh. Exit 1 if any part misses its tolerance.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from recipes import xing as XR  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402
from recipes.qwen36moe import BAND_ROWS  # noqa: E402

CH = 4608


def bf(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float64)


def make_bands(rng, rows: int, K: int, scale: float):
    """rows x K random weights as Q4_0-native chunks in std_perm's band law: (bytes, dense f64 matrix)."""
    bands, per_band = rows // BAND_ROWS, K // 128
    d = bf(np.abs(rng.standard_normal((rows, K // 32))) * scale + scale * 0.1)      # bf16 scales
    q = rng.integers(0, 16, size=(rows, K), dtype=np.uint8)
    W = np.repeat(d, 32, axis=1) * (q.astype(np.float64) - 8)
    out = np.zeros(bands * per_band * CH, np.uint8)
    d16 = d.astype(np.float32).astype(bfloat16).view(np.uint16)
    for b in range(bands):
        for i in range(per_band):
            part, kt = i % 2, i // 2
            c = out[(b * per_band + i) * CH:(b * per_band + i + 1) * CH]
            r0 = 64 * b + 32 * part
            dd = d16[r0:r0 + 32, kt * 8:kt * 8 + 8]                                    # [32 rows][8 kb]
            c[:512] = np.ascontiguousarray(dd.T).view(np.uint8).reshape(-1)           # d at 2 (kb 32 + r)
            qq = q[r0:r0 + 32, kt * 256:kt * 256 + 256]                               # [32][256]
            p = (np.arange(32)[:, None] // 16) * 4096 + np.arange(256)[None, :] * 16 + np.arange(32)[:, None] % 16
            nib = np.zeros(8192, np.uint8)
            nib[p.reshape(-1)] = qq.reshape(-1)
            c[512:] = nib[0::2] | (nib[1::2] << 4)
    return out, W


def silu(x):
    return x / (1.0 + np.exp(-x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kernels", required=True, type=Path)
    ap.add_argument("--spec-from", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--driver", default="/home/cyrus/code/phlegm/target/npu-dev/harness-build/run_kernel")
    ap.add_argument("--parts", default="qkva,qb,o,moe,dense")
    a = ap.parse_args()
    spec = spec_from_model_dir(a.spec_from)
    G, L = XR.geometry(spec), XR.layout(spec)
    rng = np.random.default_rng(1108)
    a.out.mkdir(parents=True, exist_ok=True)
    pool = np.zeros(L.POOL_BYTES, np.uint8)
    act = np.zeros(L.AD_BYTES, np.uint8)
    sh = XR.gemv_shapes(spec)
    W = {}
    for p, s in sh.items():
        b, W[p] = make_bands(rng, s["n"], s["k"], 0.02)
        pool[s["pool"]:s["pool"] + b.size] = b
    bb = XR.band_bytes(G.HID)
    nexp = max(G.MSLOTS, G.DSLOTS)
    Wg, Wu, Wd_m, Wd_d = [], [], [], []
    for e in range(nexp):
        gb, g = make_bands(rng, G.FF, G.HID, 0.02)
        ub, u = make_bands(rng, G.FF, G.HID, 0.02)
        Wg.append(g)
        Wu.append(u)
        base = L.POOL_EXP + e * L.UPS
        for band in range(G.FF // BAND_ROWS):                      # gate band b at 2 b bb, up band at (2 b + 1) bb
            pool[base + 2 * band * bb:base + (2 * band + 1) * bb] = gb[band * bb:(band + 1) * bb]
            pool[base + (2 * band + 1) * bb:base + (2 * band + 2) * bb] = ub[band * bb:(band + 1) * bb]
        for pd, lst in ((L.POOL_DOWN_M, Wd_m), (L.POOL_DOWN_D, Wd_d)):
            db, dm = make_bands(rng, G.HID, G.FF, 0.02)
            if (pd == L.POOL_DOWN_M and e < G.MSLOTS) or (pd == L.POOL_DOWN_D and e < G.DSLOTS):
                pool[pd + e * L.DNS:pd + e * L.DNS + db.size] = db
            lst.append(dm)
    pool.tofile(a.out / "pool.bin")
    xf = rng.standard_normal(max(G.HID, G.QL, G.O_K)).astype(np.float32)
    act[L.AD_XF:L.AD_XF + xf.size * 4] = xf.view(np.uint8)
    xm = rng.standard_normal(G.HID).astype(np.float32)
    act[L.AD_XM:L.AD_XM + G.HID * 2] = xm.astype(bfloat16).view(np.uint8)
    xm64 = bf(xm)
    ok = True
    ctx = a.kernels / "xg_o" / "final.xclbin"
    for part in a.parts.split(","):
        wv = np.zeros(16, np.float32)
        if part == "moe":
            wv[:G.MSLOTS] = [0.7, 0.5, 0.45, 0.35, 1.0][:G.MSLOTS]
        elif part == "dense":
            wv[:G.DSLOTS] = 1.0
        a_act = act.copy()
        a_act[L.AD_W:L.AD_W + 64] = wv.view(np.uint8)
        a_act.tofile(a.out / f"act-{part}.bin")
        cfg = [f"device", f"xclbin xg {ctx}", f"kernelx k xg {a.kernels / ('xg_' + part) / 'insts.bin'}",
               f"buf pool {L.POOL_BYTES} pool.bin", f"buf act {L.AD_BYTES} act-{part}.bin", "run k pool act",
               f"dump act act-{part}-out.bin"]
        (a.out / f"{part}.cfg").write_text("\n".join(cfg) + "\n")
        r = subprocess.run([a.driver, str(a.out / f"{part}.cfg")], capture_output=True, text=True)
        (a.out / f"{part}.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            print(f"{part}: run_kernel rc {r.returncode}\n{(r.stdout + r.stderr)[-800:]}")
            ok = False
            continue
        o = np.fromfile(a.out / f"act-{part}-out.bin", np.uint8)
        if part in sh:
            s = sh[part]
            got = o[s["ay"]:s["ay"] + s["n"] * 4].view(np.float32).astype(np.float64)
            want = W[part] @ bf(xf[:s["k"]])
            tol = 2e-3
        else:
            S = G.MSLOTS if part == "moe" else G.DSLOTS
            Wd = Wd_m if part == "moe" else Wd_d
            want = np.zeros(G.HID)
            hs_err = []
            for e in range(S):
                h = wv[e] * silu(Wg[e] @ xm64) * (Wu[e] @ xm64)
                hg = o[L.AD_H + e * L.H_STRIDE:L.AD_H + e * L.H_STRIDE + G.FF * 4].view(np.float32).astype(np.float64)
                hs_err.append(np.linalg.norm(hg - h) / np.linalg.norm(h))
                want += Wd[e] @ bf(hg)                                 # down over the NPU's own h (bf16 in prepf)
            got = o[L.AD_MOE:L.AD_MOE + G.HID * 4].view(np.float32).astype(np.float64)
            print(f"{part}: h per slot rel L2 max {max(hs_err):.2e}")
            ok &= max(hs_err) < 5e-3
            tol = 2e-3
        rel = np.linalg.norm(got - want) / np.linalg.norm(want)
        ms = [l for l in r.stdout.splitlines() if "run" in l.lower()][-1:] if r.stdout else []
        print(f"{part}: rel L2 {rel:.2e} (tol {tol:g}), max |d| {np.abs(got - want).max():.3e}  {' '.join(ms)}")
        ok &= bool(rel < tol)
    print("XG-UNIT OK" if ok else "XG-UNIT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
