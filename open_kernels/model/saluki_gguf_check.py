r"""Check the Qwen3.5 dense all-q8 GGUF view's packed bytes (OPEN-FAMILY-SALUKI) against an independent
reading of the same GGUF: llama.cpp's own gguf-py dequantizes every tensor (`gguf.quants.dequantize`, the
i-quants included), NumPy applies the conversion's inverse maps (the per-head [q | gate] split, the
linear-attention value-head untile, the conv / alpha / beta transposes), and the packed q8 half-tiles,
q4_1 split halves and q8 head chunks are decoded back to matrices through their band laws.

A tensor whose GGUF type has an integer law the q8 law keeps (IQ2_XXS, IQ3_XXS, IQ3_S, IQ1_S, IQ4_XS,
Q8_0, Q4_0) must agree up to the bf16 rounding of its scale (EXACT_LIMIT); the others (two scales or a
min per 32 values: IQ2_XS, IQ2_S, IQ1_M, Q2_K, Q4_K, ...) are requantized to int8 and report their
error against REQUANT_LIMIT. Each block-route weight must equal the sum of its hi / lo split exactly
the decode pool's q8 values (bit for bit after decoding both).

    q35_pack_dump manifest.json model.gguf OUT 0,3 head
    python open_kernels/model/saluki_gguf_check.py --gguf model.gguf --manifest manifest.json \
        --dump OUT --layers 0,3 --head --gguf-py <llama.cpp>/gguf-py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35moe_gguf_check import Gguf, bf16, from_bf16, q41_chunks, q8_chunks, stats, untile  # noqa: E402

LAW = {"IQ2_XXS", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ4_XS", "Q8_0", "Q4_0"}
EXACT_LIMIT = 2.5e-3     # bf16 RNE of one scale a 32-block: <= 2^-9 relative, ~1e-3 RMS
REQUANT_LIMIT = 1.5e-2   # int8 requantization of a 2- to 4-bit block (amax / 127 steps)
FLOATS = {"F32", "F16", "BF16"}


def q8_half_tiles(buf: np.ndarray) -> np.ndarray:
    """[n, 5120] q8 half-tiles (gemv_q8.h) -> [n, 16, 256] values."""
    n = buf.shape[0]
    s = from_bf16(buf[:, :256].copy().view(np.uint16)).reshape(n, 8, 16).transpose(0, 2, 1)   # [n, r, kb]
    codes = buf[:, 256:256 + 4096].view(np.int8).reshape(n, 256, 16).transpose(0, 2, 1)       # [n, r, k]
    return codes.astype(np.float32) * np.repeat(s, 32, axis=2)


def q8_perm_inverse(tiles: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    """pools.cpp q8_perm: pool half-tile c -> file chunk (2 band + part / 2) * ncol + kt, half part % 2."""
    ncol, per_band = in_dim // 256, in_dim // 64
    w = np.zeros((out_dim, in_dim), np.float32)
    for c in range(tiles.shape[0]):
        band, cc = divmod(c, per_band)
        part, kt = cc % 4, cc // 4
        f, half = (2 * band + part // 2) * ncol + kt, part % 2
        r0, k0 = 32 * (f // ncol) + 16 * half, 256 * (f % ncol)
        w[r0:r0 + 16, k0:k0 + 256] = tiles[c]
    return w


def std_perm_inverse(chunks: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    """pools.cpp std_perm: pool chunk c holds rows 64 (c // per_band) + 32 (c % 2), k-tile (c % per_band) // 2."""
    ncol, per_band = in_dim // 256, in_dim // 128
    w = np.zeros((out_dim, in_dim), np.float32)
    for c in range(chunks.shape[0]):
        r0 = 64 * (c // per_band) + 32 * (c % 2)
        k0 = 256 * ((c % per_band) // 2)
        w[r0:r0 + 32, k0:k0 + 256] = chunks[c]
    return w


def want_matrix(W: np.ndarray, op: dict, rows: int) -> np.ndarray:
    r = np.arange(rows)
    if op.get("take"):
        r = (r // op["take"]) * op["period"] + op["offset"] + r % op["take"]
    want = W[r]
    if op.get("kheads"):
        idx = untile(want.shape[op["vaxis"]], op["kheads"], op["vperk"], op["vdim"], op["v0"])
        want = want[idx] if op["vaxis"] == 0 else want[:, idx]
    return want


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--gguf-py", default=None, help="llama.cpp's gguf-py directory (if gguf is not importable)")
    a = ap.parse_args()
    g = Gguf(a.gguf, a.gguf_py)
    man = json.loads(Path(a.manifest).read_text())["gguf"]
    ch = man["pack"]["chunk_bytes"]
    assert ch == 5120, ch
    bad = 0
    q8_of: dict[tuple[int, str, int], np.ndarray] = {}    # (layer, tensor, offset) -> decoded q8 values

    def report(name, typ, got, want, lim=None):
        nonlocal bad
        rel, mx = stats(got, want)
        if lim is None:
            lim = 0.0 if typ in FLOATS else EXACT_LIMIT if typ in LAW else REQUANT_LIMIT
        ok = rel <= lim if lim == 0.0 else rel < lim
        bad += not ok
        tag = "" if typ in LAW or typ in FLOATS else "  (requantized to int8)"
        print(f"  {name:48s} {typ:7s} rel L2 {rel:.3e}  max {mx:.3e}  {'ok' if ok else 'FAIL'}{tag}")

    for l in map(int, a.layers.split(",")):
        lt_name = man["layers"][l]
        lt = man["layer_types"][lt_name]
        pool = np.fromfile(Path(a.dump) / f"pool{l}.bin", np.uint8)
        consts = np.fromfile(Path(a.dump) / f"consts{l}.bin", np.uint8)
        print(f"layer {l} ({lt_name}):")
        for op in lt["pack"]["pool"] + lt["pack"]["consts"]:
            buf = pool if op in lt["pack"]["pool"] else consts
            name = op["tensor"].replace("{l}", str(l))
            kind = op["op"]
            W = g.array(name)
            if kind == "q35_q8":
                rows = op["nch"] // 2 // (op["in_dim"] // 256) * 32
                tiles = q8_half_tiles(buf[op["dst"]: op["dst"] + op["nch"] * ch].reshape(-1, ch))
                got = q8_perm_inverse(tiles, rows, op["in_dim"])
                q8_of[(l, op["tensor"], op.get("offset", 0))] = got
                report(name + (f" [{op['offset']}::{op['period']}]" if op.get("take") else ""), g.type(name), got,
                       want_matrix(W, op, rows))
            elif kind == "put":
                got = from_bf16(buf[op["dst"]: op["dst"] + 2 * W.size].copy().view(np.uint16))
                report(name, g.type(name), got, bf16(W.ravel()), 0.0)
            elif kind == "q35_t":
                dr = op.get("dst_rows", op["rows"])
                got = from_bf16(buf[op["dst"]: op["dst"] + 2 * op["cols"] * dr].copy().view(np.uint16)).reshape(op["cols"], dr)
                want = W[untile(W.shape[0], op["kheads"], op["vperk"], op["vdim"], op["v0"])] if op.get("kheads") else W
                report(name, g.type(name), got[:, :op["rows"]], bf16(want.T), 0.0)
                if dr > op["rows"] and np.any(got[:, op["rows"]:]):
                    print(f"  {name}: padding lanes are not zero  FAIL")
                    bad += 1
            elif kind == "q35_vec":
                n = W.size
                got = buf[op["dst"]: op["dst"] + 4 * n].copy().view(np.float32)
                want = W.ravel()[untile(n, op["kheads"], op["vperk"], op["vdim"], op["v0"])] if op.get("kheads") else W.ravel()
                report(name, g.type(name), got, want, 0.0)
            elif kind == "q35_conv":
                taps, groups, width = op["taps"], op["groups"], op["width"]
                nchan = groups * width
                banked = from_bf16(buf[op["dst"]: op["dst"] + 2 * taps * nchan].copy().view(np.uint16)).reshape(groups, taps, width)
                got = banked.transpose(1, 0, 2).reshape(taps, nchan)
                Wc = W.reshape(nchan, taps)
                want = Wc[untile(nchan, op["kheads"], op["vperk"], op["vdim"], op["v0"])] if op.get("kheads") else Wc
                report(name, g.type(name), got, bf16(want.T), 0.0)
            else:
                print(f"  {name}: no check for {kind}  FAIL")
                bad += 1
        # the block route's split weights: hi + lo == the decode pool's q8 values, exactly
        gb = lt.get("gemm_block") or {}
        for key in ("weights", "shared_weights", "ffn_weights"):
            for wname, w in (gb.get(key) or {}).items():
                if w.get("from") != "pack":
                    continue
                f = Path(a.dump) / f"route{l}_{wname}.bin"
                if not f.exists():
                    print(f"  route {wname}: not dumped  FAIL")
                    bad += 1
                    continue
                buf = np.fromfile(f, np.uint8)
                halves: dict[tuple[str, int], dict[str, np.ndarray]] = {}
                for op in w["pack"]:
                    rows = op["nch"] // (op["in_dim"] // 256) * 32
                    vals = q41_chunks(buf[op["dst"]: op["dst"] + op["nch"] * ch].reshape(-1, ch))
                    halves.setdefault((op["tensor"], op.get("offset", 0)), {})[op["split"]] = \
                        std_perm_inverse(vals, rows, op["in_dim"])
                for (t, off), hl in halves.items():
                    ref = q8_of.get((l, t, off))
                    if ref is None:
                        print(f"  route {wname} {t}: no decode-pool q8 to compare  FAIL")
                        bad += 1
                        continue
                    same = np.array_equal(hl["hi"] + hl["lo"], ref)
                    bad += not same
                    print(f"  route {wname:12s} {t.replace('{l}', str(l)):36s} hi + lo == q8 pool: "
                          f"{'ok (bit-exact)' if same else 'FAIL'}")
    if a.head:
        op = man["pack"]["lm_head"]["ops"][0]
        name = op["tensor"]
        W = g.array(name)
        head = np.fromfile(Path(a.dump) / "lmhead.bin", np.uint8)
        nk = op["in_dim"] // 256
        per_band = 4 * nk
        sel = list(range(0, 4 * per_band)) + list(range(per_band * 900, per_band * 901))   # bands 0..3 and 900
        got = np.zeros_like(W)
        seen = np.zeros(W.shape[0], bool)
        nch = W.shape[0] // 32 * nk
        vals = q8_chunks(head[op["dst"]:op["dst"] + nch * 8704].reshape(-1, 8704)[sel])
        for v, k in zip(vals, sel):
            s, rr = k // per_band, k % per_band
            rb, kt = 4 * s + rr % 4, rr // 4
            got[32 * rb:32 * rb + 32, 256 * kt:256 * kt + 256] = v
            seen[32 * rb:32 * rb + 32] = True
        print("head:")
        report(name + " (5 bands)", g.type(name), got[seen], W[seen])
    print("ALL OK" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
