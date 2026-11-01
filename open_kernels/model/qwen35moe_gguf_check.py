r"""Check the Qwen3.5 / 3.6 MoE GGUF view's packed pool and consts bytes against an independent
reading of the same GGUF (OPEN-FAMILY-QWEN36MOE-GGUF).

The runtime's q35_* ops (upstream-patches/qwen35moe-gguf-runtime.patch) pack a layer straight
from model.gguf; `q35_pack_dump` writes those bytes. This script dequantizes the GGUF with
llama.cpp's own gguf-py (`gguf.quants.dequantize`), applies the conversion's inverse maps in
NumPy (the per-head [q | gate] split, the linear-attention value-head untile, the conv / alpha /
beta / router transposes) and decodes the packed q4_1-law chunks back into matrices through the
inverse band laws. Exact source types (Q4_0 / Q4_1 / Q4_K) must agree up to the bf16 scale
narrowing (per-type EXACT_LIMIT); requantized ones (Q8_0 / Q5_0 / Q6_K) fail unless
--allow-requant, which reports their error.

    q35_pack_dump manifest.json model.gguf OUT 0,3 head
    python open_kernels/model/qwen35moe_gguf_check.py --gguf model.gguf --manifest manifest.json \
        --dump OUT --layers 0,3 [--head] --gguf-py ~/llama.cpp/gguf-py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXACT = {"Q4_0", "Q4_1", "Q4_K"}
# Per-type relative-L2 bound for a block the q35 ops keep (codes unchanged, scales RNE to bf16).
# Q4_0 (d only) and a Q8_0 head (codes kept) measure 1.6-1.7e-3. A Q4_1 min and Q4_K's d*sc /
# dmin*mn products are rounded too: measured 3.8-5.1e-3 and 5.0-6.7e-3 (KAT-Coder-V2.5-Dev),
# whose cost on the NPU is 452/458 of 491 argmax against llama.cpp (OPEN-FAMILY-QWEN36MOE-GGUF).
EXACT_LIMIT = {"Q4_0": 2.5e-3, "Q8_0": 2.5e-3, "Q4_1": 1e-2, "Q4_K": 7e-3}
REQUANT = {"Q8_0", "Q5_0", "Q6_K"}


def bf16(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16
    return u.view(np.float32)


def from_bf16(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def code_index() -> np.ndarray:
    """[32 rows, 256 k] -> nibble index inside a chunk's code plane."""
    r = np.arange(32)[:, None]
    k = np.arange(256)[None, :]
    return (r // 16) * 4096 + k * 16 + r % 16


CI = code_index()


def q41_chunks(buf: np.ndarray) -> np.ndarray:
    """[n, 5120] q4_1-law chunks -> [n, 32, 256] values."""
    n = buf.shape[0]
    d = from_bf16(buf[:, :512].copy().view(np.uint16)).reshape(n, 8, 32).transpose(0, 2, 1)
    m = from_bf16(buf[:, 512:1024].copy().view(np.uint16)).reshape(n, 8, 32).transpose(0, 2, 1)
    codes = buf[:, 1024:]
    nib = np.empty((n, 8192), np.uint8)
    nib[:, 0::2] = codes & 15
    nib[:, 1::2] = codes >> 4
    q = nib[:, CI].astype(np.float32)                    # [n, 32, 256]
    return q * np.repeat(d, 32, axis=2) + np.repeat(m, 32, axis=2)


def q8_chunks(buf: np.ndarray) -> np.ndarray:
    """[n, 8704] lm_head q8 chunks -> [n, 32, 256] values."""
    n = buf.shape[0]
    s = from_bf16(buf[:, :512].copy().view(np.uint16)).reshape(n, 8, 32).transpose(0, 2, 1)
    q = buf[:, 512:].view(np.int8)[:, CI].astype(np.float32)
    return q * np.repeat(s, 32, axis=2)


def std_perm_inverse(chunks: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    ncol, per_band = in_dim // 256, in_dim // 128
    w = np.zeros((out_dim, in_dim), np.float32)
    for c in range(chunks.shape[0]):
        r0 = 64 * (c // per_band) + 32 * (c % 2)
        k0 = 256 * ((c % per_band) // 2)
        w[r0:r0 + 32, k0:k0 + 256] = chunks[c]
    return w


def untile(n: int, kheads: int, vperk: int, vdim: int, v0: int) -> np.ndarray:
    """Pool index -> GGUF index (the runtime's Q35Map value-head part)."""
    i = np.arange(n)
    rel = i - v0
    h, k = rel // vdim, (rel // vdim) // vperk
    j = h % vperk
    src = v0 + (j * kheads + k) * vdim + rel % vdim
    return np.where(i >= v0, src, i)


class Gguf:
    def __init__(self, path: str, gguf_py: str | None):
        if gguf_py:
            sys.path.insert(0, gguf_py)
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.dq = dequantize
        self.r = GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}

    def type(self, name: str) -> str:
        return self.t[name].tensor_type.name

    def array(self, name: str) -> np.ndarray:
        t = self.t[name]
        a = self.dq(t.data, t.tensor_type).astype(np.float32)
        return a.reshape(tuple(int(x) for x in reversed(t.shape)))


def stats(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    got, want = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    rel = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))
    mx = float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-30))
    return rel, mx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--gguf-py", default=None, help="llama.cpp's gguf-py directory (if gguf is not importable)")
    ap.add_argument("--experts", type=int, default=8, help="how many routed experts to decode per layer")
    ap.add_argument("--allow-requant", action="store_true",
                    help="report requantized tensors (Q8_0 / Q5_0 / Q6_K pools, a non-Q8_0 head) instead of failing")
    a = ap.parse_args()
    g = Gguf(a.gguf, a.gguf_py)
    man = json.loads(Path(a.manifest).read_text())["gguf"]
    ch = man["pack"]["chunk_bytes"]
    assert ch == 5120, ch
    bad = 0

    def report(name, typ, got, want, exact_types=EXACT):
        nonlocal bad
        rel, mx = stats(got, want)
        exact = typ in exact_types
        requant = not exact and typ in REQUANT
        # requantized tensors fail unless --allow-requant; F32 / F16 / BF16 small tensors are bit-equal
        lim = EXACT_LIMIT[typ] if exact else (0.25 if requant and a.allow_requant else 0.0 if requant else 1e-2)
        ok = rel < lim
        bad += not ok
        print(f"  {name:44s} {typ:5s} rel L2 {rel:.3e}  max {mx:.3e}  {'ok' if ok else 'FAIL'}"
              f"{'  (requantized)' if requant else ''}")

    for l in map(int, a.layers.split(",")):
        lt_name = man["layers"][l]
        lt = man["layer_types"][lt_name]
        pool = np.fromfile(Path(a.dump) / f"pool{l}.bin", np.uint8)
        consts = np.fromfile(Path(a.dump) / f"consts{l}.bin", np.uint8)
        print(f"layer {l} ({lt_name}):")
        for op in lt["pack"]["pool"] + lt["pack"]["consts"]:
            buf = pool if op in lt["pack"]["pool"] else consts
            name = op.get("tensor", op.get("up", "")).replace("{l}", str(l))
            kind = op["op"]
            if kind == "q35_stripes":
                up, gate = name, op["gate"].replace("{l}", str(l))
                U, G = g.array(up), g.array(gate)                     # [E, ff, hid]
                E, ns, S = op["experts"], op["stripes"], op["stripe_bytes"]
                ncol = U.shape[2] // 256
                for e in range(min(a.experts, E)):
                    base = op["dst"] + e * 2 * ns * S
                    for half, W, nm in ((0, U, up), (1, G, gate)):
                        got = np.zeros_like(W[e])
                        for k in range(ns):
                            st = pool[base + (2 * k + half) * S: base + (2 * k + half + 1) * S].reshape(-1, ch)
                            vals = q41_chunks(st)
                            for c in range(vals.shape[0]):
                                r0, kt = 128 * k + 32 * (c % 4), c // 4
                                got[r0:r0 + 32, 256 * kt:256 * kt + 256] = vals[c]
                        if e in (0, E - 1) or e == min(a.experts, E) - 1:
                            report(f"{nm}[{e}]", g.type(nm), got, W[e])
                        elif not np.allclose(got, W[e], rtol=0, atol=np.abs(W[e]).max() * 0.02):
                            report(f"{nm}[{e}]", g.type(nm), got, W[e])
                continue
            if kind == "q35_down":
                D = g.array(name)                                      # [E, hid, ff]
                E, B = op["experts"], op["expert_bytes"]
                ncol = D.shape[2] // 256
                for e in sorted({0, min(a.experts, E) - 1, E - 1}):
                    vals = q41_chunks(pool[op["dst"] + e * B: op["dst"] + (e + 1) * B].reshape(-1, ch))
                    got = np.zeros_like(D[e])
                    for c in range(vals.shape[0]):
                        r0, kt = 128 * (c // (4 * ncol)) + 32 * (c % 4), (c // 4) % ncol
                        got[r0:r0 + 32, 256 * kt:256 * kt + 256] = vals[c]
                    report(f"{name}[{e}]", g.type(name), got, D[e])
                continue
            W = g.array(name)
            if kind == "q35_mat":
                rows = op["nch"] // (op["in_dim"] // 256) * 32
                vals = q41_chunks(buf[op["dst"]: op["dst"] + op["nch"] * ch].reshape(-1, ch))
                got = std_perm_inverse(vals, rows, op["in_dim"])
                r = np.arange(rows)
                if op.get("take"):
                    r = (r // op["take"]) * op["period"] + op["offset"] + r % op["take"]
                want = W[r]
                if op.get("kheads"):
                    idx = untile(want.shape[op["vaxis"]], op["kheads"], op["vperk"], op["vdim"], op["v0"])
                    want = want[idx] if op["vaxis"] == 0 else want[:, idx]
                report(name + (f" [{op['offset']}::{op['period']}]" if op.get("take") else ""), g.type(name), got, want)
            elif kind == "put":
                n = W.size
                got = from_bf16(buf[op["dst"]: op["dst"] + 2 * n].copy().view(np.uint16))
                report(name, g.type(name), got, bf16(W.ravel()))
            elif kind == "q35_t":
                dr = op.get("dst_rows", op["rows"])
                got = from_bf16(buf[op["dst"]: op["dst"] + 2 * op["cols"] * dr].copy().view(np.uint16)).reshape(op["cols"], dr)
                want = W
                if op.get("kheads"):
                    want = W[untile(W.shape[0], op["kheads"], op["vperk"], op["vdim"], op["v0"])]
                report(name, g.type(name), got[:, :op["rows"]], bf16(want.T))
            elif kind == "q35_vec":
                n = W.size
                got = buf[op["dst"]: op["dst"] + 4 * n].copy().view(np.float32)
                want = W.ravel()[untile(n, op["kheads"], op["vperk"], op["vdim"], op["v0"])] if op.get("kheads") else W.ravel()
                report(name, g.type(name), got, want)
            elif kind == "q35_conv":
                taps, groups, width = op["taps"], op["groups"], op["width"]
                nchan = groups * width
                banked = from_bf16(buf[op["dst"]: op["dst"] + 2 * taps * nchan].copy().view(np.uint16)).reshape(groups, taps, width)
                got = banked.transpose(1, 0, 2).reshape(taps, nchan)
                Wc = W.reshape(nchan, taps)
                want = Wc[untile(nchan, op["kheads"], op["vperk"], op["vdim"], op["v0"])] if op.get("kheads") else Wc
                report(name, g.type(name), got, bf16(want.T))
            else:
                print(f"  {name}: no check for {kind}")
                bad += 1
    if a.head:
        op = man["pack"]["lm_head"]["ops"][0]
        name = op["tensor"] if op["tensor"] in g.t else "token_embd.weight"
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
        # q35_lmhead keeps a Q8_0 head's codes (scale RNE to bf16); other types are requantized to q8
        report(name + " (5 bands)", g.type(name), got[seen], W[seen], EXACT | {"Q8_0"})
    print("ALL OK" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
