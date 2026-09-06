"""Interpreter of a recipe's packing plan over a weight container (NumPy).

The plan (qwen36moe.pack_plan) says which tensor lands at which byte offset in
which chunk order; the ops here are the chunk-permutation laws phlegm verified
byte-for-byte against pools captured from FLM's own engine (they were
open_kernels/model/pools.py's build_layer_pool / build_side / build_pack; the
frozen originals live in specs/open-engine/tests/legacy_pools.py and the test
there checks this interpreter reproduces them). src/open_qwen36/pools.cpp is
the same interpreter in C++.

The container object needs one method: `raw(name) -> bytes-like` (the
tensor's bytes as stored; q4_1 chunks in the file's raster order).

  std_perm        standard [out, in] matmul tensor -> pool band order (64-row bands, K/128 chunks)
  expert_stripes  routed experts' up / gate as interleaved [up_k | gate_k] stripes, each transposed
  expert_down     routed experts' down slices, the RS=4 down law
  put             bytes verbatim (small weights), capped
  conv_transpose  conv1d [taps, NCH] bf16 -> [groups][taps][width]
  lmhead_q8       the q8 lm_head's 128-row supertile order
"""
from __future__ import annotations

import numpy as np

CH = 5120


def std_perm(nch: int, in_dim: int) -> np.ndarray:
    """pool chunk index -> file chunk index, for a standard [out, in] matmul tensor:
    a band is 64 rows x in_dim = per_band = in_dim/128 chunks; inside its band chunk i
    covers row half i % 2 and k-tile i // 2 (gemv_q4.h's band law, q4_1_pack.chunk_geometry);
    file chunk f covers rows 32*(f//ncol), cols 256*(f%ncol).

    The law phlegm verified against FLM's captured pools was written as
    cols = 1024*((c//8) % (in//1024)) + 256*((c//2) % 4); for in_dim a multiple of 1024
    that is this same k-tile order (tests/test_pack_plan.py checks the two agree there);
    this form is the one that also holds for in_dim = 2560 or 9728."""
    ncol = in_dim // 256
    per_band = in_dim // 128
    c = np.arange(nch)
    rows0 = 64 * (c // per_band) + 32 * (c % 2)
    cols0 = 256 * ((c % per_band) // 2)
    return (rows0 // 32) * ncol + cols0 // 256


def down_perm(nch: int = 128) -> np.ndarray:
    """One expert's down [HID, FF] slice: pool chunk c <- file chunk 2*rt + cg,
    rt = 4*(c//8) + c%4, cg = (c//4)%2 (validated for [2048, 512])."""
    c = np.arange(nch)
    rt = 4 * (c // 8) + (c % 4)
    return 2 * rt + (c // 4) % 2


def stripe_transpose(in_dim: int = 2048) -> np.ndarray:
    """Inside one 128-row stripe: pool chunk c <- file chunk ncol*(c%4) + c//4."""
    ncol = in_dim // 256
    c = np.arange(4 * ncol)
    return ncol * (c % 4) + c // 4


def _u8(b) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8) if not isinstance(b, np.ndarray) else b.view(np.uint8)


def _name(op: dict, key: str, layer: int) -> str:
    return op[key].replace("{l}", str(layer))


def apply_op(op: dict, m, layer: int, dst: np.ndarray) -> None:
    kind = op["op"]
    if kind == "std_perm":
        raw = _u8(m.raw(_name(op, "tensor", layer))).reshape(-1, CH)
        c0 = op.get("chunk0", 0)
        sel = raw[c0:c0 + op["nch"]]
        if sel.shape[0] != op["nch"]:
            raise ValueError(f"{op['tensor']}: {raw.shape[0]} chunks, need {c0 + op['nch']}")
        n = op["nch"] * CH
        dst[op["dst"]:op["dst"] + n] = sel[std_perm(op["nch"], op["in_dim"])].reshape(-1)
    elif kind == "expert_stripes":
        up = _u8(m.raw(_name(op, "up", layer)))
        gt = _u8(m.raw(_name(op, "gate", layer)))
        S, ns, E = op["stripe_bytes"], op["stripes"], op["experts"]
        tp = stripe_transpose(op["in_dim"])
        base = op["dst"]
        for e in range(E):
            for k in range(ns):
                src = (ns * e + k) * S
                d = base + (2 * ns * e + 2 * k) * S
                dst[d:d + S] = up[src:src + S].reshape(-1, CH)[tp].reshape(-1)
                dst[d + S:d + 2 * S] = gt[src:src + S].reshape(-1, CH)[tp].reshape(-1)
    elif kind == "expert_down":
        dn = _u8(m.raw(_name(op, "tensor", layer)))
        B, E = op["expert_bytes"], op["experts"]
        dp = down_perm(B // CH)
        base = op["dst"]
        for e in range(E):
            dst[base + e * B:base + (e + 1) * B] = dn[e * B:(e + 1) * B].reshape(-1, CH)[dp].reshape(-1)
    elif kind == "put":
        b = _u8(m.raw(_name(op, "tensor", layer)))
        if len(b) > op["cap"]:
            raise ValueError(f"{op['tensor']}: {len(b)} B does not fit its {op['cap']} B slot")
        dst[op["dst"]:op["dst"] + len(b)] = b
    elif kind == "lmhead_q8":
        # the q8 lm_head's 128-row supertile order: pool chunk k <- file chunk (4*(k//32) + (k%4))*8 + ((k%32)//4)
        ch = op["chunk_bytes"]
        raw = _u8(m.raw(_name(op, "tensor", layer))).reshape(-1, ch)
        k = np.arange(raw.shape[0])
        s, r = k // 32, k % 32
        perm = (4 * s + r % 4) * 8 + r // 4
        n = raw.shape[0] * ch
        if op["dst"] + n > len(dst):
            raise ValueError("lm_head larger than its pool")
        dst[op["dst"]:op["dst"] + n] = raw[perm].reshape(-1)
    elif kind == "conv_transpose":
        b = _u8(m.raw(_name(op, "tensor", layer)))
        taps, groups, width = op["taps"], op["groups"], op["width"]
        if len(b) != taps * groups * width * 2:
            raise ValueError(f"{op['tensor']}: {len(b)} B is not bf16[{taps}, {groups * width}]")
        w = b.view(np.uint16).reshape(taps, groups, width).transpose(1, 0, 2).reshape(-1).view(np.uint8)
        dst[op["dst"]:op["dst"] + len(w)] = w
    else:
        raise ValueError(f"unknown pack op {kind!r}")


def build_layer_pool(plan: dict, layer_type: str, m, layer: int, out: np.ndarray | None = None) -> np.ndarray:
    pool = np.zeros(plan["pool_bytes"], np.uint8) if out is None else out
    if out is not None:
        pool[:] = 0
    for op in plan["layer_types"][layer_type]["pool"]:
        apply_op(op, m, layer, pool)
    return pool


def build_consts(plan: dict, layer_type: str, m, layer: int, nbytes: int) -> np.ndarray:
    c = np.zeros(nbytes, np.uint8)
    for op in plan["layer_types"][layer_type]["consts"]:
        apply_op(op, m, layer, c)
    return c


def build_lmhead_pool(plan: dict, m) -> np.ndarray:
    """The lm_head pool: the plan's ops (q8 supertiles for the 27B, a std_perm for a q4 head)."""
    lm = plan["lm_head"]
    out = np.zeros(lm["pool_bytes"], np.uint8)
    for op in lm["ops"]:
        apply_op(op, m, 0, out)
    return out


def window_rows(p, window: int):
    """(valid, nf) for position p: the cached rows the attention core sees are [s, p) with
    s = max(0, p - (window - 1)) (window 0: all of them); it streams nf = max(1, p - s) rows and
    masks the ones at index >= valid = p - s (the dummy row at position 0)."""
    p = np.asarray(p)
    s = np.maximum(0, p - (window - 1)) if window else np.zeros_like(p)
    valid = p - s
    return valid, np.maximum(valid, 1)


def ptab(rows: int, rotary_dim: int, theta: float, ptab_row: int = 1024, inv_freq=None, window: int = 0) -> np.ndarray:
    """The position record table: row p = [i32 valid | i32 nf | cos f32[rot/2] @512 | sin f32[rot/2]
    right after the cos, @512 + 2*rot] for the RoPE over the first `rotary_dim` dims of a head (half-split
    pairs (i, i + rot/2)); attn.h reads the rot floats at +512 as [cos | sin]. `inv_freq` (rot/2 values,
    ModelSpec.rope_inv_freq -- Llama 3's scaling lives there) defaults to theta^(-2i/rot). `window`
    (rows, 0 = unbounded) makes the record count the sliding window's rows (window_rows)."""
    half = rotary_dim // 2
    if 512 + 8 * half > ptab_row:
        raise ValueError(f"a rotary dim of {rotary_dim} does not fit a {ptab_row}-byte position record")
    t = np.zeros((rows, ptab_row), np.uint8)
    p = np.arange(rows)
    valid, nf = window_rows(p, window)
    t[:, :8] = np.stack([valid, nf], 1).astype(np.int32).view(np.uint8)
    f = np.asarray(inv_freq, np.float64) if inv_freq is not None else theta ** (-np.arange(half) / half)
    if len(f) != half:
        raise ValueError(f"inv_freq has {len(f)} values, the rotary dim wants {half}")
    ang = p[:, None] * f[None, :]
    t[:, 512:512 + 4 * half] = np.cos(ang).astype(np.float32).view(np.uint8)
    t[:, 512 + 4 * half:512 + 8 * half] = np.sin(ang).astype(np.float32).view(np.uint8)
    return t.reshape(-1)
