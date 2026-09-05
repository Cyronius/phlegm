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
    pool chunk c covers rows 64*(c//per_band) + 32*(c%2), cols 1024*((c//8) % (in//1024)) + 256*((c//2)%4);
    file chunk f covers rows 32*(f//ncol), cols 256*(f%ncol)."""
    ncol = in_dim // 256
    per_band = in_dim // 128
    kgroups = max(1, in_dim // 1024)
    c = np.arange(nch)
    rows0 = 64 * (c // per_band) + 32 * (c % 2)
    cols0 = (1024 * ((c // 8) % kgroups) + 256 * ((c // 2) % 4)) % in_dim
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
    """The q8 lm_head pool: 128-row supertile order, pool chunk k <- file chunk
    (4*(k//32) + (k%4))*8 + ((k%32)//4)."""
    op = plan["lm_head"]
    ch = op["chunk_bytes"]
    raw = _u8(m.raw(op["tensor"])).reshape(-1, ch)
    k = np.arange(raw.shape[0])
    s, r = k // 32, k % 32
    perm = (4 * s + r % 4) * 8 + r // 4
    out = np.zeros(op["pool_bytes"], np.uint8)
    if raw.shape[0] * ch > op["pool_bytes"]:
        raise ValueError("lm_head larger than its pool")
    out[:raw.shape[0] * ch] = raw[perm].reshape(-1)
    return out


def ptab(rows: int, rotary_dim: int, theta: float, ptab_row: int = 1024) -> np.ndarray:
    """The position record table: row p = [i32 pos | i32 nf = max(p, 1) | cos f32[rot/2] @512 | sin @640]
    for the partial RoPE (half-split pairs (i, i + rot/2))."""
    half = rotary_dim // 2
    t = np.zeros((rows, ptab_row), np.uint8)
    p = np.arange(rows)
    t[:, :8] = np.stack([p, np.maximum(p, 1)], 1).astype(np.int32).view(np.uint8)
    ang = p[:, None] * (theta ** (-np.arange(half) / half))[None, :]
    t[:, 512:512 + 4 * half] = np.cos(ang).astype(np.float32).view(np.uint8)
    t[:, 640:640 + 4 * half] = np.sin(ang).astype(np.float32).view(np.uint8)
    return t.reshape(-1)
