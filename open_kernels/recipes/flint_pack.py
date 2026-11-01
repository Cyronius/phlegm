"""The bytes recipes/flint.py's pools hold (OPEN-FAMILY-FLINT), in NumPy: the runtime's fl_q4 pack op
(flint-runtime.patch, pools.cpp) is the same law, pinned to std_perm_gguf in gguf_pack_test.

A Flint GGUF matrix is Q4_1 (or Q4_0) blocks; the pool is q4_1_pack.py's 5120-B chunks in std_perm's
64-row bands (row split 2), the codes kept and d / m rounded to bf16 (RNE) -- exact for RecursiveSI's
int4 file, whose scales and mins are bf16 values. A slot (a routed expert, a shared-expert half or a
dense-MLP slice of 512 rows) holds [gate band b | up band b] for every band b, then the down slices of
every slot follow (recipes/flint.py layout / pool).
"""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

CH = 5120                       # q4_1_pack.py's chunk: d bf16[256] | m bf16[256] | 4096 B of nibbles
BLK = 32
Q4_1_BYTES, Q4_0_BYTES = 20, 18


def pack_q4_1_pool(b: np.ndarray, rs: int = 2) -> np.ndarray:
    """q4_1_pack.pack_q4_1_pool (the same arithmetic, kept here so the recipe package stands alone):
    GGUF Q4_1 (20 B) or Q4_0 (18 B) blocks uint8[n, nb, B] -> pool-order chunk bytes uint8[nch * 5120]."""
    n, nb, width = b.shape
    k = nb * BLK
    d = np.ascontiguousarray(b[..., 0:2]).view(np.float16).reshape(n, nb).astype(np.float32)
    if width == Q4_1_BYTES:
        m = np.ascontiguousarray(b[..., 2:4]).view(np.float16).reshape(n, nb).astype(np.float32)
        qs = b[..., 4:20]
    else:
        m, qs = -8.0 * d, b[..., 2:18]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=-1).reshape(n, k)
    d16, m16 = d.astype(bfloat16).view(np.uint16), m.astype(bfloat16).view(np.uint16)
    per_band = rs * k // 256
    nch = n * k // (32 * 256)
    out = np.empty((nch, CH), np.uint8)
    for c in range(nch):
        band, ci = divmod(c, per_band)
        r0, c0 = 32 * rs * band + 32 * (ci % rs), 256 * (ci // rs)
        kb0 = c0 // BLK
        out[c, 0:512] = d16[r0:r0 + 32, kb0:kb0 + 8].T.reshape(-1).view(np.uint8)
        out[c, 512:1024] = m16[r0:r0 + 32, kb0:kb0 + 8].T.reshape(-1).view(np.uint8)
        qq = q[r0:r0 + 32, c0:c0 + 256].reshape(2, 16, 256).transpose(0, 2, 1).reshape(-1)
        out[c, 1024:] = qq[0::2] | (qq[1::2] << 4)
    return out.reshape(-1)


def blocks(raw: np.ndarray, rows: int, cols: int, q4_1: bool = True) -> np.ndarray:
    """A GGUF Q4_1 / Q4_0 tensor's raw bytes (trailing dims flattened into rows) -> uint8[rows, cols/32, B]."""
    width = Q4_1_BYTES if q4_1 else Q4_0_BYTES
    return np.asarray(raw, np.uint8).reshape(rows, cols // BLK, width)


def matrix(b: np.ndarray, rows: int, in_dim: int, row0: int = 0, col0: int = 0) -> np.ndarray:
    """The rows x in_dim block at (row0, col0) as 64-row bands of q4_1 chunks (fl_q4 with one expert)."""
    assert rows % 64 == 0 and in_dim % 256 == 0 and col0 % BLK == 0
    return pack_q4_1_pool(np.ascontiguousarray(b[row0:row0 + rows, col0 // BLK:(col0 + in_dim) // BLK]), 2)


def band_bytes(in_dim: int) -> int:
    return in_dim // 128 * CH


def slot_ug(gate: np.ndarray, up: np.ndarray, in_dim: int) -> np.ndarray:
    """One slot's [gate band b | up band b] from the two packed projections."""
    bb = band_bytes(in_dim)
    g, u = gate.reshape(-1, bb), up.reshape(-1, bb)
    assert g.shape == u.shape
    return np.stack([g, u], axis=1).reshape(-1)
