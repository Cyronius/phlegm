"""Crack the q4nx chunk layout by replicating op255 (qkv matmul, layer 0).

Known: act [11x2048] bf16, out [11x8192] bf16 at row-offset 3, weight = packed
qkv chunks at pool offset 505282560 ([256,8,5120] = 2048 chunks, verbatim).
Observed: chunk STARTS with ~1024B of small positive bf16 (scales), nibbles after.
"""
import numpy as np
from q4nx import bf16_to_f32

CAP = "C:/caps/m0d"
act = bf16_to_f32(np.fromfile(f"{CAP}/blob_2097152_ae6193526c412737.bin", dtype=np.uint16)).reshape(512, 2048)[:11]
out = bf16_to_f32(np.fromfile(f"{CAP}/blob_9437184_31c16cad8c127d91.bin", dtype=np.uint16)).reshape(576, 8192)[3:14]
pool = np.fromfile(f"{CAP}/blob_536870912_836fd8e49f35a0b6.bin", dtype=np.uint8,
                   count=10485760, offset=505282560)
chunks = pool.reshape(2048, 5120)

def nib(q, planar, hi_first, n):
    lo = (q & 0xF).astype(np.float32)
    hi = (q >> 4).astype(np.float32)
    if hi_first:
        lo, hi = hi, lo
    v = np.empty(n, dtype=np.float32)
    if planar:
        v[: n // 2], v[n // 2 :] = lo, hi
    else:
        v[0::2], v[1::2] = lo, hi
    return v

def dq(chunk, kind, planar, hi_first, signed):
    meta, q = chunk[:1024], chunk[1024:]
    v = nib(q, planar, hi_first, 8192)
    v = v - 8.0 if signed == "off8" else np.where(v > 7, v - 16, v) if signed == "twos" else v
    mb = meta.tobytes()
    if kind == "b16_scale":                       # 512 bf16 scales, block 16
        sc = bf16_to_f32(np.frombuffer(mb, dtype=np.uint16))
        return (v.reshape(512, 16) * sc[:, None]).reshape(-1)
    if kind == "b32_scale_min_planar":            # 256 scales then 256 mins
        sc = bf16_to_f32(np.frombuffer(mb[:512], dtype=np.uint16))
        mn = bf16_to_f32(np.frombuffer(mb[512:], dtype=np.uint16))
        return (v.reshape(256, 32) * sc[:, None] + mn[:, None]).reshape(-1)
    if kind == "b32_scale_min_inter":             # (scale,min) bf16 pairs
        m = bf16_to_f32(np.frombuffer(mb, dtype=np.uint16)).reshape(256, 2)
        return (v.reshape(256, 32) * m[:, 0:1] + m[:, 1:2]).reshape(-1)
    if kind == "b32_scale_zp_inter":              # (scale, zp) zp also bf16-coded
        m = bf16_to_f32(np.frombuffer(mb, dtype=np.uint16)).reshape(256, 2)
        return ((v.reshape(256, 32) - m[:, 1:2]) * m[:, 0:1]).reshape(-1)
    raise ValueError

t0 = act[0]
ref = out[0]
res = []
for kind in ("b16_scale", "b32_scale_min_planar", "b32_scale_min_inter", "b32_scale_zp_inter"):
    for planar in (False, True):
        for hi_first in (False, True):
            for signed in ("off8", "twos", "none"):
                pred = np.empty(64)
                ok = True
                for r in range(64):
                    c, off = divmod(r, 4)
                    w = dq(chunks[c], kind, planar, hi_first, signed)
                    pred[r] = float(w[off * 2048:(off + 1) * 2048] @ t0)
                refv = ref[:64]
                denom = (np.linalg.norm(pred) * np.linalg.norm(refv)) or 1
                corr = float(pred @ refv / denom)
                res.append((abs(corr), corr, kind, planar, hi_first, signed))
res.sort(reverse=True)
for b in res[:10]:
    print(f"corr={b[1]:+.5f}  {b[2]:>22} planar={b[3]} hi_first={b[4]} signed={b[5]}")
