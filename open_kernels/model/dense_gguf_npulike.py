"""Make CPU control GGUFs for the proposed dense K-quant carriages.

Run with python -I and --gguf-py pointing at a trusted llama.cpp gguf-py.
The source GGUF remains unchanged. Outputs contain F32 replacement tensors and
are controls, not packed NPU weights. A tied embedding is preserved for input;
a separately converted output.weight is added for the head.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

PROJECTION = re.compile(r"blk\.\d+\.(?:attn_[qkv]|attn_output|ffn_(?:gate|up|down))\.weight$")
MODES = ("refit", "codes-kept", "q6-q4", "q6-int8")


def bf16(x):
    """Round float32 to bfloat16, ties to even, represented as float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def q4_refit(v):
    b = np.asarray(v, np.float32).reshape(-1, 32)
    # Match the existing q35/dense pool's outward bf16 rounding.
    def outward(x, upward):
        u = np.ascontiguousarray(x, np.float32).view(np.uint32)
        add = np.where((u >> 31) == (0 if upward else 1), 0xffff, 0).astype(np.uint64)
        return (((u.astype(np.uint64) + add) >> 16).astype(np.uint32) << 16).view(np.float32)
    m = outward(b.min(axis=1), False)
    d = outward((b.max(axis=1) - m) / np.float32(15), True)
    ratio = np.divide(b - m[:, None], d[:, None], out=np.zeros_like(b), where=d[:, None] != 0)
    q = np.floor(ratio + np.float32(.5)).clip(0, 15)
    return (m[:, None] + q * d[:, None]).reshape(v.shape)


def q8_requant(v):
    b = np.asarray(v, np.float32).reshape(-1, 32)
    s = bf16(np.max(np.abs(b), axis=1) / np.float32(127))
    ratio = np.divide(b, s[:, None], out=np.zeros_like(b), where=s[:, None] != 0)
    q = np.copysign(np.floor(np.abs(ratio) + np.float32(.5)), ratio).clip(-127, 127)
    return (q * s[:, None]).reshape(v.shape)


def kept_q4(raw, typ):
    """Decode Q4 codes with the q4_1 law's BF16 scale and offset."""
    raw = np.asarray(raw, np.uint8)
    shape = raw.shape
    if typ in ("Q4_0", "Q4_1"):
        size = 18 if typ == "Q4_0" else 20
        a = raw.reshape(-1, size)
        d = a[:, :2].copy().view('<f2').astype(np.float32)
        m = -8 * d if typ == "Q4_0" else a[:, 2:4].copy().view('<f2').astype(np.float32)
        qs = a[:, 2 if typ == "Q4_0" else 4:]
        q = np.concatenate((qs & 15, qs >> 4), axis=1).astype(np.float32)
        return (bf16(d) * q + bf16(m)).reshape(*shape[:-1], -1)
    if typ != "Q4_K":
        raise ValueError(f"unsupported kept type {typ}")
    a = raw.reshape(-1, 144)
    d = a[:, :2].copy().view('<f2').astype(np.float32)
    dm = a[:, 2:4].copy().view('<f2').astype(np.float32)
    s = a[:, 4:16].reshape(-1, 3, 4)
    sc = np.concatenate((s[:, 0] & 63, (s[:, 2] & 15) | ((s[:, 0] >> 2) & 48)), axis=1)
    mn = np.concatenate((s[:, 1] & 63, (s[:, 2] >> 4) | ((s[:, 1] >> 2) & 48)), axis=1)
    q = ((a[:, 16:].reshape(-1, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15).reshape(-1, 8, 32).astype(np.float32)
    out = bf16(d * sc.astype(np.float32))[:, :, None] * q - bf16(dm * mn.astype(np.float32))[:, :, None]
    return out.reshape(*shape[:-1], -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('src'); p.add_argument('dst'); p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--gguf-py', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.gguf_py)
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    from gguf.quants import dequantize
    if Path(a.src).resolve() == Path(a.dst).resolve():
        p.error('source and destination must differ')
    r = GGUFReader(a.src)
    arch = r.fields['general.architecture'].contents()
    w = GGUFWriter(a.dst, arch, use_temp_file=True)
    for name, f in r.fields.items():
        if name.startswith('GGUF.') or name == 'general.architecture': continue
        t = f.types[0]
        if t == GGUFValueType.ARRAY: w.add_key_value(name, f.contents(), t, sub_type=f.types[-1])
        else: w.add_key_value(name, f.contents(), t)
    names = {t.name for t in r.tensors}
    tied = 'output.weight' not in names
    for t in r.tensors:
        ty = t.tensor_type.name
        head = t.name == 'output.weight'
        projection = bool(PROJECTION.fullmatch(t.name))
        law = ('q8' if head else 'refit' if projection and a.mode == 'refit' and ty in ('Q4_K','Q6_K','Q8_0','Q5_K') else
               'kept' if projection and ty in ('Q4_0','Q4_1','Q4_K') else
               'refit' if projection and ty == 'Q6_K' and a.mode == 'q6-q4' else
               'q8' if projection and ty == 'Q6_K' and a.mode == 'q6-int8' else None)
        if projection and ty not in ('Q4_0', 'Q4_1', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_0'):
            raise ValueError(f'{t.name}: unsupported projection type {ty}')
        if projection and ty in ('Q5_K', 'Q8_0') and a.mode != 'refit':
            raise ValueError(f'{t.name}: {ty} carriage is not implemented')
        if law is None:
            if projection: print(f'{t.name} {ty} unchanged relative_l2=0', flush=True)
            w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
        else:
            v = dequantize(t.data, t.tensor_type).astype(np.float32)
            out = kept_q4(t.data, ty) if law == 'kept' else q4_refit(v) if law == 'refit' else q8_requant(v)
            out = out.reshape(v.shape)
            err = np.linalg.norm((out-v).ravel().astype(np.float64)) / max(np.linalg.norm(v.ravel().astype(np.float64)), 1e-30)
            print(f'{t.name} {ty} {law} relative_l2={err:.8g}', flush=True)
            w.add_tensor(t.name, out)
        if tied and t.name == 'token_embd.weight':
            v = dequantize(t.data, t.tensor_type).astype(np.float32)
            out = q8_requant(v)
            err = np.linalg.norm((out-v).ravel().astype(np.float64)) / max(np.linalg.norm(v.ravel().astype(np.float64)), 1e-30)
            print(f'output.weight (tied) {ty} q8 relative_l2={err:.8g}', flush=True)
            w.add_tensor('output.weight', out)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()


if __name__ == '__main__':
    main()
