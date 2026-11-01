r"""A control GGUF for the Qwen3.5 / 3.6 MoE GGUF view (OPEN-FAMILY-QWEN36MOE-GGUF): the same
file with every tensor the runtime's q35_* ops REQUANTIZE replaced by the values the NPU pool
then holds, stored F32, so llama.cpp on it computes with the NPU's weights.

  q35_mat / q35_stripes / q35_down on a Q8_0 / Q5_0 / Q6_K tensor: pools.cpp q35_put's q4_1 law
      (mu = bf16_floor(min), du = bf16_ceil((max - mu) / 15), q = floor((v - mu) / du + 0.5))
  q35_lmhead on a non-Q8_0 head: scale = bf16_ceil(amax / 127), q = round(v / scale)
  Q4_0 / Q4_1 / Q4_K tensors (kept exactly up to the bf16 scale narrowing) and the F32 small
  tensors are copied byte for byte.

llama.cpp on the original GGUF vs on this one is the cost of the requantization; this one vs the
NPU is the NPU's arithmetic.

    python open_kernels/model/qwen35moe_gguf_npulike.py IN.gguf OUT.gguf --gguf-py ~/llama.cpp/gguf-py
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

REQUANT = {"Q8_0", "Q5_0", "Q6_K"}
MATS = re.compile(r"blk\.\d+\.(ffn_(up|gate|down)_(exps|shexp)|attn_(qkv|gate|q|k|v|output)|ssm_out)\.weight$")


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def bf16_floor(x):
    u = _bits(x)
    r = np.where(u >> 31, (u.astype(np.uint64) + 0xFFFF) >> 16, u >> 16).astype(np.uint32) << 16
    return r.view(np.float32)


def bf16_ceil(x):
    u = _bits(x)
    r = np.where(u >> 31, u >> 16, (u.astype(np.uint64) + 0xFFFF) >> 16).astype(np.uint32) << 16
    return r.view(np.float32)


def q4_1_law(v: np.ndarray) -> np.ndarray:
    b = v.reshape(-1, 32).astype(np.float32)
    mf = bf16_floor(b.min(1))
    d = bf16_ceil(((b.max(1) - mf) / np.float32(15)).astype(np.float32))
    inv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, 1), 0).astype(np.float32)
    q = np.clip(np.floor((b - mf[:, None]) * inv[:, None] + np.float32(0.5)), 0, 15)
    return (mf[:, None] + q * d[:, None]).astype(np.float32).reshape(v.shape)


def q8_law(v: np.ndarray) -> np.ndarray:
    b = v.reshape(-1, 32).astype(np.float32)
    s = bf16_ceil((np.abs(b).max(1) / np.float32(127)).astype(np.float32))
    inv = np.where(s > 0, np.float32(1) / np.where(s > 0, s, 1), 0).astype(np.float32)
    q = np.clip(np.rint(b * inv[:, None]), -127, 127)     # lround vs rint differ only on exact halves
    return (q * s[:, None]).astype(np.float32).reshape(v.shape)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--gguf-py", default=None)
    a = ap.parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    from gguf.quants import dequantize
    r = GGUFReader(a.src)
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(a.dst, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        t = f.types[0]
        if t == GGUFValueType.ARRAY:
            w.add_key_value(name, f.contents(), t, sub_type=f.types[-1])
        else:
            w.add_key_value(name, f.contents(), t)
    changed = 0
    for t in r.tensors:
        ty = t.tensor_type.name
        shape = tuple(int(x) for x in reversed(t.shape))
        law = None
        if ty in REQUANT and MATS.search(t.name):
            law = q4_1_law
        elif t.name == "output.weight" and ty != "Q8_0":
            law = q8_law
        if law is None:
            w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
            continue
        v = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape)
        w.add_tensor(t.name, law(v))
        changed += 1
        print(f"{t.name}: {ty} -> F32 ({law.__name__})", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"{changed} tensors replaced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
