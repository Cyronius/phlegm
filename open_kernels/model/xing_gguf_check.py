r"""Xing 4.0's packed pools against gguf-py (OPEN-FAMILY-XING): the samples xing_pack_check wrote (every sampled
weight of the packed layer pools and of the head: tensor, file row, file column, the value read back out of the
pool through the Q4_0-native / f32-scale q8 chunk laws) against llama.cpp's own dequantization of the GGUF
(`gguf.quants.dequantize`), which shares no code with the runtime's packer or its checker. A Q4_0 value must be
bit-exact (every scale of the file is a bf16 value); the Q8_0 head's fp16 scales are widened to f32 exactly.

    python -I open_kernels/model/xing_gguf_check.py --gguf model.gguf --samples samples.tsv [--gguf-py <tree>/gguf-py]
"""
from __future__ import annotations

import argparse
import collections
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--gguf-py", default=None)
    a = ap.parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(a.gguf)
    tens = {t.name: t for t in r.tensors}
    by: dict[str, list[tuple[int, int, float]]] = collections.defaultdict(list)
    for line in open(a.samples):
        name, row, col, v = line.rstrip("\n").split("\t")
        by[name].append((int(row), int(col), float(v)))
    bad = total = 0
    worst = 0.0
    for name, rows in sorted(by.items()):
        t = tens[name]
        cols = int(t.shape[0])
        data = np.asarray(t.data).reshape(-1)
        nbytes_row = data.size // (int(np.prod([int(d) for d in t.shape])) // cols)
        cache: dict[int, np.ndarray] = {}
        for row, col, got in rows:
            if row not in cache:
                cache[row] = dequantize(data[row * nbytes_row:(row + 1) * nbytes_row], t.tensor_type).reshape(-1)
            want = float(np.float32(cache[row][col]))
            total += 1
            if np.float32(got) != np.float32(want):
                bad += 1
                worst = max(worst, abs(got - want))
                if bad <= 10:
                    print(f"MISMATCH {name} row {row} col {col}: pool {got!r} gguf-py {want!r}")
        print(f"{name} ({t.tensor_type.name}): {len(rows)} samples")
    print(f"{total} samples over {len(by)} tensors: {bad} differ (worst {worst:g})")
    print("ALL OK" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
