"""CPU feasibility study; not an NPU benchmark or a model-quality gate.

Run with the ArtLite environment: uses its host BFP encoder without copying it.
Compare single, three-product, and four-product BFP against dense BF16 operands.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from artlite.operators import bfp16


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--k', type=int, default=2048)
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--seeds', default='212,213,214')
    args = ap.parse_args()
    if min(args.n, args.k, args.tokens) <= 0 or args.k % 32:
        ap.error('positive dimensions and K divisible by32 required')
    torch.set_num_threads(4)
    results = []
    for seed in map(int, args.seeds.split(',')):
        rng = np.random.default_rng(seed)
        shape = (args.n, args.k // 32)
        # Same source distribution as check_gemm.py, independent dense reference.
        d = rng.uniform(.001, .03, shape).astype(np.float16)
        m = rng.uniform(-.12, .03, shape).astype(np.float16)
        q = rng.integers(0, 16, (*shape, 32), dtype=np.uint8)
        source = (q.astype(np.float64) * d.astype(np.float64)[..., None]
                  + m.astype(np.float64)[..., None]).reshape(args.n, args.k)
        def bf(x):
            return torch.from_numpy(np.asarray(x, np.float32)).to(torch.bfloat16).float().numpy()
        w = bf((q.astype(np.float32) * bf(d)[..., None]
                + bf(m)[..., None]).reshape(args.n, args.k))
        x = bf(rng.standard_normal((args.tokens, args.k)))
        wh, xh = bfp16.values(w), bfp16.values(x)
        wl, xl = bfp16.values(bf(w - wh)), bfp16.values(bf(x - xh))
        # FP32 accumulation after each64-K tile models our retained accumulator
        # policy. Does not model register pressure, scheduling, or DMA layout.
        outs = {name: np.zeros((args.n, args.tokens), np.float32)
                for name in ('single', 'three', 'four')}
        for k0 in range(0, args.k, 64):
            sl = slice(k0, k0 + 64)
            hh = wh[:, sl] @ xh[:, sl].T
            hl = wh[:, sl] @ xl[:, sl].T
            lh = wl[:, sl] @ xh[:, sl].T
            ll = wl[:, sl] @ xl[:, sl].T
            outs['single'] += hh
            outs['three'] += hh + hl + lh
            outs['four'] += hh + hl + lh + ll
        ref = w.astype(np.float64) @ x.astype(np.float64).T
        src = source @ x.astype(np.float64).T
        def rel(a, b):
            return float(np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(b))
        results.append(dict(seed=seed, modes={name: dict(
            operand_rel_fro=rel(y, ref), source_rel_fro=rel(y, src),
            passed=bool(np.isfinite(y).all() and rel(y, ref) <= .0001 and rel(y, src) <= .005))
            for name, y in outs.items()}))
    encoder = Path(inspect.getfile(bfp16))
    report = dict(kind='cpu-feasibility-only', n=args.n, k=args.k, tokens=args.tokens,
                  encoder=str(encoder), encoder_sha256=hashlib.sha256(encoder.read_bytes()).hexdigest(),
                  numpy=np.__version__, torch=torch.__version__, results=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
