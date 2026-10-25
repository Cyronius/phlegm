"""Check an f32-scale GGUF projection GEMM against independent dense products.

Artifacts stay in --out. No score here is a full-model quality or speed claim.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'open_kernels'))
sys.path.insert(0, str(ROOT / 'npu_offload/gemm_rtp'))
from gguf_pool import pack_q4_pool
from npue import tile_b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kernel-dir', required=True, type=Path)
    ap.add_argument('--driver', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--k', type=int, default=256)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--seed', type=int, default=212)
    ap.add_argument('--activation-format', choices=('bf16', 'bfp-hilo', 'bfp-hilo-pairs'), default='bf16')
    args = ap.parse_args()
    n, k, t = args.n, args.k, args.tokens
    if min(n, k, t) <= 0 or any(v % 256 for v in (n, k, t)):
        ap.error('N, K and tokens must be positive multiples of 256')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # Construct GGUF bytes from independent codes/scales, including signed mins
    # and scales whose fp16 values change when the kernel narrows them to bf16.
    d = rng.uniform(.001, .03, (n, k // 32)).astype(np.float16)
    m = rng.uniform(-.12, .03, d.shape).astype(np.float16)
    q = rng.integers(0, 16, (n, k // 32, 32), dtype=np.uint8)
    raw = np.empty((n, k // 32, 20), np.uint8)
    raw[..., :2] = d.view(np.uint8).reshape(n, k // 32, 2)
    raw[..., 2:4] = m.view(np.uint8).reshape(n, k // 32, 2)
    raw[..., 4:] = q[..., :16] | (q[..., 16:] << 4)
    pool = pack_q4_pool(raw, rs=2)
    x = rng.standard_normal((t, k)).astype(bfloat16)
    tiled = tile_b(np.ascontiguousarray(x.T).view(np.uint16), 64, 32, 8, 8, order='k,n')
    pool.tofile(out / 'weights.bin')
    if args.activation_format != 'bf16':
        # Probe-only dependency; no ArtLite import for the accepted BF16 check.
        from artlite.operators import bfp16
        # tile_b is [K64][T32][K8][T8][k][t]; panels need rows of8 K.
        v = tiled.reshape(k // 64, t // 32, 8, 4, 8, 8).transpose(0, 1, 3, 2, 5, 4)
        v = v.view(bfloat16).astype(np.float32)
        hi = bfp16.block_bytes(v)
        lo = bfp16.block_bytes((v - bfp16.block_values(hi)).astype(bfloat16).astype(np.float32))
        # [K64][T32][token16][K8][token8][hi,lo][row8][9bytes].
        if args.activation_format == 'bfp-hilo-pairs':
            shape = (k // 64, t // 32, 2, 2, 8, 8, 9)
            hi = hi.reshape(shape).transpose(0, 1, 2, 4, 3, 5, 6)
            lo = lo.reshape(shape).transpose(0, 1, 2, 4, 3, 5, 6)
        tiled = np.ascontiguousarray(np.stack([hi, lo], axis=-3)).reshape(-1)
    tiled.tofile(out / 'input.bin')
    # No inverse of the pool packer participates in either numerical reference.
    exact = (q.astype(np.float64) * d.astype(np.float64)[..., None]
             + m.astype(np.float64)[..., None]).reshape(n, k)
    narrowed = (q.astype(np.float32) * d.astype(bfloat16).astype(np.float32)[..., None]
                + m.astype(bfloat16).astype(np.float32)[..., None]).astype(bfloat16).reshape(n, k)
    reference = exact @ x.astype(np.float64).T
    operand_reference = narrowed.astype(np.float64) @ x.astype(np.float64).T
    kernel = args.kernel_dir.resolve()
    cfg = ['device', f'xclbin G {kernel / "final.xclbin"}',
           f'kernelx k G {kernel / "insts.bin"}',
           f'buf w {pool.nbytes} weights.bin', f'buf x {tiled.nbytes} input.bin',
           f'buf y {n * t * 4}', 'run k w x y', 'run k w x y',
           f'dump y output.bin {n * t * 4}']
    (out / 'run.cfg').write_text('\n'.join(cfg) + '\n')
    run = subprocess.run([str(args.driver.resolve()), str(out / 'run.cfg')],
                         capture_output=True, text=True)
    (out / 'driver.log').write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(f'driver failed ({run.returncode}); see {out / "driver.log"}')
    got = np.fromfile(out / 'output.bin', '<f4')
    if got.size != n * t or not np.isfinite(got).all():
        raise RuntimeError('incomplete or nonfinite GEMM output')
    got = got.reshape(n, t).astype(np.float64)
    rel = float(np.linalg.norm(got - reference) / np.linalg.norm(reference))
    operand_rel = float(np.linalg.norm(got - operand_reference) / np.linalg.norm(operand_reference))
    result = dict(n=n, k=k, tokens=t, seed=args.seed, activation_format=args.activation_format, rel_fro=rel,
                  operand_rel_fro=operand_rel, passed=rel <= .005 and operand_rel <= .0001)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
