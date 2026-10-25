"""Gate complete teacher-forced NPU dumps against independent CPU logits.

Uses the issue-212 gate: median cosine >= .9999, at most one percentage
point less argmax agreement than the sequential control, loss <= control * 1.005.
Counts the argmax allowance with integers so an exact one-point change passes.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(cpu_path, sequential, candidate, ids):
    n = len(ids)
    cpu = np.fromfile(cpu_path, '<f4')
    if n < 2 or cpu.size % n:
        raise ValueError('CPU shape does not match token count')
    cpu = cpu.reshape(n, -1).astype(np.float64)
    vocab = cpu.shape[1]
    if not vocab or not np.isfinite(cpu).all():
        raise ValueError('empty or nonfinite CPU reference')
    results = {}
    for name, prefix in [('sequential', sequential), ('candidate', candidate)]:
        prefix = Path(prefix)
        expected = {f'{prefix.name}_p{i}.bin' for i in range(n)}
        actual = {p.name for p in prefix.parent.glob(f'{prefix.name}_p*.bin')}
        if actual != expected:
            raise ValueError(f'{name}: incomplete or extra position coverage')
        cosines, losses, flips = [], [], []
        for i in range(n):
            logits = np.fromfile(prefix.parent / f'{prefix.name}_p{i}.bin', '<f4').astype(np.float64)
            if len(logits) < vocab or not np.isfinite(logits).all():
                raise ValueError(f'{name}: bad logits at {i}')
            logits = logits[:vocab]
            cosines.append(float(logits @ cpu[i] / (np.linalg.norm(logits) * np.linalg.norm(cpu[i]))))
            if logits.argmax() != cpu[i].argmax():
                best = np.sort(cpu[i])[-2:]
                flips.append(dict(position=i, cpu_margin=float(best[-1] - best[-2])))
            if i + 1 < n:
                m = logits.max()
                losses.append(float(np.log(np.exp(logits - m).sum()) + m - logits[ids[i + 1]]))
        results[name] = dict(positions=n, argmax_equal=n - len(flips), flips=flips,
            cosine_median=float(np.median(cosines)), cosine_min=min(cosines),
            next_token_loss=float(np.mean(losses)))
    base, cand = results['sequential'], results['candidate']
    results['passed'] = (cand['cosine_median'] >= .9999
        and 100 * (base['argmax_equal'] - cand['argmax_equal']) <= n
        and cand['next_token_loss'] <= base['next_token_loss'] * 1.005)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu', required=True, type=Path)
    ap.add_argument('--sequential', required=True, type=Path)
    ap.add_argument('--candidate', required=True, type=Path)
    ap.add_argument('--ids-file', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    ids = [int(v) for v in args.ids_file.read_text().replace(',', ' ').split()]
    result = compare(args.cpu, args.sequential, args.candidate, ids)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
