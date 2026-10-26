"""Gate complete teacher-forced NPU dumps against independent CPU logits.

Uses the issue-212 gate: median cosine >= .9999, at most one percentage
point less argmax agreement than the sequential control, loss <= control * 1.005.
Counts the argmax allowance with integers so an exact one-point change passes.
The explicit turbo profile uses .9990, five percentage points, and 1.01.
CPU-reference forward KL is reported for both routes, without a KL gate.
"""
import argparse
import json
from pathlib import Path

import numpy as np


PROFILES = {
    'strict': dict(cosine_median_min=.9999, argmax_drop_percentage_points_max=1,
                   next_token_loss_ratio_max=1.005),
    'turbo': dict(cosine_median_min=.9990, argmax_drop_percentage_points_max=5,
                  next_token_loss_ratio_max=1.01),
}


def compare(cpu_path, sequential, candidate, ids, profile='strict'):
    if profile not in PROFILES:
        raise ValueError(f'unknown quality profile: {profile}')
    thresholds = PROFILES[profile].copy()
    n = len(ids)
    cpu = np.fromfile(cpu_path, '<f4')
    if n < 2 or cpu.size % n:
        raise ValueError('CPU shape does not match token count')
    cpu = cpu.reshape(n, -1).astype(np.float64)
    vocab = cpu.shape[1]
    if not vocab or not np.isfinite(cpu).all():
        raise ValueError('empty or nonfinite CPU reference')
    if any(not isinstance(token, (int, np.integer)) or not 0 <= token < vocab for token in ids):
        raise ValueError('token IDs outside CPU vocabulary')
    results = dict(profile=profile, thresholds=thresholds)
    for name, prefix in [('sequential', sequential), ('candidate', candidate)]:
        prefix = Path(prefix)
        expected = {f'{prefix.name}_p{i}.bin' for i in range(n)}
        actual = {p.name for p in prefix.parent.glob(f'{prefix.name}_p*.bin')}
        if actual != expected:
            raise ValueError(f'{name}: incomplete or extra position coverage')
        cosines, losses, flips, kls = [], [], [], []
        for i in range(n):
            logits = np.fromfile(prefix.parent / f'{prefix.name}_p{i}.bin', '<f4').astype(np.float64)
            if len(logits) < vocab or not np.isfinite(logits).all():
                raise ValueError(f'{name}: bad logits at {i}')
            logits = logits[:vocab]
            norm_product = np.linalg.norm(logits) * np.linalg.norm(cpu[i])
            if not np.isfinite(norm_product) or norm_product <= 0:
                raise ValueError(f'{name}: undefined cosine at {i}')
            cosines.append(float(logits @ cpu[i] / norm_product))
            # Shift before subtracting logsumexp to avoid cancellation for
            # large common offsets. KL direction is CPU reference || route.
            logq = logits - logits.max()
            logq -= np.log(np.exp(logq).sum())
            logp = cpu[i] - cpu[i].max()
            logp -= np.log(np.exp(logp).sum())
            kls.append(max(0., float(np.exp(logp) @ (logp - logq))))
            if logits.argmax() != cpu[i].argmax():
                best = np.sort(cpu[i])[-2:]
                flips.append(dict(position=i, cpu_margin=float(best[-1] - best[-2])))
            if i + 1 < n:
                m = logits.max()
                losses.append(float(np.log(np.exp(logits - m).sum()) + m - logits[ids[i + 1]]))
        if not np.isfinite(cosines + losses + kls).all():
            raise ValueError(f'{name}: nonfinite quality metrics')
        results[name] = dict(positions=n, argmax_equal=n - len(flips), flips=flips,
            cosine_median=float(np.median(cosines)), cosine_min=min(cosines),
            next_token_loss=float(np.mean(losses)),
            cpu_forward_kl_median=float(np.median(kls)),
            cpu_forward_kl_mean=float(np.mean(kls)), cpu_forward_kl_max=max(kls))
    base, cand = results['sequential'], results['candidate']
    results['passed'] = (cand['cosine_median'] >= thresholds['cosine_median_min']
        and 100 * (base['argmax_equal'] - cand['argmax_equal'])
            <= n * thresholds['argmax_drop_percentage_points_max']
        and cand['next_token_loss'] <= base['next_token_loss'] * thresholds['next_token_loss_ratio_max'])
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu', required=True, type=Path)
    ap.add_argument('--sequential', required=True, type=Path)
    ap.add_argument('--candidate', required=True, type=Path)
    ap.add_argument('--ids-file', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--profile', choices=PROFILES, default='strict')
    args = ap.parse_args()
    ids = [int(v) for v in args.ids_file.read_text().replace(',', ' ').split()]
    result = compare(args.cpu, args.sequential, args.candidate, ids, profile=args.profile)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
