"""Paired diagnostic timing of already numerically checked harness GEMMs.

Not a whole-model benchmark. Configurations must describe the same arithmetic
problem; numerical checks are a separate prerequisite. Retains invalid pairs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def snapshot():
    p = subprocess.run(['xrt-smi', 'examine', '-r', 'platform'], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError('xrt-smi failed')
    f = subprocess.run(['fuser', '/dev/accel/accel0'], capture_output=True, text=True)
    return dict(platform=p.stdout, host_load=Path('/proc/loadavg').read_text().strip(),
                npu_clients=f.stdout.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--driver', required=True, type=Path)
    ap.add_argument('--reference-cfg', required=True, type=Path)
    ap.add_argument('--candidate-cfg', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--reps', type=int, default=60)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--timing-only-ablation', action='store_true', help='candidate outputs are wrong; timing attribution only')
    args = ap.parse_args()
    if args.reps < 20 or args.rounds < 3:
        ap.error('at least20 reps and3 rounds required')
    args.out.mkdir(parents=True, exist_ok=False)
    meta = dict(description=__doc__, driver_sha256=digest(args.driver),
                reps=args.reps, discarded_per_process=4, rounds=args.rounds, artifacts={},
                candidate_outputs_valid=not args.timing_only_ablation)
    configs = {}
    for name, path in [('reference', args.reference_cfg), ('candidate', args.candidate_cfg)]:
        lines, hashes, runs = [], {}, []
        for line in path.read_text().splitlines():
            words = line.split()
            if not words:
                continue
            if words[0] == 'run':
                runs.append(line)
                continue
            if words[0] == 'dump':
                continue
            if words[0] == 'buf' and len(words) == 4:
                p = (path.parent / words[3]).resolve()
                words[3] = str(p)
                hashes[str(p)] = digest(p)
            if words[0] in ('kernelx', 'xclbin'):
                p = Path(words[-1]).resolve()
                hashes[str(p)] = digest(p)
            lines.append(' '.join(words))
        if not runs or len(set(runs)) != 1:
            raise RuntimeError('requires one repeated GEMM run command')
        lines.extend([runs[0]] * args.reps)
        configs[name] = args.out / f'{name}.cfg'
        configs[name].write_text('\n'.join(lines) + '\n')
        meta['artifacts'][name] = hashes
    (args.out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    count = 0
    def trial(name, kind, round_index):
        nonlocal count
        before = snapshot()
        run = subprocess.run([str(args.driver.resolve()), str(configs[name].resolve())],
                             capture_output=True, text=True)
        after = snapshot()
        log = args.out / f'{count:03d}-{kind}-{name}.log'
        count += 1
        log.write_text(run.stdout + run.stderr)
        samples = re.findall(r'-> state (\d+) \(([\d.]+) ms\)', run.stdout + run.stderr)
        if run.returncode or len(samples) != args.reps or any(s[0] != '4' for s in samples):
            raise RuntimeError(f'failed run: {log}')
        row = dict(variant=name, kind=kind, round=round_index,
                   samples_ms=[float(s[1]) for s in samples], before=before, after=after)
        row['median_ms'] = statistics.median(row['samples_ms'][4:])
        row['environment_valid'] = (before['platform'] == after['platform']
                                    and not before['npu_clients'] and not after['npu_clients'])
        with (args.out / 'raw.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        return row
    pairs = []
    for i in range(args.rounds):
        pre = trial('reference', 'control_before', i)
        order = ('reference', 'candidate') if i % 2 == 0 else ('candidate', 'reference')
        rows = {name: trial(name, 'measured', i) for name in order}
        post = trial('reference', 'control_after', i)
        drift = abs(post['median_ms'] / pre['median_ms'] - 1)
        env_ok = all(x['environment_valid'] for x in [pre, *rows.values(), post])
        env_ok &= len({x['before']['platform'] for x in [pre, *rows.values(), post]}) == 1
        pair = dict(round=i, valid=bool(drift <= .05 and env_ok), control_drift=drift,
                    reference_ms=rows['reference']['median_ms'], candidate_ms=rows['candidate']['median_ms'],
                    candidate_speedup=rows['reference']['median_ms'] / rows['candidate']['median_ms'])
        pairs.append(pair)
        print(json.dumps(pair), flush=True)
    valid = [p for p in pairs if p['valid']]
    result = dict(pairs=pairs, valid_pairs=len(valid),
                  median_speedup=statistics.median(p['candidate_speedup'] for p in valid) if valid else None)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
