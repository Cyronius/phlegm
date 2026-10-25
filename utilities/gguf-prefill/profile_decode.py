"""Synthetic DX attention-compute ablation, not an inference benchmark.

ATTN_NULL outputs are wrong. Both variants retain the same weight/KV traffic;
the cache starts zero-filled and a single layer repeatedly consumes a fixed token.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cli', required=True, type=Path)
    ap.add_argument('--model-dir', required=True, type=Path)
    ap.add_argument('--kernels', required=True, type=Path)
    ap.add_argument('--null-dx', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--positions', default='0,1024,4096,8192')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--reps', type=int, default=200)
    args = ap.parse_args()
    positions = [int(v) for v in args.positions.split(',')]
    if min(positions) < 0 or max(positions) >= 16384 or args.rounds < 1 or args.reps < 200:
        ap.error('invalid position/round/repetition count')
    args.out.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.kernels / 'manifest.json').read_text())
    for variant in ('native', 'null'):
        m = json.loads(json.dumps(original))
        for section in [m, m['gguf']]:
            for key, value in section['contexts'].items():
                section['contexts'][key] = str((args.kernels / value).resolve())
            for kernel in section['kernels'].values():
                kernel['insts'] = str((args.kernels / kernel['insts']).resolve())
            for layer in section['layer_types'].values():
                for key in ('gemm_block', 'gemm_block_variants'):
                    layer.pop(key, None)
        if variant == 'null':
            m['gguf']['contexts']['dx'] = str((args.null_dx / 'final.xclbin').resolve())
            m['gguf']['kernels']['dx']['insts'] = str((args.null_dx / 'insts.bin').resolve())
        d = args.out / variant
        d.mkdir()
        (d / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    meta = dict(description=__doc__, cli_sha256=digest(args.cli),
                source_manifest_sha256=digest(args.kernels / 'manifest.json'),
                null_xclbin_sha256=digest(args.null_dx / 'final.xclbin'),
                null_instructions_sha256=digest(args.null_dx / 'insts.bin'),
                threads=4, context_capacity=16384, layers=1, repetitions=args.reps)
    (args.out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OFLM_', 'OMP_'))}
    env.update(OMP_NUM_THREADS='4', OMP_WAIT_POLICY='PASSIVE')
    count = 0
    records = []
    def trial(variant, pos, kind, round_index):
        nonlocal count
        command = [str(args.cli.resolve()), '--model', str(args.model_dir.resolve()),
                   '--kernels', str((args.out / variant).resolve()), '--ids', '128000',
                   '--layers', '1', '--max-ctx', '16384', '--max-tokens', '1',
                   '--decode-route', 'npu', '--pmode', 'none', '--at-position', str(pos),
                   '--bench-kernel', f'dx:{args.reps}:0']
        r = subprocess.run(command, capture_output=True, text=True, env=env)
        log = args.out / f'{count:03d}-{kind}-{variant}-p{pos}.log'
        count += 1
        log.write_text(r.stdout + r.stderr)
        match = re.search(r'dx on layer 0, \d+ reps: ([\d.]+) min, ([\d.]+) mean, ([\d.]+) submit', r.stderr)
        if r.returncode or not match:
            raise RuntimeError(f'profile failed: inspect {log}')
        row = dict(variant=variant, position=pos, kind=kind, round=round_index,
                   min_ms=float(match[1]), mean_ms=float(match[2]), submit_ms=float(match[3]),
                   host_load=Path('/proc/loadavg').read_text().strip(), log=log.name)
        records.append(row)
        with (args.out / 'raw.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        return row
    summary = {}
    for pos in positions:
        for v in ('native', 'null'):
            trial(v, pos, 'warmup', -1)
        pairs = []
        for i in range(args.rounds):
            pre = trial('native', 0, 'control_before', i)
            order = ('native', 'null') if i % 2 == 0 else ('null', 'native')
            measured = {v: trial(v, pos, 'measured', i) for v in order}
            post = trial('native', 0, 'control_after', i)
            drift = abs(post['mean_ms'] / pre['mean_ms'] - 1)
            pair = dict(position=pos, round=i, valid=drift <= .05, control_drift=drift,
                        native_ms=measured['native']['mean_ms'], null_ms=measured['null']['mean_ms'])
            with (args.out / 'pairs.jsonl').open('a') as f:
                f.write(json.dumps(pair) + '\n')
            pairs.append(pair)
            print(json.dumps(pair), flush=True)
        good = [p for p in pairs if p['valid']]
        summary[pos] = dict(valid_pairs=len(good), requested_pairs=args.rounds,
                            native_ms=statistics.median(p['native_ms'] for p in good) if good else None,
                            null_ms=statistics.median(p['null_ms'] for p in good) if good else None)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
