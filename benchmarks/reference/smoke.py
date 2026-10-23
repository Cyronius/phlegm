"""A short engine correctness check, not a publishable benchmark or score export."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--kernels', type=Path, required=True)
    parser.add_argument('--ids', default='1,2,3,4')
    parser.add_argument('--prefill-route', choices=('auto', 'sequential', 'block'), default='auto')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    ids = [int(v) for v in args.ids.split(',')]
    requests = [
        dict(command='init', model_dir=str(args.model_dir.resolve()), kernel_dir=str(args.kernels.resolve()),
             max_ctx=64, dispatch_timeout_ms=60000, prefill_mode='fast', prefill_route=args.prefill_route),
        dict(command='reset'), dict(command='prefill', prompt=ids),
        dict(command='trial', prompt=ids, output_tokens=8),
        dict(command='reset'), dict(command='prefill', prompt=ids), dict(command='quit'),
    ]
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('OFLM_', 'OMP_'))}
    env.update(OMP_WAIT_POLICY='PASSIVE', OMP_NUM_THREADS=str(os.cpu_count() or 1))
    result = subprocess.run([str(args.bridge.resolve())], input=''.join(json.dumps(r) + '\n' for r in requests),
                            text=True, capture_output=True, timeout=args.timeout, env=env)
    if result.returncode:
        raise RuntimeError(f'bridge failed ({result.returncode}): {result.stdout}\n{result.stderr}')
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(replies) == len(requests) + 1 and all(r.get('ok') for r in replies), replies
    assert replies[0]['protocol'] == 1
    assert replies[0]['engine'] == 'openflowlm-cpp-reference'
    assert replies[3]['token'] == replies[6]['token'], 'reset did not reproduce the first greedy token'
    trial = replies[4]
    assert trial['prompt_tokens'] == len(ids) and trial['output_tokens'] == 8 and trial['decode_steps'] == 7
    sample = trial['sample']
    assert sample['prefill_ms'] > 0 and sample['decode_ms'] > 0
    assert len(sample['decode_token_ms']) == 7 and all(t >= 0 for t in sample['decode_token_ms'])
    assert sum(sample['decode_token_ms']) <= sample['decode_ms'] + 1e-6
    print(f"Smoke check passed: resident engine, {replies[1]['prefill_route']} prefill, complete decode, reproducible reset.")


if __name__ == '__main__':
    main()
