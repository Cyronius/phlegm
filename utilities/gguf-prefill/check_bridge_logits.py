"""Compare resident bridge logits with a CLI dump on the same greedy sequence."""
import argparse
import base64
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bridge', required=True, type=Path)
    ap.add_argument('--model-dir', required=True, type=Path)
    ap.add_argument('--kernels', required=True, type=Path)
    ap.add_argument('--ids-file', required=True, type=Path)
    ap.add_argument('--cli-prefix', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--steps', type=int, default=16)
    args = ap.parse_args()
    if args.steps < 1:
        ap.error('--steps must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ids = [int(v) for v in args.ids_file.read_text().replace(',', ' ').split()]
    if not ids or len(ids) + args.steps - 1 > 16384:
        ap.error('token sequence must fit the context capacity')
    with args.out.with_suffix('.stderr').open('w') as log:
        p = subprocess.Popen([str(args.bridge.resolve())], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, text=True)
        def read():
            r = json.loads(p.stdout.readline())
            if not r.get('ok'):
                raise RuntimeError(r)
            return r
        def request(**value):
            p.stdin.write(json.dumps(value) + '\n')
            p.stdin.flush()
            return read()
        read()
        init = request(command='init', model_dir=str(args.model_dir.resolve()),
                       kernel_dir=str(args.kernels.resolve()), max_ctx=16384,
                       dispatch_timeout_ms=120000, prefill_mode='fast', prefill_route='auto')
        result = []
        try:
            reply = request(command='prefill_logits', prompt=ids)
            for step in range(args.steps):
                logits = np.frombuffer(base64.b64decode(reply['logits_f32_le_base64']), '<f4')
                ref = np.fromfile(str(args.cli_prefix) + f'_t{step}.bin', '<f4')
                if ref.size < logits.size or not np.isfinite(logits).all():
                    raise ValueError('incomplete or nonfinite logits')
                ref = ref[:logits.size]
                result.append(dict(step=step, identical=np.array_equal(ref, logits),
                                   max_abs=float(np.abs(ref - logits).max())))
                if step + 1 < args.steps:
                    reply = request(command='decode_logits', token=int(ref.argmax()))
            request(command='quit')
            p.wait(timeout=60)
        finally:
            if p.poll() is None:
                p.stdin.close()
                p.wait(timeout=60)
    passed = all(r['identical'] for r in result)
    args.out.write_text(json.dumps(dict(init=init, results=result, passed=passed), indent=2) + '\n')
    print(json.dumps(dict(route=init['prefill_route'], steps=len(result), passed=passed)))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
