#!/usr/bin/env python3
"""Record and compare raw bridge logits. Run real captures under Dibs and npu-run.sh.

`cases` prints capture commands and writes case lists, reporting missing coverage explicitly.
`record` and `compare` accept a JSON case list produced by `cases` or supplied by hand.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import time

BINARY = 'phlegm-reference-bench' + ('.exe' if os.name == 'nt' else '')
ROOT = Path(__file__).resolve().parents[2]
INDEX = json.loads((ROOT / 'bridges/index.json').read_text())
SOURCES = {'qwen36moe': 'qwen35moe-ref', 'qwen35': 'saluki/ref', 'flint': 'flint-ref',
           'gemma4': 'gemma4-26b-ref', 'gptoss': 'gptoss-ref', 'hyv3': 'hy-ref',
           'lfm2moe': 'lfm2moe-ref', 'lfm3': 'lfm3', 'nemotronh': 'nemotronh/ref',
           'llama3': 'nanbeige/ref', 'xing': 'xing/ref'}


def ids(path):
    return [int(x) for x in re.split(r'[\s,]+', Path(path).read_text().strip()) if x]


def boundaries(value, parent=''):
    result = set()
    if isinstance(value, dict):
        for key, v in value.items():
            if isinstance(v, int) and not isinstance(v, bool) and v > 1 and (
                'window' in key.lower() or (('block' in parent.lower() or 'block' in key.lower()) and key.lower() in ('t', 'block_t', 'block_size', 'block_length'))):
                result.add(v)
            result.update(boundaries(v, key))
    elif isinstance(value, list):
        for v in value:
            result.update(boundaries(v, parent))
    return result


class Bridge:
    def __init__(self, exe):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.receive()

    def receive(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('bridge closed stdout')
        reply = json.loads(line)
        if reply.get('ok') is False:
            raise RuntimeError(reply.get('error', 'bridge error'))
        n = reply.get('logits_f32_le_bytes', 0)
        payload = self.p.stdout.read(n)
        if len(payload) != n or n % 4:
            raise RuntimeError('incomplete logits payload')
        return reply, payload

    def request(self, command, **kwargs):
        self.p.stdin.write((json.dumps(dict(command=command, **kwargs)) + '\n').encode())
        self.p.stdin.flush()
        return self.receive()

    def close(self):
        try:
            self.request('quit')
        finally:
            self.p.wait(timeout=30)


def capture(exe, cases):
    result = []
    for case in cases:
        tokens = ids(case['ids'])[:case['length']]
        if len(tokens) != case['length'] or not tokens:
            raise ValueError('insufficient tokens: ' + str(case))
        bridge = Bridge(exe)
        try:
            info, _ = bridge.request('init', model_dir=case['model'], kernel_dir=case['kernels'],
                                     max_ctx=case.get('max_ctx', 32768), dispatch_timeout_ms=120000,
                                     prefill_route=case['route'], prefill_mode='fast')
            bridge.request('reset')
            start = time.perf_counter()
            reply, data = bridge.request('prefill_logits', prompt=tokens, mark=len(tokens))
            elapsed = time.perf_counter() - start
            def save(step, payload, seconds):
                if len(payload) != info['real_vocab'] * 4:
                    raise ValueError('logits size does not match real_vocab')
                result.append({**case, 'step': step, 'sha256': hashlib.sha256(payload).hexdigest(),
                               'real_vocab': info['real_vocab'], 'route_label': info['prefill_route'],
                               'seconds': seconds})
                values = struct.unpack('<%df' % info['real_vocab'], payload)
                return max(range(len(values)), key=values.__getitem__)
            prefill_payload = data
            token = save('prefill', data, elapsed)
            for step in range(1, 9):
                start = time.perf_counter()
                _, data = bridge.request('decode_logits', token=token)
                token = save('decode%d' % step, data, time.perf_counter() - start)
            bridge.request('rewind', pos=reply['mark'])
            # Block routes may round the mark down; replay the remaining prompt first.
            if reply['mark'] < len(tokens):
                _, prefill_payload = bridge.request('prefill_logits', prompt=tokens[reply['mark']:])
            first_token = max(range(info['real_vocab']), key=struct.unpack('<%df' % info['real_vocab'], prefill_payload).__getitem__)
            start = time.perf_counter()
            _, data = bridge.request('decode_logits', token=first_token)
            save('rewind_decode', data, time.perf_counter() - start)
        finally:
            bridge.close()
    return result


def key(row):
    return tuple(row[k] for k in ('model', 'kernels', 'route', 'ids', 'length', 'step'))


def compare(a, b):
    left, right = {key(r): r for r in a}, {key(r): r for r in b}
    mismatch = 0
    for k in sorted(left.keys() | right.keys()):
        x, y = left.get(k), right.get(k)
        if x is None or y is None or any(x[v] != y[v] for v in ('sha256', 'real_vocab', 'route_label')):
            print('MISMATCH', k, x and x['sha256'], y and y['sha256'])
            mismatch += 1
    print('%d mismatches across %d cases' % (mismatch, len(left.keys() | right.keys())))
    return mismatch


def generate(args):
    registry = json.loads((ROOT / 'phlegm-core/config/model_list.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group in registry['models'].values():
        for model in group.values():
            model_dir = args.models / model['name']
            manifest = model_dir / 'open_kernels/manifest.json'
            if not manifest.is_file():
                print('MISSING manifest', manifest, file=sys.stderr); continue
            family = json.loads(manifest.read_text())['family']
            source = args.dev / SOURCES[family]
            if family == 'qwen36moe':
                name = model['name'].lower()
                variant = next((x for x in ('clover', 'nex', 'ornith') if x in name), None)
                if variant:
                    source /= variant
                elif 'q4pure' in name:
                    source /= 'q4pure'
            files = sorted(source.glob('*.ids'))
            if family == 'llama3':
                files.append(ROOT / 'specs/open-engine/tests/fixtures/issue212/llama-context-8192.ids')
            profile = {'hyv3': 'hy', 'lfm2moe': 'lfm25'}.get(family, family)
            files += sorted((args.dev / 'prefill-prof').glob(profile + '-*.ids'))
            files = [p for p in files if p.is_file()]
            if not any('short' in p.name or 'chat' in p.name for p in files):
                print('MISSING short-token file for', model_dir, file=sys.stderr)
            if not any('long' in p.name or len(ids(p)) >= 1024 for p in files):
                print('MISSING long-token file for', model_dir, file=sys.stderr)
            if not any('forced' in p.name for p in files):
                print('MISSING forced-token file for', model_dir, file=sys.stderr)
            if not files:
                print('MISSING token files for', model_dir, file=sys.stderr); continue
            cases = []
            for kernel in [model_dir / 'open_kernels', model_dir / 'open_kernels.seq']:
                if not (kernel / 'manifest.json').is_file(): continue
                bounds = boundaries(json.loads((kernel / 'manifest.json').read_text()))
                for path in files:
                    n = len(ids(path))
                    lengths = {n} | {v for b in bounds for v in (b-1, b, b+1) if 0 < v <= n}
                    for length in sorted(lengths):
                        for route in ('auto', 'sequential'):
                            cases.append(dict(model=str(model_dir), kernels=str(kernel), route=route, ids=str(path), length=length))
                for b in sorted(bounds):
                    if not any(len(ids(p)) >= b+1 for p in files):
                        print('MISSING tokens around boundary', b, 'for', model_dir, file=sys.stderr)
            dest = args.output_dir / (model['name'] + '.json')
            dest.write_text(json.dumps(cases, indent=2) + '\n')
            bridge = args.dev / 'bridges' / INDEX[family] / BINARY
            golden = args.dev / 'one-runtime/golden' / (model['name'] + '.json')
            print(shlex.join(['python', str(Path(__file__).resolve()), 'record', '--bridge', str(bridge), '--cases', str(dest), '--out', str(golden)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cases = sub.add_parser('cases'); cases.add_argument('--models', type=Path, required=True); cases.add_argument('--dev', type=Path, required=True); cases.add_argument('--output-dir', type=Path, required=True)
    for name in ('record', 'compare'):
        p = sub.add_parser(name); p.add_argument('--bridge', required=True); p.add_argument('--cases', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
        if name == 'compare':
            group = p.add_mutually_exclusive_group(required=True); group.add_argument('--other'); group.add_argument('--golden', type=Path)
            p.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'cases': generate(args); return 0
    case_list = json.loads(args.cases.read_text())
    if args.command == 'record':
        args.out.write_text(json.dumps(capture(args.bridge, case_list), indent=2) + '\n'); return 0
    if args.rounds < 1: parser.error('--rounds must be positive')
    runs = []
    for rd in range(args.rounds):
        if args.golden:
            runs.append({'round': rd, 'bridge': capture(args.bridge, case_list), 'golden': json.loads(args.golden.read_text())})
        else:
            order = [('bridge', args.bridge), ('other', args.other)]
            if rd % 2: order.reverse()
            run = {'round': rd}
            for label, exe in order: run[label] = capture(exe, case_list)
            runs.append(run)
    args.out.write_text(json.dumps(runs, indent=2) + '\n')
    mismatches = [compare(r['bridge'], r.get('other', r.get('golden'))) for r in runs]
    return int(any(mismatches))


if __name__ == '__main__':
    sys.exit(main())
