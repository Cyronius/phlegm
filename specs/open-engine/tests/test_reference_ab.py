import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[3] / 'benchmarks/reference/ab.py'


def test_raw_bridge_record_and_compare(tmp_path):
    bridge = tmp_path / 'bridge'
    bridge.write_text('''#!/usr/bin/env python3
import json, os, struct, sys
out = sys.stdout.buffer
pos = 0
mark = 0
def reply(v, data=b''):
    out.write((json.dumps(dict(ok=True, **v)) + "\\n").encode() + data); out.flush()
reply({})
for line in sys.stdin:
    r = json.loads(line); cmd = r['command']
    if cmd == 'quit': reply({}); break
    if cmd == 'init': reply(dict(real_vocab=3, prefill_route=r['prefill_route']))
    elif cmd == 'reset': pos = 0; reply({})
    elif cmd == 'rewind': pos = r['pos']; reply(dict(position=pos))
    elif cmd == 'prefill_logits':
        pos += len(r['prompt']); mark = r.get('mark', pos)
        reply(dict(mark=mark, logits_f32_le_bytes=12), struct.pack('<3f', pos, 1, 2))
    elif cmd == 'decode_logits':
        pos += 1
        reply(dict(logits_f32_le_bytes=12), struct.pack('<3f', pos + int(os.environ.get('AB_PLANT', '0')), 1, 2))
''')
    bridge.chmod(0o755)
    tokens = tmp_path / 'tokens.ids'; tokens.write_text('1,2,3,4')
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps([dict(model='m', kernels='k', route='auto', ids=str(tokens), length=4)]))
    first, second, report = [tmp_path / n for n in ('one.json', 'two.json', 'report.json')]
    def run(*args, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], env=env, capture_output=True, text=True)
    for out in (first, second):
        result = run('record', '--bridge', bridge, '--cases', cases, '--out', out)
        assert result.returncode == 0, result.stderr
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    assert [r['sha256'] for r in a] == [r['sha256'] for r in b]
    assert len(a) == 10 and a[-1]['sha256'] == a[1]['sha256']
    import os
    result = run('compare', '--bridge', bridge, '--golden', first, '--cases', cases, '--out', report, '--rounds', '1', env={**os.environ, 'AB_PLANT': '1'})
    assert result.returncode == 1 and 'MISMATCH' in result.stdout

    paired = run('compare', '--bridge', bridge, '--other', bridge, '--cases', cases, '--out', report, '--rounds', '2')
    assert paired.returncode == 0, paired.stderr
    assert paired.stdout.count('0 mismatches') == 2
