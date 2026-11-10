#!/usr/bin/env python3
"""Offline S6 gate. Run with python -I under dibs cpu; no model inference.

The reference is a separately built llama.cpp vocab_only helper, not Phlegm's
tokenizer. Publisher tokenizer.json comparisons use Python's native tokenizers.
"""
import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import tempfile
import unicodedata


def metadata(path):
    data = path.read_bytes()
    pos = 24

    def take(n):
        nonlocal pos
        if n < 0 or n > len(data) - pos:
            raise ValueError('truncated metadata')
        out = data[pos:pos+n]
        pos += n
        return out

    def num(fmt):
        return struct.unpack('<' + fmt, take(struct.calcsize(fmt)))[0]

    def string():
        return take(num('Q')).decode('utf-8')

    def value(ty):
        if ty == 8:
            return string()
        if ty == 9:
            elem, count = num('I'), num('Q')
            if count > 2000000 or elem == 9:
                raise ValueError('invalid metadata array')
            return [value(elem) for _ in range(count)]
        return num({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[ty])

    if data[:4] != b'GGUF':
        raise ValueError('not GGUF')
    out = {}
    for _ in range(struct.unpack_from('<Q', data, 16)[0]):
        key = string()
        out[key] = value(num('I'))
    return out


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {}
    expected = {}
    for folder in sorted(args.headers.iterdir()):
        if not (folder / 'source.json').exists():
            continue
        name = folder.name
        m = metadata(folder / 'header.gguf')
        with tempfile.TemporaryDirectory(prefix='phlegm-s6-') as tmp:
            pathlib.Path(tmp, 'published.gguf').symlink_to(folder / 'header.gguf')
            probe = subprocess.run([str(args.probe), tmp, str(args.corpus)], capture_output=True, text=True)
        if probe.returncode:
            summary[name] = {'error': probe.stderr.strip()}
            print(name, 'FAIL', probe.stderr.strip(), flush=True)
            continue
        result = json.loads(probe.stdout)
        cases = result['cases']
        requests = [dict(text=c['text'], add_special=flag) for c in cases for flag in (False, True)]
        requests.append(dict(stop_ids=True))
        reference = subprocess.run([str(args.reference), str(folder / 'vocab.gguf')],
            input=''.join(json.dumps(r) + '\n' for r in requests), text=True, capture_output=True)
        if reference.returncode:
            summary[name] = {'reference_error': reference.stderr.strip()}
            print(name, 'REFERENCE FAIL', reference.stderr.strip(), flush=True)
            continue
        lines = [json.loads(line) for line in reference.stdout.splitlines()]
        assert len(lines) == len(requests)
        ids, stop_ids = lines[:-1], lines[-1]['stop_ids']
        stop_mismatch = sorted(result['stop_ids']) != stop_ids
        mismatches = []
        for i, case in enumerate(cases):
            for j, key in enumerate(('ids', 'with_special')):
                if case[key] != ids[2*i+j]:
                    mismatches.append(dict(case=case['name'], mode=key, actual=case[key], expected=ids[2*i+j]))
        publisher = None
        publisher_path = folder / 'publisher-tokenizer' / 'tokenizer.json'
        if not publisher_path.exists():
            publisher_path = folder / 'mirror-tokenizer' / 'tokenizer.json'
        nfc_control = None
        if publisher_path.exists():
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(str(publisher_path))
            publisher = []
            for case in cases:
                actual = tk.encode(case['text'], add_special_tokens=False).ids
                if actual != case['ids']:
                    publisher.append(dict(case=case['name'], actual=case['ids'], expected=actual))
            publisher_config = json.loads(publisher_path.read_text())
            if publisher and publisher_config.get('normalizer') == {'type': 'NFC'}:
                texts = {c['name']:c['text'] for c in cases}
                control = subprocess.run([str(args.reference), str(folder / 'vocab.gguf')],
                    input=''.join(json.dumps(dict(text=unicodedata.normalize('NFC', texts[c['case']]),
                                  add_special=False)) + '\n' for c in publisher), text=True, capture_output=True, check=True)
                normalized_ids = [json.loads(line) for line in control.stdout.splitlines()]
                assert len(normalized_ids) == len(publisher)
                nfc_control = {c['case']: c['expected'] == ids for c,ids in zip(publisher, normalized_ids)}
        detail = dict(source=json.loads((folder / 'source.json').read_text()),
                      metadata={k:v for k,v in m.items() if k.startswith('tokenizer.') and not isinstance(v,list)},
                      cases=cases, mismatches=mismatches, publisher_mismatches=publisher,
                      stop_ids=stop_ids, stop_ids_mismatch=stop_mismatch,
                      publisher_source=json.loads(publisher_path.with_name('source.json').read_text()) if publisher_path.exists() else None,
                      nfc_control_matches_publisher=nfc_control)
        detail_path = args.output / (name + '.json')
        detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + '\n')
        summary[name] = dict(cases=len(cases), llama_mismatches=len(mismatches),
            stop_ids_mismatch=stop_mismatch,
            publisher_mismatches=None if publisher is None else len(publisher),
            nfc_control_matches_publisher=nfc_control,
            evidence_sha256=hashlib.sha256(detail_path.read_bytes()).hexdigest())
        expected[name] = dict(source=detail['source'], stop_ids=stop_ids, cases=[dict(name=c['name'], text=c['text'],
                              ids=ids[2*i], with_special=ids[2*i+1]) for i,c in enumerate(cases)])
        print(name, summary[name], flush=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'expected.json').write_text(json.dumps(expected, ensure_ascii=False, separators=(',', ':')) + '\n')
    if any('error' in v or 'reference_error' in v or v.get('llama_mismatches') or v.get('publisher_mismatches') or v.get('stop_ids_mismatch') for v in summary.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headers', type=pathlib.Path, required=True)
    p.add_argument('--probe', type=pathlib.Path, required=True)
    p.add_argument('--reference', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--corpus', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2] / 'phlegm-core/tests/fixtures/gguf-tokenizer/corpus.json')
    run(p.parse_args())
