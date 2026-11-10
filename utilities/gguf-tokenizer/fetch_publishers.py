#!/usr/bin/env python3
"""Fetch public publisher tokenizer data only; run with dibs network/python -I."""
import argparse
import hashlib
import json
import pathlib
import urllib.request
import urllib.error

REPOS = {
    'llama32-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama32-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama31-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen3-06b': 'Qwen/Qwen3-0.6B',
    'qwen3-4b': 'Qwen/Qwen3-4B',
    'qwen3-8b': 'Qwen/Qwen3-8B',
    'gemma4-12b': 'google/gemma-4-12B-it',
    'qwen35-9b': 'Qwen/Qwen3.5-9B',
    'gemma4-e4b': 'google/gemma-4-E4B-it',
}
MIRRORS = {
    'llama32-1b': 'unsloth/Llama-3.2-1B-Instruct',
    'llama32-3b': 'unsloth/Llama-3.2-3B-Instruct',
    'llama31-8b': 'unsloth/Meta-Llama-3.1-8B-Instruct',
}


def fetch(root, mirrors=False):
    for name, repo in (MIRRORS if mirrors else REPOS).items():
        dest = root / name / ('mirror-tokenizer' if mirrors else 'publisher-tokenizer')
        dest.mkdir(exist_ok=False)
        source = dict(repo=repo)
        try:
            with urllib.request.urlopen('https://huggingface.co/api/models/' + repo, timeout=60) as r:
                source['revision'] = json.load(r)['sha']
            url = f"https://huggingface.co/{repo}/resolve/{source['revision']}/tokenizer.json"
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise ValueError('tokenizer exceeds 64 MiB')
            (dest / 'tokenizer.json').write_bytes(data)
            source['sha256'] = hashlib.sha256(data).hexdigest()
            source['bytes'] = len(data)
            print(name, 'OK', len(data), flush=True)
        except (urllib.error.URLError, ValueError) as exc:
            source['unavailable'] = str(exc)
            print(name, 'UNAVAILABLE', exc, flush=True)
        (dest / 'source.json').write_text(json.dumps(source, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=pathlib.Path)
    p.add_argument('--mirrors', action='store_true')
    args = p.parse_args()
    fetch(args.root, args.mirrors)
