#!/usr/bin/env python3
"""Fetch only GGUF headers, pinned by the public HF API; never fetch weights.

Run under dibs network with python -I. Each download gets a new directory.
"""
import argparse
import hashlib
import json
import pathlib
import struct
import urllib.request
import urllib.parse

TARGETS = {
    "llama32-1b": "unsloth/Llama-3.2-1B-Instruct-GGUF",
    "llama32-3b": "unsloth/Llama-3.2-3B-Instruct-GGUF",
    "llama31-8b": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    "qwen3-06b": "unsloth/Qwen3-0.6B-GGUF",
    "qwen3-4b": "unsloth/Qwen3-4B-GGUF",
    "qwen3-8b": "unsloth/Qwen3-8B-GGUF",
    "gemma4-12b": "lmstudio-community/gemma-4-12B-it-GGUF",
    "qwen35-9b": "lmstudio-community/Qwen3.5-9B-GGUF",
    "gemma4-e4b": "lmstudio-community/gemma-4-E4B-it-GGUF",
}


def header_end(data):
    """Return metadata/tensor-header boundaries, or EOFError for another range."""
    pos = 0

    def take(n):
        nonlocal pos
        if n > len(data) - pos:
            raise EOFError
        out = data[pos:pos + n]
        pos += n
        return out

    def number(fmt):
        return struct.unpack('<' + fmt, take(struct.calcsize(fmt)))[0]

    def string():
        return take(number('Q'))

    def value(ty, depth=0):
        if depth > 4:
            raise ValueError('nested array')
        if ty == 8:
            string()
        elif ty == 9:
            elem, count = number('I'), number('Q')
            if count > 10000000:
                raise ValueError('array too large')
            for _ in range(count):
                value(elem, depth + 1)
        else:
            take({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                  7: 1, 10: 8, 11: 8, 12: 8}[ty])

    if take(4) != b'GGUF' or number('I') not in (2, 3):
        raise ValueError('not GGUF v2/v3')
    tensors, metadata = number('Q'), number('Q')
    if tensors > 1000000 or metadata > 100000:
        raise ValueError('invalid counts')
    for _ in range(metadata):
        string()
        value(number('I'))
    metadata_end = pos
    for _ in range(tensors):
        string()
        dims = number('I')
        if not 0 < dims <= 8:
            raise ValueError('invalid tensor dimensions')
        take(dims * 8 + 12)
    return metadata_end, pos


def fetch(root, name, repo):
    dest = root / name
    dest.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen('https://huggingface.co/api/models/' + repo, timeout=60) as r:
        info = json.load(r)
    revision = info['sha']
    files = [s['rfilename'] for s in info['siblings']
             if s['rfilename'].lower().endswith('q4_k_m.gguf')]
    if len(files) != 1:
        raise ValueError(f'{repo}: expected one Q4_K_M file, got {files}')
    file = files[0]
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(file)}'
    data = bytearray()
    ranges = []
    while len(data) < 128 * 1024 * 1024:
        start, end = len(data), len(data) + 4 * 1024 * 1024 - 1
        req = urllib.request.Request(url + f'?s6_range={start}', headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.status != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                raise ValueError('server ignored Range; refusing a weight download')
            block = r.read(end - start + 2)
            if len(block) != end - start + 1:
                raise ValueError('unexpected range length')
            ranges.append(r.headers['Content-Range'])
            data.extend(block)
        try:
            meta_end, end = header_end(data)
            break
        except EOFError:
            pass
    else:
        raise ValueError('header exceeds 128 MiB')
    # Preserve the original header, and a separate lossless metadata-only GGUF
    # for llama.cpp vocab_only (zero tensor count, otherwise identical bytes).
    original = bytes(data[:end])
    vocab = bytearray(data[:meta_end])
    struct.pack_into('<Q', vocab, 8, 0)
    vocab.extend(b'\0' * (-len(vocab) % 32))
    (dest / 'header.gguf').write_bytes(original)
    (dest / 'vocab.gguf').write_bytes(vocab)
    source = dict(repo=repo, revision=revision, file=file, url=url, ranges=ranges,
                  header_bytes=end, metadata_bytes=meta_end,
                  header_sha256=hashlib.sha256(original).hexdigest(),
                  vocab_sha256=hashlib.sha256(vocab).hexdigest())
    (dest / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
    print(name, revision, file, end, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=pathlib.Path)
    parser.add_argument('--target', choices=TARGETS, action='append')
    args = parser.parse_args()
    failures = {}
    for name in args.target or TARGETS:
        try:
            fetch(args.root, name, TARGETS[name])
        except Exception as exc:
            failures[name] = str(exc)
            print(name, 'FAILED:', exc, flush=True)
    if failures:
        print(json.dumps(failures, indent=2))
        raise SystemExit(1)
