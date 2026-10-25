"""Dump ArtLite's sequential teacher-forced logits for the source quality gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from artlite.gguf_model import GGUFFile
from models.llama3.llama import NpuLlama

from artlite_worker import provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--ids-file', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--max-ctx', type=int, default=16384)
    ap.add_argument('--kv-k', choices=('single', 'hilo'), default='single')
    args = ap.parse_args()
    ids = [int(v) for v in args.ids_file.read_text().replace(',', ' ').split()]
    if not ids or len(ids) > args.max_ctx:
        ap.error('invalid token count/capacity')
    args.out.mkdir(parents=True, exist_ok=False)
    output = sys.stdout
    sys.stdout = sys.stderr
    model = NpuLlama(GGUFFile(str(args.model.resolve())), capacity=args.max_ctx,
                     kv_k=args.kv_k, threads=4, prefill=False)
    model.reset()
    model.stale()
    for i, token in enumerate(ids):
        model.step(token).float().numpy().tofile(args.out / f'logits_p{i}.bin')
    with args.model.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    meta = dict(provenance=provenance(), model_sha256=digest, positions=len(ids),
                route='sequential', kv_k=args.kv_k, context_capacity=args.max_ctx)
    (args.out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), file=output)


if __name__ == '__main__':
    main()
