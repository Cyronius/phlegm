"""Dump actual ArtLite prefill logits through its own NPU head, with seq control.

Quality-only instrumentation, not a speed measurement. The native chunk planner
and prefill operators run unchanged. Snapshot final PX residual rows before the
next chunk overwrites PX, then run ArtLite's own NPU LM head on each real row.
No CPU-head substitution and no sequential-attention substitution are allowed.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time


def fingerprint(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--ids-file', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--sequential-out', required=True, type=Path)
    ap.add_argument('--max-ctx', type=int, default=16384)
    ap.add_argument('--kv-k', choices=('single', 'hilo'), default='single')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    ids = [int(v) for v in args.ids_file.read_text().replace(',', ' ').split()]
    if not ids or min(ids) < 0 or len(ids) > args.max_ctx or args.threads < 1:
        ap.error('invalid tokens/capacity/threads')
    if args.out.exists() or args.sequential_out.exists() or args.out.resolve() == args.sequential_out.resolve():
        ap.error('use two distinct fresh output directories')
    # Locate the shared provenance helper whether this reviewed script is kept
    # in target/npu-dev/parallel-pipeline or moved into utilities/gguf-prefill.
    repo = next((p for p in Path(__file__).resolve().parents
                 if (p / 'utilities/gguf-prefill/artlite_worker.py').is_file()), None)
    if repo is None:
        ap.error('cannot locate Phlegm provenance helper')
    sys.path.insert(0, str(repo / 'utilities/gguf-prefill'))
    # Match the survey worker: use the ArtLite venv's installed compiler/AIE
    # stack rather than inheriting Phlegm's separately sourced toolchain.
    for name in ['PEANO_INSTALL_DIR', 'MLIR_AIE_INSTALL_DIR']:
        os.environ.pop(name, None)
    os.environ['PATH'] = str(Path(sys.executable).absolute().parent) + os.pathsep + os.environ['PATH']
    import numpy as np
    from artlite.gguf_model import GGUFFile
    from artlite.operators.llama.prefill import LlamaPrefill
    from artlite.operators.overlay.model import ModelOverlay
    from models.llama3.llama import NpuLlama
    from artlite_worker import provenance

    args.out.mkdir(parents=True)
    args.sequential_out.mkdir(parents=True)
    output = sys.stdout
    sys.stdout = sys.stderr
    meta = dict(route='native-prefill-residuals-with-own-NPU-head', quality_only=True,
                attention_substitution=False, cpu_head=False, expected_positions=len(ids),
                model_sha256=fingerprint(args.model), ids_sha256=fingerprint(args.ids_file),
                runner_sha256=fingerprint(Path(__file__)), provenance=provenance(),
                kv_k=args.kv_k, context_capacity=args.max_ctx, threads=args.threads,
                backend=os.environ.get('ARTLITE_BACKEND', 'default'), chunks=[],
                source_files={inspect.getfile(t): fingerprint(Path(inspect.getfile(t)))
                              for t in [NpuLlama, LlamaPrefill, ModelOverlay]},
                created_unix=time.time(), completed=False)
    covered = set()
    model = None
    original_run = original_steps = None
    final_dump = None

    def save_logits(logits, directory, position):
        # logits_of returns a view overwritten by the next head dispatch.
        values = logits.float().clone().numpy().astype('<f4', copy=False)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise RuntimeError(f'bad logits at position{position}')
        values.tofile(directory / f'logits_p{position}.bin')
        return values.copy()

    try:
        model = NpuLlama(GGUFFile(str(args.model.resolve())), capacity=args.max_ctx,
                         kv_k=args.kv_k, threads=args.threads, prefill=True)
        lm = model.overlay.image.region(-1, 'lm')
        meta['head'] = dict(source_tensor='output.weight' if 'output.weight' in model.gguf else 'token_embd.weight',
                            tile_formats=sorted({block.fmt for block in lm.blocks}),
                            implementation='ModelOverlay.logits_of/LlamaOverlay._lm_phase, including output_norm')
        model.prepare(len(ids))
        model.reset()
        model.stale()
        original_run, original_steps = model._prefill_run, model.steps

        def capture(args_tuple, T, n):
            nonlocal final_dump
            p0 = model.position
            original_run(args_tuple, T, n)  # Runs native prefill, then marks overlay stale.
            # Fetch/copy all rows before any subsequent chunk can reuse PX.
            hidden = [model.prefill.hidden(i).clone() for i in range(n)]
            chunk = dict(position=p0, padded_tokens=T, real_tokens=n,
                         prefill_dispatches=model.prefill.dispatches)
            meta['chunks'].append(chunk)
            for i, row in enumerate(hidden):
                pos = p0 + i
                if pos in covered:
                    raise RuntimeError(f'duplicate prefill position{pos}')
                logits = model.cap(model._dispatch(lambda row=row: model.overlay.logits_of(row)).float())
                dumped = save_logits(logits, args.out, pos)
                if pos == len(ids) - 1:
                    final_dump = dumped
                covered.add(pos)
            print(json.dumps(dict(captured_chunk=chunk, covered_positions=len(covered))), file=sys.stderr, flush=True)

        def reject_decode_fallback(*unused_args, **unused_kwargs):
            raise RuntimeError('native planner requested decode fallback; refusing to label it prefill')

        model._prefill_run = capture
        model.steps = reject_decode_fallback
        try:
            model.prompt(ids, logits=False)
        finally:
            model._prefill_run = original_run
            model.steps = original_steps
        if covered != set(range(len(ids))) or final_dump is None:
            raise RuntimeError('incomplete real-prefill position coverage')
        meta['positions'] = len(covered)
        # Extra head dispatches must not change later prefill behavior/state.
        # Compare the final captured head with the uninstrumented public route.
        model.reset()
        model.stale()
        normal_last = model.forward(ids).float().clone().numpy().astype('<f4', copy=False)
        meta['last_matches_uninstrumented_forward'] = bool(np.array_equal(final_dump, normal_last))
        normal_last.tofile(args.out / 'verify_last_forward.bin')
        if not meta['last_matches_uninstrumented_forward']:
            raise RuntimeError('instrumented last logits differ from native forward')
        # Same loaded weights, own head and cache precision: only the route changes.
        model.reset()
        model.stale()
        for i, token in enumerate(ids):
            save_logits(model.step(token), args.sequential_out, i)
        meta['completed'] = True
        meta['sequential_control'] = str(args.sequential_out.resolve())
        (args.sequential_out / 'metadata.json').write_text(json.dumps(
            dict(route='sequential', positions=len(ids), kv_k=args.kv_k,
                 context_capacity=args.max_ctx, cpu_head=False, model_sha256=meta['model_sha256'],
                 paired_prefill_metadata=str((args.out / 'metadata.json').resolve())), indent=2) + '\n')
    except Exception as e:
        meta['error'] = repr(e)
        meta['positions'] = len(covered)
        raise
    finally:
        meta['finished_unix'] = time.time()
        (args.out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), file=output)


if __name__ == '__main__':
    main()
