"""Paired development engine-only trials of two Phlegm routes.

Both resident workers use the same bridge and GGUF with saved, identical prompt
and forced decode tokens. This is phlegm-engine-survey-v1, not HTTP timing or a
parity acceptance. Compilation, loading, resets, IPC and inventory are untimed.
All samples, discarded pairs and failed stability controls are retained.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from paired_survey import Worker, fingerprint

PROTOCOL = 'phlegm-engine-survey-v1'
ROUTES = ('reference', 'candidate')


def token_list(value, where):
    if not isinstance(value, list) or not value or any(type(t) is not int or t < 0 for t in value):
        raise ValueError(f'{where}: expected a nonempty list of nonnegative integer token IDs')
    return value


def load_inputs(path, sizes, max_ctx):
    """Validate saved paired_survey inputs before opening any worker/NPU context."""
    if not sizes or len(sizes) != len(set(sizes)) or min(sizes) < 256:
        raise ValueError('sizes must be unique and at least256 for the256-token controls')
    selected = {}
    wanted = set(sizes)
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f'line{line_no}: expected a JSON object')
            if record.get('kind') != 'inputs' or record.get('length') not in wanted:
                continue
            length = record['length']
            if length in selected:
                raise ValueError(f'duplicate inputs record for length{length}')
            prompt = token_list(record.get('prompt'), f'line{line_no} prompt')
            trajectory = token_list(record.get('trajectory'), f'line{line_no} trajectory')
            if len(prompt) != length or len(trajectory) != 128:
                raise ValueError(f'length{length}: need exactly{length} prompt and128 trajectory tokens')
            if length + len(trajectory) - 1 > max_ctx:
                raise ValueError(f'length{length}: prompt plus127 forced steps exceed capacity{max_ctx}')
            selected[length] = dict(prompt=prompt, trajectory=trajectory)
    missing = wanted - set(selected)
    if missing:
        raise ValueError(f'no inputs records for sizes{sorted(missing)}')
    return selected


def platform():
    return subprocess.check_output(['xrt-smi', 'examine', '-r', 'platform'], text=True).strip()


def npu_clients():
    probe = subprocess.run(['fuser', '/dev/accel/accel0'], capture_output=True, text=True)
    if probe.returncode not in (0, 1):
        raise RuntimeError('cannot inventory NPU clients: ' + probe.stderr)
    clients = []
    for pid in probe.stdout.split():
        if pid.isdigit():
            try:
                name = (Path('/proc') / pid / 'comm').read_text().strip()
            except FileNotFoundError:
                name = '(exited)'
            clients.append(dict(pid=int(pid), name=name))
    return clients


def artifact_files(kernel_dir):
    """Hash actual contexts/streams in both manifest sections, not directory names."""
    manifest_path = kernel_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    paths = {manifest_path.resolve()}
    for section in [manifest, *([manifest['gguf']] if 'gguf' in manifest else [])]:
        for value in section.get('contexts', {}).values():
            if not isinstance(value, str):
                raise ValueError('unsupported non-path context descriptor')
            paths.add((kernel_dir / value).resolve())
        for kernel in section.get('kernels', {}).values():
            if 'insts' in kernel:
                paths.add((kernel_dir / kernel['insts']).resolve())
    for name in ['toolchain.json', 'bfp-hilo-probe.json']:
        path = kernel_dir / name
        if path.exists():
            paths.add(path.resolve())
    return paths



def weight_and_tail_contract(kernel_dir):
    """Select the source-GGUF route and verify model/head equality across A/B."""
    whole = json.loads((kernel_dir / 'manifest.json').read_text())
    m = whole.get('gguf', whole)
    contract = {k: m.get(k) for k in ['family', 'spec', 'spec_hash', 'hf_config_check',
                                      'hf_config_defaults', 'layout', 'layers', 'pack', 'tail']}
    contract['layer_weights'] = {name: {k: value.get(k) for k in ['pool', 'consts']}
        | {'gemm_weights': value.get('gemm_block', {}).get('weights')}
        for name, value in m.get('layer_types', {}).items()}
    payloads = {}
    for step in m.get('tail', []):
        if step.get('op') != 'run':
            continue
        name = step['kernel']
        kernel = m['kernels'][name]
        payloads[name] = dict(instructions=fingerprint(kernel_dir / kernel['insts']),
            context=fingerprint(kernel_dir / m['contexts'][kernel['context']]))
    contract['tail_payload_sha256'] = payloads
    return contract


def hashes(paths):
    return {str(p): fingerprint(p) for p in sorted(paths)}



def host_load_status(samples, maximum):
    """Check both snapshots of every measured/control row; no bound means unchecked."""
    if maximum is None:
        return dict(host_idle=None, max_observed_host_load=None, host_load_violations=[])
    values, violations = [], []
    for row in samples:
        for key in ['load_before', 'load_after']:
            raw = row.get(key)
            detail = dict(route=row.get('route', row.get('engine')), kind=row.get('kind'), snapshot=key)
            try:
                value = float(str(raw).split()[0])
                if not math.isfinite(value) or value < 0:
                    raise ValueError('invalid load')
            except (ValueError, IndexError):
                violations.append(dict(**detail, raw=raw, reason='host_load_unavailable'))
                continue
            values.append(value)
            if value > maximum:
                violations.append(dict(**detail, load1=value, reason='host_load_above_limit'))
    return dict(host_idle=not violations,
                max_observed_host_load=max(values) if values else None,
                host_load_violations=violations)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bridge', required=True, type=Path)
    ap.add_argument('--model-dir', required=True, type=Path)
    ap.add_argument('--reference-kernels', required=True, type=Path)
    ap.add_argument('--candidate-kernels', required=True, type=Path)
    ap.add_argument('--inputs-jsonl', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--sizes', default='1024,8192')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--max-ctx', type=int, default=16384)
    ap.add_argument('--max-host-load', type=float, help='reject pairs when any before/after load1 exceeds this positive finite bound')
    ap.add_argument('--reverse-init', action='store_true',
                    help='initialize candidate before reference; keep trial order and route labels unchanged')
    args = ap.parse_args()
    initialization_order = tuple(reversed(ROUTES)) if args.reverse_init else ROUTES
    try:
        sizes = [int(x) for x in args.sizes.split(',')]
        if args.rounds < 1 or args.threads < 1 or args.max_ctx < 1:
            raise ValueError('rounds, threads and capacity must be positive')
        if args.max_host_load is not None and (not math.isfinite(args.max_host_load) or args.max_host_load <= 0):
            raise ValueError('max-host-load must be positive and finite')
        inputs = load_inputs(args.inputs_jsonl, sizes, args.max_ctx)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        ap.error(str(e))
    if args.out.exists():
        ap.error('output directory already exists; use a fresh path')
    build_path = args.bridge.with_suffix('.build.json')
    bridge_build = json.loads(build_path.read_text())
    bridge_sha = fingerprint(args.bridge)
    if bridge_build.get('binary_sha256') != bridge_sha:
        ap.error('bridge build sidecar binary hash does not match bridge')
    kernel_dirs = dict(reference=args.reference_kernels, candidate=args.candidate_kernels)
    critical_paths = {args.bridge.resolve(), build_path.resolve(), args.inputs_jsonl.resolve(),
                      (args.model_dir / 'model.gguf').resolve(), Path(__file__).resolve(),
                      Path(__file__).with_name('paired_survey.py').resolve()}
    paths_by_route = {name: artifact_files(path) for name, path in kernel_dirs.items()}
    contracts = {name: weight_and_tail_contract(path) for name, path in kernel_dirs.items()}
    if contracts['reference'] != contracts['candidate']:
        ap.error('reference and candidate differ in model/weight/head/tail contract; use matched bundles')
    for paths in paths_by_route.values():
        critical_paths.update(paths)
    # Hashing precedes worker creation/timing; recheck after all timed trials end.
    before_hashes = hashes(critical_paths)
    root = Path(__file__).resolve().parents[2]
    source_paths = [root / p for p in [
        'benchmarks/reference/bridge.cpp', 'benchmarks/reference/build.py',
        'open_kernels/designs/gemm_q4_prefill/gemm_bfp_compensated_probe.cc',
        'open_kernels/designs/gemm_q4_prefill/gemm_bfp_compensated_emit.h',
        'utilities/gguf-prefill/build_bfp_stream_probe.py',
        'utilities/gguf-prefill/prepare_bfp_hilo_image.py']]
    working_source_hashes = hashes({p.resolve() for p in source_paths if p.exists()})
    args.out.mkdir(parents=True)
    raw_path = args.out / 'raw.jsonl'
    workers, records, pairs = {}, [], []
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('OFLM_', 'OMP_'))}
    env.update(OMP_WAIT_POLICY='PASSIVE', OMP_NUM_THREADS=str(args.threads))
    meta = dict(protocol=PROTOCOL, development=True, engine_only=True,
                comparison='same-engine reference versus candidate route',
                worker_initialization_order=list(initialization_order),
                matched_weight_and_tail_contract=contracts['reference'],
                created_unix=time.time(), context_capacity=args.max_ctx, output_tokens=128,
                control_prompt_tokens=256, control_output_tokens=16,
                control_trajectory_note='first15 forced tokens from smallest selected saved128-token trajectory; no generation is timed',
                threads=args.threads, prefill_route='auto', max_host_load=args.max_host_load, bridge_sha256=bridge_sha,
                bridge_build=bridge_build, bridge_build_sha256=fingerprint(build_path),
                model_sha256=before_hashes[str((args.model_dir / 'model.gguf').resolve())],
                inputs_jsonl_sha256=fingerprint(args.inputs_jsonl), runner_sha256=fingerprint(Path(__file__)),
                helper_sha256=fingerprint(Path(__file__).with_name('paired_survey.py')),
                artifacts_before=before_hashes, working_tree_source_files=working_source_hashes,
                source_note='bridge_build source_files identify compiled runtime sources; working_tree_source_files are separate present-tree snapshots',
                routes={name: dict(kernel_dir=str(path.resolve()),
                                   manifest_sha256=fingerprint(path / 'manifest.json'),
                                   artifacts=hashes(paths_by_route[name]),
                                   toolchain=json.loads((path / 'toolchain.json').read_text())
                                   if (path / 'toolchain.json').exists() else None)
                        for name, path in kernel_dirs.items()})

    def record(value):
        records.append(value)
        with raw_path.open('a') as f:
            f.write(json.dumps(value) + '\n')

    def write_meta():
        (args.out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')

    def trial(route, prompt, forced, kind, length, round_index):
        before, load_before = platform(), Path('/proc/loadavg').read_text().strip()
        clients_before = npu_clients()
        reply = workers[route].request(command='survey_trial', prompt=prompt, decode_tokens=forced)
        after, clients_after = platform(), npu_clients()
        if (reply.get('survey_protocol') != PROTOCOL or reply.get('prompt_tokens') != len(prompt)
                or reply.get('decode_steps') != len(forced) or reply.get('output_tokens') != len(forced) + 1
                or len(reply.get('greedy_tokens', [])) != len(forced) + 1):
            raise RuntimeError('survey response counts/protocol disagree')
        sample = reply['sample']
        for metric in ['prefill_ms', 'decode_ms']:
            if not isinstance(sample.get(metric), (int, float)) or not math.isfinite(sample[metric]) or sample[metric] <= 0:
                raise RuntimeError('nonfinite or nonpositive timing returned')
        if len(sample.get('decode_token_ms', [])) != len(forced):
            raise RuntimeError('decode per-step sample count disagrees')
        known = {w.process.pid for w in workers.values()}
        result = dict(engine=route, route=route, kind=kind, length=length, round=round_index,
                      platform_before=before, platform_after=after,
                      load_before=load_before, load_after=Path('/proc/loadavg').read_text().strip(),
                      clients_before=clients_before, clients_after=clients_after,
                      clients_known=all(c['pid'] in known for c in clients_before + clients_after),
                      platform_stable=before == after, **reply)
        record(result)
        return result

    smallest = inputs[min(sizes)]
    control_prompt, control_forced = smallest['prompt'][:256], smallest['trajectory'][:15]
    for length in sizes:
        record(dict(kind='inputs', length=length, **inputs[length]))
    record(dict(kind='control_inputs', prompt=control_prompt, decode_tokens=control_forced, output_tokens=16))
    failure = None
    try:
        for name in initialization_order:
            workers[name] = Worker([str(args.bridge.resolve())], env, args.out / f'{name}.stderr')
            init = workers[name].request(command='init', model_dir=str(args.model_dir.resolve()),
                kernel_dir=str(kernel_dirs[name].resolve()), max_ctx=args.max_ctx,
                dispatch_timeout_ms=120000, prefill_mode='fast', prefill_route='auto')
            meta['routes'][name].update(init=init, hello=workers[name].hello)
        meta['worker_pids'] = {name: w.process.pid for name, w in workers.items()}
        meta['platform'] = platform()
        meta['clients_after_init'] = npu_clients()
        write_meta()
        for length in sizes:
            prompt, trajectory = inputs[length]['prompt'], inputs[length]['trajectory']
            forced = trajectory[:-1]
            for name in ROUTES:
                trial(name, prompt, forced, 'warmup', length, -1)
            # One complete bracketed pair is explicitly discarded after warmups.
            for i in range(-1, args.rounds):
                order = ROUTES if i < 0 or i % 2 == 0 else tuple(reversed(ROUTES))
                discarded = i < 0
                prefix = 'discarded_' if discarded else ''
                pre = {e: trial(e, control_prompt, control_forced, prefix + 'control_before', length, i) for e in order}
                measured = {e: trial(e, prompt, forced, prefix + 'measured', length, i) for e in order}
                post = {e: trial(e, control_prompt, control_forced, prefix + 'control_after', length, i) for e in order}
                drift = {e: {metric: abs(post[e]['sample'][metric] / pre[e]['sample'][metric] - 1)
                             for metric in ['prefill_ms', 'decode_ms']} for e in order}
                samples = [group[e] for group in [pre, measured, post] for e in order]
                host = host_load_status(samples, args.max_host_load)
                reasons = []
                if not all(max(drift[e].values()) <= .05 for e in order):
                    reasons.append('control_drift_above_5_percent')
                if not all(x['platform_stable'] for x in samples) or len({x['platform_before'] for x in samples}) != 1:
                    reasons.append('platform_changed')
                if not all(x['clients_known'] for x in samples):
                    reasons.append('unknown_npu_clients')
                if host['host_idle'] is False:
                    reasons.extend(sorted({v['reason'] for v in host['host_load_violations']}))
                valid = not reasons
                pair = dict(kind='paired_result', length=length, round=i, discarded=discarded,
                            valid=valid, control_drift=drift, reason=';'.join(reasons) if reasons else 'ok',
                            invalid_reasons=reasons, max_host_load=args.max_host_load, **host,
                            prefill_ratio=measured['reference']['sample']['prefill_ms'] / measured['candidate']['sample']['prefill_ms'],
                            decode_ratio=measured['reference']['sample']['decode_ms'] / measured['candidate']['sample']['decode_ms'],
                            reference_sample=measured['reference']['sample'], candidate_sample=measured['candidate']['sample'])
                pairs.append(pair)
                record(pair)
                print(f'{length} round{i}: discarded={discarded} stable={valid}', flush=True)
        after_hashes = hashes(critical_paths)
        meta['artifacts_after'] = after_hashes
        meta['artifacts_unchanged'] = after_hashes == before_hashes
        if not meta['artifacts_unchanged']:
            record(dict(kind='artifact_change', valid=False, before=before_hashes, after=after_hashes))
        summary = {}
        for length in sizes:
            valid = [p for p in pairs if p['length'] == length and p['valid'] and not p['discarded']
                     and meta['artifacts_unchanged']]
            summary[length] = dict(valid_pairs=len(valid), requested_pairs=args.rounds,
                median_prefill_ratio=statistics.median(p['prefill_ratio'] for p in valid) if valid else None,
                median_decode_ratio=statistics.median(p['decode_ratio'] for p in valid) if valid else None,
                reference_prefill_tps=statistics.median(length * 1000 / p['reference_sample']['prefill_ms'] for p in valid) if valid else None,
                candidate_prefill_tps=statistics.median(length * 1000 / p['candidate_sample']['prefill_ms'] for p in valid) if valid else None,
                reference_decode_tps=statistics.median(127000 / p['reference_sample']['decode_ms'] for p in valid) if valid else None,
                candidate_decode_tps=statistics.median(127000 / p['candidate_sample']['decode_ms'] for p in valid) if valid else None)
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
        meta['completed'] = True
    except Exception as e:
        failure = e
        meta['completed'] = False
        meta['error'] = repr(e)
        record(dict(kind='error', error=repr(e)))
        raise
    finally:
        meta['finished_unix'] = time.time()
        write_meta()
        cleanup_errors = []
        for name, worker in reversed(list(workers.items())):
            try:
                worker.close()
            except Exception as e:
                cleanup_errors.append(dict(route=name, error=repr(e)))
        if cleanup_errors:
            meta['cleanup_errors'] = cleanup_errors
            write_meta()
            if failure is None:
                raise RuntimeError(f'worker cleanup failed: {cleanup_errors}')


if __name__ == '__main__':
    main()
