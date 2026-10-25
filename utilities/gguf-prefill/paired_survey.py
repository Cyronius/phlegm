"""Alternating development engine trials on identical teacher-forced tokens.

This is phlegm-engine-survey-v1, not phlegm-bench-v1 or an HTTP TTFT test.
Compilation, model load, IPC, resets and environment collection are not timed.
All samples, including failed stability controls, remain in the output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


class Worker:
    def __init__(self, command, env, log):
        self.log = log.open('w')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, env=env, text=True, bufsize=1)
        self.hello = self.read()

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'worker {self.process.pid} closed output; inspect stderr log')
        result = json.loads(line)
        if not result.get('ok'):
            raise RuntimeError(result.get('error', result))
        return result

    def request(self, **request):
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            self.request(command='quit')
            self.process.wait(timeout=60)
        self.log.close()


def fingerprint(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bridge', required=True, type=Path)
    ap.add_argument('--artlite-python', required=True, type=Path)
    ap.add_argument('--model-dir', required=True, type=Path)
    ap.add_argument('--kernels', required=True, type=Path)
    ap.add_argument('--ids-file', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--sizes', default='1024,2048,4096,8192')
    ap.add_argument('--max-ctx', type=int, default=16384)
    ap.add_argument('--outputs', type=int, default=128)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--prefill-route', choices=('auto', 'block', 'sequential'), default='block')
    args = ap.parse_args()
    sizes = [int(v) for v in args.sizes.split(',')]
    ids = [int(v) for v in args.ids_file.read_text().replace(',', ' ').split()]
    if (min(sizes) < 1 or max(sizes) > len(ids) or args.outputs < 2 or args.rounds < 1
            or max(sizes) + args.outputs - 1 > args.max_ctx):
        ap.error('invalid lengths, rounds, or capacity')
    args.out.mkdir(parents=True, exist_ok=False)
    raw_path = args.out / 'raw.jsonl'
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('OFLM_', 'OMP_'))}
    env.update(OMP_WAIT_POLICY='PASSIVE', OMP_NUM_THREADS=str(args.threads))
    art_env = dict(env)
    for name in ('PEANO_INSTALL_DIR', 'MLIR_AIE_INSTALL_DIR'):
        art_env.pop(name, None)
    art_env['PATH'] = str(args.artlite_python.absolute().parent) + os.pathsep + env['PATH']
    art_env['ARTLITE_CACHE_DIR'] = str(args.out.resolve().parents[0] / 'artlite-survey-cache')
    workers = {}
    records = []

    def record(value):
        records.append(value)
        with raw_path.open('a') as f:
            f.write(json.dumps(value) + '\n')

    def platform():
        return subprocess.check_output(['xrt-smi', 'examine', '-r', 'platform'], text=True).strip()

    def npu_clients():
        probe = subprocess.run(['fuser', '/dev/accel/accel0'], capture_output=True, text=True)
        if probe.returncode not in (0, 1):
            raise RuntimeError('cannot inventory NPU clients: ' + probe.stderr)
        clients = []
        for pid in probe.stdout.split():
            if not pid.isdigit():
                continue
            try:
                name = (Path('/proc') / pid / 'comm').read_text().strip()
            except FileNotFoundError:
                name = '(exited)'
            clients.append(dict(pid=int(pid), name=name))
        return clients

    def trial(engine, prompt, forced, kind, length, round_index):
        before, load_before = platform(), Path('/proc/loadavg').read_text().strip()
        clients_before = npu_clients()
        reply = workers[engine].request(command='survey_trial', prompt=prompt, decode_tokens=forced)
        after = platform()
        clients_after = npu_clients()
        if (reply.get('survey_protocol') != 'phlegm-engine-survey-v1'
                or reply['prompt_tokens'] != len(prompt) or reply['decode_steps'] != len(forced)):
            raise RuntimeError('survey response counts/protocol disagree')
        result = dict(engine=engine, kind=kind, length=length, round=round_index,
                      platform_before=before, platform_after=after,
                      load_before=load_before, load_after=Path('/proc/loadavg').read_text().strip(),
                      clients_before=clients_before, clients_after=clients_after,
                      clients_known=all(c['pid'] in {w.process.pid for w in workers.values()}
                                        for c in clients_before + clients_after),
                      platform_stable=before == after, **reply)
        record(result)
        return result

    try:
        workers['artlite'] = Worker([str(args.artlite_python.absolute()),
            str(Path(__file__).with_name('artlite_worker.py').resolve())], art_env, args.out / 'artlite.stderr')
        art_init = workers['artlite'].request(command='init', model=str((args.model_dir / 'model.gguf').resolve()),
                                             max_ctx=args.max_ctx, threads=args.threads, kv_k='single')
        workers['phlegm'] = Worker([str(args.bridge.resolve())], env, args.out / 'phlegm.stderr')
        cpp_init = workers['phlegm'].request(command='init', model_dir=str(args.model_dir.resolve()),
            kernel_dir=str(args.kernels.resolve()), max_ctx=args.max_ctx, dispatch_timeout_ms=120000,
            prefill_mode='fast', prefill_route=args.prefill_route)
        meta = dict(protocol='phlegm-engine-survey-v1', development=True, created_unix=time.time(),
                    context_capacity=args.max_ctx, output_tokens=args.outputs, threads=args.threads,
                    artlite=art_init, phlegm=cpp_init,
                    model_sha256=fingerprint(args.model_dir / 'model.gguf'),
                    bridge_sha256=fingerprint(args.bridge), manifest_sha256=fingerprint(args.kernels / 'manifest.json'),
                    ids_sha256=fingerprint(args.ids_file), platform=platform(),
                    quantization_note='ArtLite reads source GGUF; Phlegm uses Q4_1 projections with BF16 compute scales and a Q8-repacked head; label head source quant separately')
        meta['runner_sha256'] = fingerprint(Path(__file__))
        meta['worker_pids'] = {name: w.process.pid for name, w in workers.items()}
        meta['artlite_worker_sha256'] = fingerprint(Path(__file__).with_name('artlite_worker.py'))
        meta['bridge_build'] = json.loads(args.bridge.with_suffix('.build.json').read_text())
        meta['kernel_toolchain'] = json.loads((args.kernels / 'toolchain.json').read_text())
        (args.out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        control_prompt = ids[:min(256, min(sizes))]
        control_tokens = workers['artlite'].request(command='trajectory', prompt=control_prompt,
                                                   output_tokens=16)['tokens'][:-1]
        for length in sizes:
            prompt = ids[:length]
            trajectory = workers['artlite'].request(command='trajectory', prompt=prompt,
                                                     output_tokens=args.outputs)['tokens']
            record(dict(kind='inputs', length=length, prompt=prompt, trajectory=trajectory))
            forced = trajectory[:-1]
            # Discard a warmup and the first paired unit; keep their raw evidence.
            for warmup in range(2):
                for engine in ('artlite', 'phlegm'):
                    trial(engine, prompt, forced, 'warmup', length, warmup)
            for round_index in range(args.rounds):
                order = ('artlite', 'phlegm') if round_index % 2 == 0 else ('phlegm', 'artlite')
                pre = {e: trial(e, control_prompt, control_tokens, 'control_before', length, round_index) for e in order}
                measured = {e: trial(e, prompt, forced, 'measured', length, round_index) for e in order}
                post = {e: trial(e, control_prompt, control_tokens, 'control_after', length, round_index) for e in order}
                drift = {e: {metric: abs(post[e]['sample'][metric] / pre[e]['sample'][metric] - 1)
                             for metric in ('prefill_ms', 'decode_ms')} for e in order}
                samples = [group[e] for group in (pre, measured, post) for e in order]
                valid = (all(max(drift[e].values()) <= .05 for e in order)
                         and all(r['platform_stable'] and r['clients_known'] for r in samples)
                         and len({r['platform_before'] for r in samples}) == 1)
                record(dict(kind='paired_result', length=length, round=round_index, valid=valid,
                    control_drift=drift, prefill_ratio=measured['artlite']['sample']['prefill_ms'] / measured['phlegm']['sample']['prefill_ms'],
                    decode_ratio=measured['artlite']['sample']['decode_ms'] / measured['phlegm']['sample']['decode_ms']))
                print(f'{length} round {round_index}: stable={valid}', flush=True)
        summary = {}
        for length in sizes:
            pairs = [r for r in records if r['kind'] == 'paired_result' and r['length'] == length and r['valid']]
            summary[length] = dict(valid_pairs=len(pairs), requested_pairs=args.rounds,
                median_prefill_ratio=statistics.median(r['prefill_ratio'] for r in pairs) if pairs else None,
                median_decode_ratio=statistics.median(r['decode_ratio'] for r in pairs) if pairs else None)
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    finally:
        for worker in reversed(list(workers.values())):
            worker.close()


if __name__ == '__main__':
    main()
