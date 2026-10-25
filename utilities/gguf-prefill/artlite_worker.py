"""Resident ArtLite worker for the development engine survey (no HTTP timing)."""
import json
import importlib.metadata
import inspect
from pathlib import Path
import subprocess
import sys
import time

from artlite.gguf_model import GGUFFile
from models.llama3.llama import NpuLlama

PROTOCOL = sys.stdout

def reply(value):
    print(json.dumps(dict(ok=True, **value)), file=PROTOCOL, flush=True)


def provenance():
    root = Path(inspect.getfile(NpuLlama)).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return dict(source_root=git('rev-parse', '--show-toplevel'), source_commit=git('rev-parse', 'HEAD'),
                source_status=git('status', '--porcelain'), python=sys.version,
                packages={name: importlib.metadata.version(name)
                          for name in ('mlir-aie', 'llvm-aie', 'torch', 'numpy', 'gguf')})


def main():
    sys.stdout = sys.stderr
    model = None
    reply({'survey_protocol': 'phlegm-engine-survey-v1', 'engine': 'artlite-native'})
    for line in sys.stdin:
        request = json.loads(line)
        command = request['command']
        if command == 'quit':
            reply({})
            return
        if command == 'init':
            model = NpuLlama(GGUFFile(str(Path(request['model']).resolve())),
                             capacity=request['max_ctx'], kv_k=request.get('kv_k', 'single'),
                             threads=request.get('threads', 4))
            reply({'capacity': model.capacity, 'description': model.describe(), 'provenance': provenance()})
            continue
        if model is None:
            raise ValueError('init required')
        ids = request['prompt']
        if not ids or len(ids) >= model.capacity:
            raise ValueError('invalid prompt length')
        model.prepare(len(ids))  # compilation is never a timed operation
        model.reset()
        model.stale()  # the other engine may have used another hardware context
        start = time.perf_counter_ns()
        logits = model.forward(ids)
        previous = int(logits.argmax())
        prefill_ms = (time.perf_counter_ns() - start) / 1e6
        if command == 'trajectory':
            tokens = [previous]
            for _ in range(request['output_tokens'] - 1):
                previous = int(model.step(previous).argmax())
                tokens.append(previous)
            reply({'tokens': tokens})
            continue
        if command != 'survey_trial':
            raise ValueError('unknown command')
        forced = request['decode_tokens']
        if len(ids) + len(forced) > model.capacity:
            raise ValueError('decode exceeds capacity')
        predicted, latencies = [previous], []
        start = time.perf_counter_ns()
        for token in forced:
            step_start = time.perf_counter_ns()
            previous = int(model.step(token).argmax())
            latencies.append((time.perf_counter_ns() - step_start) / 1e6)
            predicted.append(previous)
        decode_ms = (time.perf_counter_ns() - start) / 1e6
        reply({'survey_protocol': 'phlegm-engine-survey-v1', 'prompt_tokens': len(ids),
               'output_tokens': len(forced) + 1, 'decode_steps': len(forced),
               'greedy_tokens': predicted, 'sample': dict(prefill_ms=prefill_ms,
                   decode_ms=decode_ms, decode_token_ms=latencies)})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'ok': False, 'error': str(error)}), file=PROTOCOL, flush=True)
        raise
