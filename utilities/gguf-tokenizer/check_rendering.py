#!/usr/bin/env python3
"""Compare the three Rust-rendered chats to independent sandboxed Python Jinja2."""
import argparse
import json
import pathlib
import runpy
from jinja2.sandbox import ImmutableSandboxedEnvironment


def run(args):
    metadata = runpy.run_path(str(pathlib.Path(__file__).with_name('check.py')))['metadata']
    corpus = json.loads(args.corpus.read_text())
    count = 0
    for evidence in sorted(args.evidence.glob('*.json')):
        if evidence.stem in ('summary', 'expected', 'provenance', 'rendering'):
            continue
        result = json.loads(evidence.read_text())
        m = metadata(args.headers / evidence.stem / 'header.gguf')
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])

        def fail(message):
            raise ValueError(message)

        env.globals['raise_exception'] = fail
        for name in ('bos', 'eos'):
            key = f'tokenizer.ggml.{name}_token_id'
            if key in m:
                env.globals[name + '_token'] = m['tokenizer.ggml.tokens'][m[key]]
        template = env.from_string(m['tokenizer.chat_template'])
        for chat in corpus['conversations']:
            rendered = template.render(messages=chat['messages'], tools=None, add_generation_prompt=True,
                                       **corpus['template_kwargs'])
            actual = next(c['text'] for c in result['cases'] if c['name'] == chat['name'])
            assert actual == rendered, f"{evidence.stem}/{chat['name']}: rendered bytes differ"
            count += 1
        print(evidence.stem, '3/3 rendered chats equal Jinja2', flush=True)
    assert count == 27, count
    print(f'PASS: {count}/27 rendered conversations')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headers', type=pathlib.Path, required=True)
    p.add_argument('--evidence', type=pathlib.Path, required=True)
    p.add_argument('--corpus', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2] / 'phlegm-core/tests/fixtures/gguf-tokenizer/corpus.json')
    run(p.parse_args())
