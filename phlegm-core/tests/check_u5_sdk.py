"""Hardware-free U5 SDK acceptance. Start `phlegm serve --mock --port 8767` first.

/tmp/phlegm-u5-sdk-venv/bin/python -I phlegm-core/tests/check_u5_sdk.py
The tool fixture is explicitly deterministic mock behavior, not model evidence.
"""
import argparse
import importlib.metadata
import json
from openai import BadRequestError, OpenAI

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:8767/v1')
parser.add_argument('--model', default='phlegm-mock')
args = parser.parse_args()
client = OpenAI(base_url=args.base_url, api_key='local-test', timeout=20)
messages = [{'role': 'user', 'content': 'hello'}]
checks = 0


def check(condition, label):
    global checks
    assert condition, label
    checks += 1
    print(f'PASS {label}', flush=True)


r = client.chat.completions.create(model=args.model, messages=messages, max_completion_tokens=8)
check(r.choices[0].message.content == '[mock np' and r.usage.completion_tokens == 8, 'chat + max_completion_tokens')
chunks = list(client.chat.completions.create(model=args.model, messages=messages, max_tokens=8,
                                           stream=True, stream_options={'include_usage': True}))
check(''.join(c.choices[0].delta.content or '' for c in chunks if c.choices) == '[mock np'
      and not chunks[-1].choices and chunks[-1].usage.completion_tokens == 8, 'streaming chat + usage')
for prompt in ('abc', ['abc'], [97, 98, 99], [[97, 98, 99]]):
    r = client.completions.create(model=args.model, prompt=prompt, max_tokens=8)
    check(r.choices[0].text == '[mock np' and r.usage.prompt_tokens == 3, f'raw completion prompt {prompt!r}')
chunks = list(client.completions.create(model=args.model, prompt='abc', max_tokens=8, stream=True,
                                       stream_options={'include_usage': True}))
check(''.join(c.choices[0].text for c in chunks if c.choices) == '[mock np'
      and not chunks[-1].choices and chunks[-1].usage.prompt_tokens == 3, 'streaming completion + usage')
for stream in (False, True):
    r = client.chat.completions.create(model=args.model, messages=messages, stop=' npu]', stream=stream)
    text = ''.join(c.choices[0].delta.content or '' for c in r if c.choices) if stream else r.choices[0].message.content
    check(text == '[mock', f'chat stop stream={stream}')
    r = client.completions.create(model=args.model, prompt='abc', stop=[' npu]', 'never'], stream=stream)
    text = ''.join(c.choices[0].text for c in r if c.choices) if stream else r.choices[0].text
    check(text == '[mock', f'completion stop stream={stream}')
    r = client.chat.completions.create(model=args.model, messages=messages, logprobs=True, top_logprobs=2,
                                       max_tokens=3, stream=stream)
    entries = [e for c in r if c.choices and c.choices[0].logprobs for e in c.choices[0].logprobs.content] if stream else r.choices[0].logprobs.content
    check(bytes(b for e in entries for b in e.bytes) == b'[mo'
          and all(len(e.top_logprobs) == 2 and e.logprob < 0 for e in entries), f'chat logprobs stream={stream}')
    r = client.completions.create(model=args.model, prompt='abc', logprobs=2, max_tokens=3, stream=stream)
    values = [c.choices[0].logprobs for c in r if c.choices and c.choices[0].logprobs] if stream else [r.choices[0].logprobs]
    check(''.join(t for v in values for t in v.tokens) == '[mo'
          and [o for v in values for o in v.text_offset] == [0, 1, 2], f'legacy logprobs stream={stream}')

tools = [{'type': 'function', 'function': {'name': 'get_status',
          'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
for stream in (False, True):
    turns = [{'role': 'user', 'content': 'Call get_status and report its result.'}]
    response = client.chat.completions.create(model=args.model, messages=turns, tools=tools,
                                               tool_choice='required', max_tokens=256, stream=stream)
    if stream:
        content, calls, finish, saw_content_first = '', [], None, False
        for chunk in response:
            choice = chunk.choices[0]
            content += choice.delta.content or ''
            if content and not calls:
                saw_content_first = True
            calls.extend(choice.delta.tool_calls or [])
            finish = choice.finish_reason or finish
        message = {'role': 'assistant', 'content': content,
                   'tool_calls': [{'id': c.id, 'type': c.type,
                                  'function': {'name': c.function.name, 'arguments': c.function.arguments}}
                                 for c in calls]}
        check(saw_content_first and content == 'Checking.' and finish == 'tool_calls', 'tools stream content before calls')
    else:
        message = response.choices[0].message.model_dump(exclude_none=True)
        check(response.choices[0].finish_reason == 'tool_calls', 'full tool call finish')
    call = message['tool_calls'][0]
    check(call['function']['name'] == 'get_status' and json.loads(call['function']['arguments']) == {}, f'tool call stream={stream}')
    turns.extend([message, {'role': 'tool', 'tool_call_id': call['id'], 'content': 'sunny'}])
    final = client.chat.completions.create(model=args.model, messages=turns, tools=tools, tool_choice='none')
    check('sunny' in final.choices[0].message.content and not final.choices[0].message.tool_calls, f'tool loop stream={stream}')
for options in ({'suffix': 'x'}, {'echo': True}, {'prompt': ['one', 'two']}):
    try:
        client.completions.create(**({'model': args.model, 'prompt': 'abc'} | options))
        raise AssertionError(f'expected 400: {options}')
    except BadRequestError:
        check(True, f'explicit unsupported option {options}')
print(f'{checks} checks passed; OpenAI Python SDK {importlib.metadata.version("openai")}; mock only')
