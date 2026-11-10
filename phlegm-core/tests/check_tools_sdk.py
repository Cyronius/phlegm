"""Hardware acceptance client. Requires the OpenAI Python SDK and a separately started server.

python -I phlegm-core/tests/check_tools_sdk.py --base-url http://127.0.0.1:8415/v1 --model MODEL
Run once for each of KAT Coder, Gemma 4 26B-A4B and GPT-OSS 20B.
"""
import argparse
import json
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()
client = OpenAI(base_url=args.base_url, api_key='local-test', timeout=300)
tools = [{'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Get the weather for a city.',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                   'required': ['city'], 'additionalProperties': False}}}]
for streaming in (False, True):
    messages = [{'role': 'user', 'content': 'Use get_weather for Paris, then tell me its result.'}]
    options = dict(model=args.model, messages=messages, tools=tools,
                   tool_choice='required', reasoning_effort='none', temperature=0, max_tokens=256)
    if not streaming:
        response = client.chat.completions.create(**options)
        assert response.choices[0].finish_reason == 'tool_calls', response
        message = response.choices[0].message.model_dump(exclude_none=True)
    else:
        calls, content, reasoning = {}, '', ''
        finish = None
        for chunk in client.chat.completions.create(**options, stream=True):
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish = choice.finish_reason or finish
            content += choice.delta.content or ''
            reasoning += getattr(choice.delta, 'reasoning_content', '') or ''
            for delta in choice.delta.tool_calls or []:
                call = calls.setdefault(delta.index, {'id': '', 'type': 'function',
                                                      'function': {'name': '', 'arguments': ''}})
                call['id'] += delta.id or ''
                if delta.function:
                    call['function']['name'] += delta.function.name or ''
                    call['function']['arguments'] += delta.function.arguments or ''
        assert finish == 'tool_calls', finish
        message = {'role': 'assistant', 'content': content or None,
                   'tool_calls': [calls[i] for i in sorted(calls)]}
        if reasoning:
            message['reasoning_content'] = reasoning
    assert message['tool_calls'], message
    messages.append(message)
    for call in message['tool_calls']:
        assert call['function']['name'] == 'get_weather', call
        assert json.loads(call['function']['arguments']) == {'city': 'Paris'}, call
        messages.append({'role': 'tool', 'tool_call_id': call['id'],
                         'content': '{"city":"Paris","weather":"sunny","temperature_c":20}'})
    final = client.chat.completions.create(model=args.model, messages=messages, tools=tools,
                                          tool_choice='none', reasoning_effort='none', temperature=0, max_tokens=256)
    assert final.choices[0].message.content, final
    assert not final.choices[0].message.tool_calls, final
    assert 'sunny' in final.choices[0].message.content.lower(), final
    print(args.model, 'stream' if streaming else 'full', 'PASS', final.choices[0].message.content)
