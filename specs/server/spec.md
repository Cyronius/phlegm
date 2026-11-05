# phlegm-server spec

Canonical spec for phlegm-server: the API surface that lets served models (local or remote)
plug into a fixed set of API shapes. phlegm-tray is a GUI control surface for this server; it
has no independent spec requirements beyond calling this API.

Requirement prefix: `SERVER`.

## SERVER-VISION: Vision-understanding API

**Applies to:** phlegm-server
**Verification:** test

Models like Florence-2, Grounded SAM, SAM/SAM 2/SAM 3, YOLOE, YOLO-World, OWL-ViT, PaliGemma,
Qwen-VL, InternVL, GLM-OCR, docTR, TrOCR, Donut, and Moondream do image understanding, not image
generation or chat. No external de facto standard covers all of their output shapes together
(Hugging Face's Inference API task types come closest but do not unify them), so phlegm-server
defines its own.

One endpoint. A `task` field on the request selects which output shape the response uses. This
mirrors how Florence-2 itself works: one model, a task-prompt token picks the behavior.

### Request

```json
{
  "model": "florence-2-large",
  "task": "detection",
  "image": "<base64 or url>",
  "prompt": "find all red cars",
  "options": {}
}
```

- `model`: required. Selects the served model instance.
- `task`: required. One of `caption`, `detection`, `segmentation`, `ocr`, `grounding`, `vqa`.
- `prompt`: required for `grounding` and `vqa`. Required for `detection` only when the model is
  open-vocabulary (text-prompted). Ignored otherwise.
- `image`: required. A base64-encoded image or a fetchable URL.
- `options`: optional, model-specific (e.g. confidence threshold, detail level).

### Response shapes, by task

**`caption` and `vqa`** — plain text:

```json
{ "task": "caption", "text": "a red car parked on a street" }
```

**`detection` and `grounding`** — a box list, `box` always `[x1, y1, x2, y2]` in pixel
coordinates:

```json
{
  "task": "detection",
  "boxes": [
    { "label": "red car", "score": 0.94, "box": [120, 80, 340, 260] }
  ]
}
```

**`segmentation`** — a mask list. `mask` is either an RLE string (COCO-style) or a base64 PNG:

```json
{
  "task": "segmentation",
  "masks": [
    { "label": "red car", "score": 0.91, "mask": "<RLE or base64 PNG>" }
  ]
}
```

**`ocr`** — a text-region list, same box format as detection:

```json
{
  "task": "ocr",
  "regions": [
    { "text": "STOP", "score": 0.99, "box": [40, 40, 140, 70] }
  ]
}
```

### Acceptance criteria

- A request with `task: "caption"` returns a body with a `text` string field and no `boxes`,
  `masks`, or `regions` field.
- A request with `task: "detection"` returns a `boxes` array; each entry has `label` (string),
  `score` (number between 0 and 1), and `box` as a 4-number `[x1, y1, x2, y2]` array.
- A request with `task: "segmentation"` returns a `masks` array; each entry has `label`, `score`,
  and `mask` as a non-empty string (RLE or base64 PNG).
- A request with `task: "ocr"` returns a `regions` array; each entry has `text`, `score`, and
  `box` in the same 4-number format as `detection`.
- A request with `task: "grounding"` and no `prompt` is rejected with a client error, not a
  silent empty result.
- A request naming a `task` the loaded model does not support (e.g. `ocr` against SAM) is
  rejected with a clear error identifying the unsupported task, not a 500 or a malformed response.
- `box` coordinates are always `[x1, y1, x2, y2]` in pixel space across `detection`, `grounding`,
  and `ocr` — never normalized 0–1 coordinates, and never `[x, y, w, h]`.

**External tests:** none yet — phlegm-server does not exist as a buildable service. Tests land
in `specs/server/tests/` once the server's request/response layer is implemented.

## SERVER-REASONING: Reasoning apart from the answer

Chat templates render with the model's thinking switch on by default (`think` and
`reasoning_effort`, SERVER-SETTINGS and SERVER-REASONING-EFFORT), so a reasoning model writes its
thinking before the answer. `/v1/chat/completions` returns that thinking in `message.reasoning_content`
(streamed: `delta.reasoning_content`) and only the answer in `content`, the field llama.cpp's
and vLLM's OpenAI servers use. `phlegm run` prints the reasoning on stderr between
`[reasoning]` / `[/reasoning]` and the answer alone on stdout.

The markers come from the model's own files at load (`phlegm-core` `tokenizer.rs`,
`ReasoningFormat` / `Reasoning`), not from its name:

- `<|channel|>` in the chat template: GPT-OSS harmony channels (analysis, then final).
- `response_template.fields.thinking.{open, close}` in `tokenizer_config.json`: Gemma 4,
  `<|channel>thought\n` ... `<channel|>`.
- `<think>` and `</think>` in the chat template: Qwen3.5 (KAT, Ornith, Nex).

A rendered prompt that already ends in the opener (Qwen's generation prompt ends in `<think>\n`)
starts the reply inside the block. A reply with no opener is all answer; a reply cut off inside
its reasoning has an empty `content` and `finish_reason: "length"`.

### Acceptance criteria

- Neither field ever contains a marker. While streaming, each field only grows, and a partly
  decoded marker is held back until it resolves (`tagged_reasoning_streams_monotonically_and_splits`,
  `harmony_progress_grows_reasoning_then_answer`).
- A model whose files mark no reasoning block returns `content` only, unchanged.
- `phlegm run` defaults to the `max_tokens` setting, 0, as llama.cpp's `-n -1` does (`-1` is
  accepted for 0): until a stop token or the end of `--max-ctx` (a positive value is clamped to
  the room left). With 128, thinking models'
  answers were cut off behind the reasoning (KAT needed 256 for a one-line question).

## SERVER-REASONING-EFFORT: How much a model thinks

Requests and settings choose thinking in OpenAI's terms, and phlegm translates the choice into
whatever the model's chat template reads (`phlegm-core` `reasoning_effort.rs`):

- **Request:** `reasoning_effort` (OpenAI's Chat Completions field: `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`; anything else is HTTP 400), else the on/off switch
  `chat_template_kwargs.enable_thinking` (vLLM, llama.cpp) or `think` (OpenFlowLM), else the
  model's settings. Switching on keeps an effort the settings name.
- **Settings:** `reasoning_effort` (`--reasoning-effort`, registry `details.reasoning_effort`):
  `default` follows `think`; any OpenAI name sets the effort.
- **Template:** a template that reads `enable_thinking` gets it (false only for `none`). A
  template that reads `reasoning_effort` gets the value its registry entry maps the effort to
  (`details.reasoning_efforts`, OpenAI name -> template value; the nearest name, the higher of two
  equally near). Without a table an explicit effort passes through as written. "On" is `medium`.
- **Passthrough:** every other `chat_template_kwargs` key reaches the template as written (vLLM's
  behavior), after the translated ones, so a client can still send a template's own value. A
  template that fails to render (one that raises on a value it does not know) is HTTP 400, where
  it used to send the model an empty prompt.

Tables shipped (2026-11-04, from each template's source):

| model | template values | none | minimal | low | medium (on) | high | xhigh |
|---|---|---|---|---|---|---|---|
| Hy-Memory 30B | no_think / low / high; anything else is no_think | no_think | low | low | high | high | high |
| Nex-N2.5-mini | none / medium (also the default) / high | none | none | medium | medium | high | high |
| gpt-oss 20B | low / medium / high; cannot turn off | low | low | low | medium | high | high |

Qwen3.5 (KAT, Ornith), LFM3 and Gemma 4 read `enable_thinking` only, so every effort above `none`
is plain "on". Hy-Memory's entry sets `think: false`: its template's default is `no_think`, so
"on" by default would have changed what the model does unasked. Before this, `think` and
`enable_thinking` never reached Hy-Memory or Nex, and no request could set gpt-oss's effort.

### Acceptance criteria

- `reasoning_effort.rs`: `hy_memory_and_nex_get_their_own_values`,
  `a_model_that_cannot_stop_thinking_gets_its_lowest_effort`,
  `switch_templates_and_templates_without_a_table`, `requests_override_settings`,
  `bad_tables_are_errors`; `effort_templates_get_their_own_values` (`tokenizer.rs`) renders a
  reduced Hy-Memory template; `invalid_sampling_parameters_are_rejected_without_panicking` covers
  a bad `reasoning_effort` and a non-object `chat_template_kwargs`.
- Hardware (2026-11-04, `phlegm serve` with no flags, seed 1, "A farmer has 17 sheep. All but 9
  run away..."; `target/npu-dev/effort/check.py`, `check.json`):
  - Hy-Memory: default, `none` and `enable_thinking: false` answer at once ("8", 1 token, no
    reasoning); `low` thinks (123 tokens) and `high` / `think: true` think (85) and answer "9".
  - Nex Q4K: `none` and `enable_thinking: false` answer in 1 token; default / `low` / `think: true`
    (template value medium) close the block themselves (4 tokens); `high` (`<think>\n`) reasons
    for 49 tokens. All answer "9".
  - gpt-oss: `none` / `low` / `enable_thinking: false` reason for 32 tokens, default (medium) and
    `think: true` 373, `high` 143. All answer "9".

## SERVER-OUTPUT-BUDGET: Output tokens per request

`/v1/chat/completions` sets a request's output budget the way llama.cpp's server does
(`n_predict = -1` by default): generate until a stop token or the end of the context.

- **Without `max_tokens`:** the model's `max_tokens` setting (SERVER-SETTINGS; its env var is
  `PHLEGM_MAX_TOKENS`; default 0 = open-ended).
- **Cap:** `PHLEGM_MAX_TOKENS_CAP` (default 0 = no cap beyond the context).
- **Context:** the budget never runs past the backend's context (`Backend::context_capacity`:
  prompt + max_tokens - 1 <= `--max-ctx`). A request that reaches it finishes with
  `finish_reason: "length"` instead of failing, and an explicit `max_tokens` larger than the room
  left is clamped rather than rejected.
- **Prompt too long:** a prompt longer than the context is HTTP 400.
- **No fixed context:** a backend without one (the mock) uses a 4096-token fallback bound.

The old defaults (64, capped at 512) cut thinking models off inside their reasoning.

### Acceptance criteria

- `output_budget_follows_llama_cpp_server` (`phlegm-core` `server.rs`) covers the open-ended
  default, clamping to the context, the exact-fit prompt, the too-long prompt and the env overrides.

## SERVER-PROMPT-REUSE: Earlier turns are not prefilled again

The NPU bridge keeps its context between requests, and a request prefills only what its prompt adds
to it (llama.cpp's server does the same with its slot cache; `phlegm-core` `bench/reference.rs`
`reuse_start`, `benchmarks/reference/bridge.cpp`).

- **Attention rows** sit at absolute positions in the cache (sliding windows included), so going back
  to any earlier position only moves the position (`rewind`); rows past it are rewritten later.
- **Recurrent layers** (DeltaNet, LFM's short conv) cannot be run backwards. A prefill saves their
  state at a **mark** (`prefill_logits` with `mark`): the server asks for it where the conversation's
  messages end and the generation prompt begins (`GenParams::reuse_mark`, from the template rendered
  without the generation prompt), since templates rewrite the last turn there (Qwen's and LFM's drop
  the reasoning of earlier replies). On the block routes the mark lands on the last block boundary at
  or before it, so the split adds no block and the blocks are the ones a single prefill would run;
  the mark keeps no KV rows.
- **A request** starts at the longest of: the end of the bridge's context when the prompt extends it
  (the earlier prompt plus every reply token fed back); the mark, when the prompts agree up to it; on
  a model without recurrent layers, wherever the prompts part. At least one token is always
  prefilled, for the logits. Otherwise it resets.
- **Usage:** `usage.prompt_tokens_details.cached_tokens` (OpenAI's field) reports the reused tokens;
  `phlegm` logs `prompt N tokens, R reused` per request. `PHLEGM_PROMPT_REUSE=0` turns reuse off.
- A failed request leaves nothing to reuse (the bridge exits on any error).

### Acceptance criteria

- `reuse_start_takes_the_longest_valid_start` and `resident_bridge_reuses_the_shared_prompt` (a fake
  bridge that keeps a position and a mark) in `bench/reference.rs`.
- Hardware, per model (`target/npu-dev/reuse/check.py`): the reuse paths' logits and 8 greedy decode
  steps are bit-identical to the same prefills from a reset (rewind to the mark; appending after fed
  decode tokens; on attention-only models a rewind to an unaligned position).

Results (2026-11-05, development, host load 3-10, 16 threads, default bridges; the first turn is
1506 tokens with the mark asked at 1500, the second 1800 tokens sharing the first 1500, the third
appends 48 tokens after 8 fed decode tokens; `reuse/*.json`):

| model (route, recurrent layers) | mark | rewind / append / seek bit-identical | 2nd turn, reset -> reuse | 3rd turn, reset -> append |
|---|---|---|---|---|
| LFM3 (block, 13) | 1280 | yes / yes / - | 6.4 -> 2.4 s | 6.4 -> 0.8 s |
| LFM2.5 (block, 30) | 1280 | yes / yes / - | 16.3 -> 5.5 s | 16.8 -> 1.3 s |
| Flint (block, 0) | 1280 | yes / yes / yes | 22.8 -> 7.6 s | 23.6 -> 1.5 s |
| Hy-Memory (block, 0) | 1280 | yes / yes / yes | 25.8 -> 8.7 s | 26.7 -> 1.6 s |
| KAT Coder Q4pure (layer-major, 30) | 1280 | yes / yes / - | 8.0 -> 4.1 s | 8.3 -> 1.3 s |
| gpt-oss (its own prefill, 0) | 1500 | yes / yes / yes | 6.9 -> 2.3 s | 7.2 -> 1.1 s |
| Gemma 4 (layer-major, 0) | 1280 | yes / yes / yes | 24.2 -> 9.3 s | 24.9 -> 2.8 s |

Against a reset and one prefill of the whole prompt (what phlegm did before), the reuse paths are
the route's own arithmetic in other pieces, not a new approximation:
- **Block routes:** the mark is block-aligned, so the rewind path runs the very blocks a single
  prefill runs: bit-identical (LFM3, LFM2.5, Flint, Hy-Memory, Gemma 4, KAT on `--prefill-route block`).
- **Layer-major routes** (KAT, gpt-oss): the MoE runs once over each call's tokens, so a prefill in two
  calls differs from one call at rounding level (max |logit difference| 0.91 KAT, 0.29 gpt-oss; argmax
  and 8 greedy tokens equal).
- **Appending** reads the reply's KV rows as the NPU decode kernels wrote them, where a single
  prefill recomputes them on the prefill route: max |logit difference| 0.04-0.97, argmax equal; the
  8 greedy tokens after it equal on 6 of 8 models (LFM2.5 and Gemma 4 part at a later step).
- **An unaligned rewind** on an attention-only model keeps rows that came from blocks composed
  differently from a single prefill's (block-composition rounding, already open on these routes).

Through `phlegm serve` (LFM3, `reuse/turns.py`, a ~2950-token log question then two short
follow-ups, greedy, 400 reply tokens): turns 2 and 3 reuse 2816 tokens and take 21 s instead of
29-30 s (two alternating rounds; the replies' reasoning has the same length with and without reuse,
1117 / 1103 / 1188 characters, the text itself was not recorded). KAT Coder: 3328 and 3584 of
3600 / 3624 tokens reused (its no-reuse round was cut short to free the machine).

## SERVER-SETTINGS: Per-model runtime defaults

`phlegm run` and `phlegm serve` take their sampling, template, context and prefill defaults from
the model's settings, in OpenFlowLM's two-file format (openflowlm-next #214, its
`docs/config.md`; `phlegm-core` `settings.rs`):

- **Definitions:** `phlegm-core/config/settings.json` (OpenFlowLM's `src/config/settings.json`
  shape: key, cli_flag, env_var, type, options / min / max, scope, delivery, category,
  reload_required, registry_key, default_value). Phlegm adds the type `float`, `min_exclusive`,
  and `generation_config` (the `generation_config.json` field a model directory supplies).
- **Shipped registry:** `phlegm-core/config/model_list.json`, built into the binary
  (`PHLEGM_CONFIG_PATH` reads another): `version`, `defaults_version` (the first 16 hex digits
  of sha256 of the sorted, compact JSON of `defaults`), `defaults`, and
  `models.<family>.<size>` entries (`name` = the model directory's name, `details`,
  `default_context_length`, `sampling`). A `--model-dir` finds its entry by that name,
  case-insensitively; `--config-model family:size` names one directly.
- **User overlay:** `PHLEGM_SETTINGS_PATH`, else `$XDG_CONFIG_HOME/phlegm/settings.json`, else
  `~/.config/phlegm/settings.json` (`%USERPROFILE%\.phlegm\settings.json` on Windows). Sparse:
  `defaults` (every model), `settings.<family>.<size>` (one model), `models` (entries added or
  changed, deep-merged into the registry), `version`, `defaults_version`. Phlegm never writes it
  except through `phlegm config set` / `unset`. A bad value or an unknown key in it is reported
  and ignored, not fatal; a changed `defaults_version` is reported with the overridden keys.

Settings: `prefill_mode`, `prefill_route`, `ctx_len` (`--max-ctx`), `think`, `reasoning_effort`, `temperature`,
`top_k`, `top_p`, `repetition_penalty`, `repetition_penalty_scope`, `presence_penalty`,
`frequency_penalty`, `length_bias` (+ `_start`, `_window`), `think_end_bias` (+ `_start`,
`_window`), `max_tokens` (`PHLEGM_MAX_TOKENS`). Precedence, lowest
to highest: the built-in default; the model directory's `generation_config.json` (HF semantics:
without `do_sample: true` it is greedy and only `repetition_penalty` applies; with it a field
left out takes HF's default, temperature 1, top_k 50, top_p 1); the registry entry; the shipped
`defaults`; the user's `defaults`; the user's per-model settings; the env var; the flag; and in
the server the request's own fields (`temperature`, `top_p`, `top_k`, `repetition_penalty` or
llama.cpp's `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `reasoning_effort`,
`chat_template_kwargs` or `think`). This is
OpenFlowLM's order with `generation_config.json` added under the registry, so a model directory
outside the registry still samples the way its maker set it.

**Penalties on the reply** (2026-11-05, `sampler.rs` `Penalties`, Clover's `sampling_clover_moe.py`):
`presence_penalty` and `frequency_penalty` count the reply's tokens only, as OpenAI and vLLM define
them (in [-2, 2]; a request may set them). `repetition_penalty_scope` picks what the repetition
penalty counts: `context` (prompt and reply, as HF, llama.cpp and vLLM do; the default) or
`response` (Clover). `length_bias` is added to the stop tokens' logits and `think_end_bias` to the
reasoning block's closing token until the reply closes it, each 0 through `_start` reply tokens and
rising linearly to full over `_window` more. Order: frequency, presence, repetition, the ramps
(Clover's), then temperature / top-k / top-p. They apply under greedy decoding too, and
`generation_config.json`'s `presence_penalty`, `frequency_penalty` and `dllm_*` fields count
whatever `do_sample` says. Tests: `presence_and_frequency_count_the_reply_only`,
`repetition_can_count_the_reply_only`, `ramps_grow_with_the_reply_and_stop_once_seen`.

The shipped entries copy each model's `generation_config.json` (2026-11-04): LFM3 temperature
0.1, top_k 50, repetition_penalty 1.1; LFM2.5 0.1 / 50 / 1.05; Gemma 4 1.0 / 64 / top_p 0.95;
Qwen3.5 MoE (KAT, Ornith) and Flint 1.0 / 20 / 0.95; gpt-oss 1.0 / 50. Before this every request
without fields was greedy with no repetition penalty, whatever the model asked for. Two models
ship no generation config, so their entries take what the publisher says instead (no LM Studio
catalogue entry exists for either, checked 2026-11-04):
- **Nex-N2.5-mini:** temperature 0.7, top_p 0.95, top_k 40, the model card's "Our evaluations
  use" values (nex-agi/Nex-N2.5-mini @ 87420286).
- **Hy-Memory-30B-A3B:** temperature 0 (greedy). The card's only setting is its quickstart's
  `temperature=0.0` (tencent/Hy-Memory-30B-A3B @ 64a460a9); neither the card, the GitHub repo nor
  the report (arXiv 2610.22196) gives top-k, top-p or a sampled temperature.

Both templates switch reasoning with `reasoning_effort`, not `enable_thinking`; their registry
entries map OpenAI's effort names to their values (SERVER-REASONING-EFFORT).

**Seed:** a request (or `phlegm run`) without `seed` samples from a fresh random seed, as
OpenAI's and llama.cpp's servers do; `seed` reproduces a reply. Until 2026-11-04 the missing seed
was a fixed constant, so identical requests returned identical replies.

**Context:** `ctx_len` defaults to 32768 (OpenFlowLM's usual `default_context_length`), and every
registry entry states it; it was 4096, the kernel manifests' `max_ctx_default`. Capacity is a
runtime allocation (no descriptor compiles it in: `specs/open-engine/spec.md`, `kvslice`): at most
`kv_row` x attention layers a token from the manifests, 20-225 KB (at 32768: 0.6-0.7 GiB for the
Qwen3.5 MoE and LFM models, 1.5 GiB gpt-oss, 3 GiB Flint and Hy-Memory, 6.9 GiB Gemma 4; an upper
bound, counting sliding-window layers as full). Measured 2026-11-04 (`target/npu-dev/fastdefault/ctxcost.py`,
`ctxcost.json`; `phlegm serve --max-ctx 4096` against `32768`, one load each, greedy, thinking
off; development host, other sessions' jobs between models), 4096 -> 32768:

| model | prefill ~2.9-3.6k tokens | short-prompt first token | decode ms/token | needle at ~4.6-5.7k tokens |
|---|---|---|---|---|
| LFM3 1.7B | 33.0 -> 32.0 s | 0.90 -> 0.97 s | 58 -> 55 | found (4642 tokens, 68 s) |
| LFM2.5 24B | 46.6 -> 36.4 s | 1.06 -> 0.99 s | 144 -> 142 | found (4641, 164 s; 414 tokens, reasons first) |
| Flint 40B | 620 -> 614 s | 2.0 -> 1.9 s | 204 -> 196 | found (4552, 1009 s) |
| Gemma 4 26B | 54.1 -> 53.9 s | 2.07 -> 2.01 s | 165 -> 192 (4 steps) | found (5435, 88 s) |
| gpt-oss 20B | 18.4 -> 19.5 s | 1.15 -> 1.11 s | 152 -> 159 | found (4620, 45 s) |
| Hy-Memory 30B | 666 -> 657 s | 4.3 -> 3.8 s | 204 -> 233 (4 steps) | found (4621, 1211 s) |
| KAT Q4K 35B | 31.1 -> 31.0 s | 1.58 -> 1.43 s | 116 -> 140 (4 steps) | found (5698, 53 s) |

The larger capacity costs nothing measurable (the "4 steps" decode figures come from a 5-token
reply and are noise), and positions past 4096 answer correctly. What does cost is long-prompt
prefill, at either capacity: Flint and Hy-Memory prefill at ~4.5 tokens/s, and LFM3 falls from
~230 tokens/s at 843 tokens to 89 at 2950 and 24 at 11688 (488 s, still correct). A first probe at
11.7k tokens ran past the server's 600 s request timeout (HTTP 500) on LFM2.5 and Flint.

Later the same day the LFM routes' host attention became a vectorised, tiled version of the same
arithmetic (bit-identical logits; `open_kernels/upstream-patches/block-host-attention-runtime.patch`,
`.plans/long-prefill.md`) and phlegm's bridge got 16 host threads instead of 4 (`--threads`
defaults to half the logical CPUs, 4 to 16). The same probes through `phlegm serve` at 32768
(`ctxcost-fa.json`, `ctx32k-fa.json`; host shared with two other sessions' builds and NPU jobs):
LFM3 2950 tokens 32.0 -> 9.6 s, needle at 4642 68 -> 16.5 s, 11688 tokens 488 -> 48 s (correct);
LFM2.5 2949 tokens 36.4 -> 22.7 s, 11687 tokens now answers correctly in 231 s (477 reply tokens)
where it timed out. Flint and Hy-Memory then got block prefill routes (OPEN-FAMILY-FLINT,
OPEN-FAMILY-HYV3; bridges `flint-blk`, `hyv3-blk`) with the same attention (`ctxcost-blk.json`,
`ctx32k-blk.json`): Flint 2891 tokens 614 -> 41.6 s, needle at 4552 1009 -> 77.8 s, 11482 tokens
correct in 236 s (was HTTP 500); Hy-Memory 2938 tokens 657 -> 50.1 s, needle at 4621 1211 -> 92.6 s,
11633 tokens correct in 204 s.

`phlegm config show [--model-dir DIR | --config-model TAG] [--json]` prints the paths and every
value with its source (shipped, generation_config, user, env); `set` / `unset` write the user
file atomically; `list` prints the registry. (OpenFlowLM's guided flow, full-screen browser,
`review` and legacy migration are not ported.)

Sampling cost: with top-k, top-p, the softmax and the draw run over the k candidates, not the
vocabulary (262k vocabulary, top-k 64, top-p 0.95: 1.1 ms a token instead of 4.2; release build,
loaded host, 2026-11-04). Top-p without top-k still sorts the vocabulary (~18 ms a token there);
no shipped model uses it.

### Acceptance criteria

- `the_shipped_files_are_consistent` (`settings.rs`): every built-in default and registry value
  passes its setting's checks, flags and names are unique, `defaults_version` is current.
- `precedence_runs_from_built_in_to_the_command_line`,
  `a_model_outside_the_registry_takes_its_generation_config_with_hf_semantics`,
  `the_user_file_overlays_models_and_bad_entries_are_reported_not_fatal`.
- `requests_take_the_model_defaults_and_override_them` and
  `invalid_sampling_parameters_are_rejected_without_panicking` (`server.rs`).
- `the_candidate_list_draws_what_the_full_vector_draws` (`sampler.rs`): the top-k candidate
  sampler draws exactly the tokens the full-vector one does for the same seed;
  `no_seed_draws_differently_each_time`, `seed_reproducible`.
- Context: the table above; every registry entry carries `default_context_length` 32768.
- Hardware (2026-11-04, LFM3, `phlegm serve` with no flags): "Name three primary colors" took
  268 / 210 / 199 tokens (seeds 1-3) at the model's settings against 1254 greedy without the
  penalty (that request's override reproduces the earlier greedy run token count exactly);
  `enable_thinking: false` answered in 5 tokens.
