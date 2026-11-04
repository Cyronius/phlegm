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

Chat templates render with `enable_thinking` from the model's `think` setting (default true;
SERVER-SETTINGS), so a reasoning model writes its thinking before the answer. `/v1/chat/completions` returns that thinking in `message.reasoning_content`
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

Settings: `prefill_mode`, `prefill_route`, `ctx_len` (`--max-ctx`), `think`, `temperature`,
`top_k`, `top_p`, `repetition_penalty`, `max_tokens` (`PHLEGM_MAX_TOKENS`). Precedence, lowest
to highest: the built-in default; the model directory's `generation_config.json` (HF semantics:
without `do_sample: true` it is greedy and only `repetition_penalty` applies; with it a field
left out takes HF's default, temperature 1, top_k 50, top_p 1); the registry entry; the shipped
`defaults`; the user's `defaults`; the user's per-model settings; the env var; the flag; and in
the server the request's own fields (`temperature`, `top_p`, `top_k`, `repetition_penalty` or
llama.cpp's `repeat_penalty`, `seed`, `chat_template_kwargs.enable_thinking` or `think`). This is
OpenFlowLM's order with `generation_config.json` added under the registry, so a model directory
outside the registry still samples the way its maker set it.

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

Both templates switch reasoning with `chat_template_kwargs.reasoning_effort`, not
`enable_thinking`, so `think` does not reach them yet.

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
