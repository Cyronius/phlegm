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

Chat templates render with `enable_thinking=true`, so a reasoning model writes its thinking
before the answer. `/v1/chat/completions` returns that thinking in `message.reasoning_content`
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
- `phlegm run` defaults to `--max-tokens -1`, as llama.cpp does: until a stop token or the end
  of `--max-ctx` (a positive value is clamped to the room left). With 128, thinking models'
  answers were cut off behind the reasoning (KAT needed 256 for a one-line question).

## SERVER-OUTPUT-BUDGET: Output tokens per request

`/v1/chat/completions` sets a request's output budget the way llama.cpp's server does
(`n_predict = -1` by default): generate until a stop token or the end of the context.

- **Without `max_tokens`:** `PHLEGM_MAX_TOKENS` (default 0 = open-ended).
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
