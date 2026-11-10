# phlegm-server spec

Canonical spec for phlegm-server: the API surface that lets served models (local or remote)
plug into a fixed set of API shapes. phlegm-tray is a GUI control surface for this server; it
has no independent spec requirements beyond calling this API.

Requirement prefix: `SERVER`.

## SERVER-CHAT-PAGE: Built-in chat

GET `/` serves a self-contained offline chat page. It lists models, streams chat and
reasoning, shows tool calls, and keeps conversations in browser localStorage; the API
key is entered in the page and is not stored. The page is public so it can prompt for
the key; API endpoints retain their configured authentication.

## SERVER-HTTP: Browser access, authentication and chat usage

The chat server accepts browser preflight `OPTIONS` requests and sends CORS
headers with JSON and event-stream responses. `PHLEGM_ALLOWED_ORIGINS` or
`--allowed-origins` defaults to `loopback`: HTTP or HTTPS origins on `localhost`,
`127.0.0.1`, or `[::1]`, with any port. `*` and a comma-separated list of exact
origins are opt-in. CORS is not authentication.
`PHLEGM_API_KEY` or `--api-key` enables `Authorization: Bearer <key>` on all
non-preflight routes except GET `/health` and GET `/`; it is disabled by default. An invalid or missing key
returns 401. The command-line value takes precedence over the environment.

Chat requests accept `max_completion_tokens` in preference to `max_tokens`.
With `stream_options.include_usage: true`, the final event before `[DONE]`
has an empty `choices` array and the prompt, completion, total and cached
token counts. These HTTP behaviors are covered by the server's mock tests;
they do not establish hardware inference correctness.

## SERVER-GENERATION: Stop, legacy completions and log probabilities (U5)

Both `/v1/chat/completions` and `/v1/completions` accept `stop`: one nonempty
string or an array of up to four nonempty strings (null/omitted/empty array disable
it). Invalid shapes and more than four entries are HTTP 400. Matching runs on the
decoded generated text, before reasoning splitting and tool parsing, including
matches inside a token and across token/UTF-8 boundaries. The earliest match ends
generation, removes the stop string and everything after it, and sets
`finish_reason: "stop"`, even at the output budget. Streaming holds the longest
suffix that is a prefix of any stop string, and flushes it when generation ends
without a match. Thus a stop can truncate reasoning or a call; a truncated call
retains the existing plain-text fallback. Usage counts actual generated tokens,
including those withheld to recognize a stop; it does not retokenize visible text.

`POST /v1/completions` takes a raw `prompt`, without applying a chat template:
a string, token IDs, a single-element string array or a single-element array of
token-ID arrays. Empty inputs, invalid/out-of-vocabulary IDs, and multiple prompts
return 400 (batching is out of scope). It accepts `max_tokens`, the same sampling
fields/penalties/seed as chat, `stop`, `stream`, and `stream_options.include_usage`.
Responses use `object: "text_completion"` and `choices[].text`, including SSE
frames, followed by `[DONE]`. Only `n=1` and `best_of=1` are supported. `echo=true`
and non-null `suffix` return explicit 400 errors; omit them or use `echo=false`.
Invalid token budgets, stream options and logprob options also return 400.

Chat accepts boolean `logprobs`, with `top_logprobs` an integer from 0 to 20 that
requires `logprobs=true`. `choices[].logprobs.content` entries contain `token`,
`bytes`, `logprob`, and `top_logprobs` (each candidate also has decoded bytes).
Legacy completions accept integer `logprobs` from 0 to 5 and return the legacy
`tokens`, `token_logprobs`, `top_logprobs` maps and `text_offset` arrays. The chosen
token is always included in a legacy top map, even if outside the requested top
set. As a Phlegm extension, legacy logprobs also include parallel `bytes` arrays.
Offsets count Unicode characters in the decoded output; tokens splitting one
UTF-8 character share its offset. Standalone token strings can contain replacement
characters; their exact bytes, rather than those strings, reconstruct split UTF-8.
Special control tokens have null bytes. Output shapes and limits follow the official
[Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [legacy Completions reference](https://developers.openai.com/api/reference/resources/completions/methods/create).

Distribution choice: normalize the backend's raw model logits at temperature 1,
before temperature scaling and top-k/top-p filtering. Phlegm also excludes all
presence/frequency/repetition penalties, length/think-end ramps and the U4 mask
from *reported* probabilities. These still affect token selection, so a constrained
or penalized chosen token can be absent from `top_logprobs`. This is an explicit
Phlegm choice for penalties and masking, not a claim about undocumented OpenAI
implementation details. Logprobs describe verified generated tokens before the
reasoning/tool split (including their protocol tokens), rather than rescoring
rendered content. Stop-matched tokens and tokens overlapping the removed suffix
are omitted from logprobs; a token cut in its middle therefore has no score for its
visible fragment. SSE may deliver logprobs in separate empty-content chunks;
probabilities held for a possible stop or call marker are released only when safe.
No log-softmax or logprob vocabulary is constructed when logprobs are not requested.

`Backend::generate` passes `TokenEvent { id, logits }` to a callback returning
`ControlFlow<()>`. The callback receives only verified tokens (also the contract
for future P4 speculation). Raw logits are borrowed during the callback, and a
backend without logits reports an explicit generation error when they are requested.
On `Break`, the backend issues no further decode; kept context and reuse marks
continue to describe what the bridge actually ran, excluding the last output token
unless it was already fed back. The fake-bridge regression checks early cancellation
and a follow-up that reuses the exact prefix, with the bridge command log asserted.

Offline verification, 2026-11-09: new HTTP tests in `server_generation_tests.rs`
cover both routes, stop validation/overlap/partial suffixes, stops inside tokens,
UTF-8 bytes and offsets, usage, error shapes, raw probabilities under penalties
and a schema mask, reasoning, tool truncation, and all six CPU tool fixtures.
A socket handshake test requires content to reach the client while the backend
is blocked waiting to continue; it proves tool-enabled content is streamed during
generation. No model timing claim is made. The ordinary Rust suite and the SDK
smoke test use no NPU and add no crate dependencies.

Final Rust checks: 121 core unit tests passed, five ignored; two enabled core
integration tests passed, two integration prerequisites ignored; all four CLI tests
passed. `cargo build -p phlegm-cli --offline` and `git diff --check` passed.
Commands, raw-log paths/hashes and corrected development failures:
[u5-validation.json](u5-validation.json).

Mock SDK acceptance (OpenAI Python 3.48.0, 24 checks, repeated against the final binary):

```sh
dibs run build --label 'U5 follow-up: build mock SDK server' --eta 5m -- cargo build -p phlegm-cli --offline
dibs run port-8767 --label 'U5 follow-up: mock SDK server' --eta 10m -- target/debug/phlegm serve --mock --port 8767
# In another shell while the server holds the port:
/tmp/phlegm-u5-sdk-venv/bin/python -I phlegm-core/tests/check_u5_sdk.py
```

The mock has a deterministic no-argument tool fixture when required/named tools
are supplied, and echoes tool results on the follow-up. Its logits are synthetic;
this SDK pass establishes client/API compatibility, not model-quality evidence.

Hardware loops remain untested (this task explicitly prohibited NPU execution).
For each of `KAT-Coder-V2.5-Dev-Q4pure-NPU2`, `gemma-4-26B-A4B-it-NPU2` and
`gpt-oss-20b-NPU2`, run later, replacing `MODEL` with that directory basename:

```sh
dibs run npu port-8767 --label 'U5: hardware SDK server' --eta 20m -- /home/cyrus/code/phlegm/target/npu-dev/npu-run.sh target/debug/phlegm serve --model-dir /home/cyrus/models/MODEL --port 8767
# In another shell while that server holds the port:
/tmp/phlegm-u5-sdk-venv/bin/python -I phlegm-core/tests/check_tools_sdk.py --base-url http://127.0.0.1:8767/v1 --model MODEL
```

The mock-specific `check_u5_sdk.py` intentionally asserts mock text and is not a
hardware quality gate. Hardware stop/logprob/completion checks should use the same
API calls with model-independent assertions, followed by the tool loop above.

## SERVER-MODEL-SWITCHING: One server for chat, images and embeddings (U7)

`phlegm serve` with no single-model selector starts a lazy model server. It exposes
chat, legacy completions, embeddings, `/v1/images/generations` and
`/v1/images/edits` on the same address with the same authentication/CORS policy.
No backend, worker or NPU lock is acquired during discovery or GET requests.
`serve --model-dir DIR`, `serve --embed-model DIR` and `serve --mock` retain their
single-model modes. `PHLEGM_MODEL_DIR` selects the single-model mode unless an
explicit `--models-dir` or `--image-config` selects discovery instead.

**Discovery and model names.** `--models-dir DIR` overrides `PHLEGM_MODELS_DIR`,
otherwise `$HOME/models` (`$USERPROFILE/models` on Windows). The registry contains
names and settings, not installation paths; it does not supply another search
root today. Discovery scans that root and its immediate child directories once
at startup, including directory symlinks. Restart to discover additions/removals.
Registry entries without local weights are excluded. Chat IDs are directory
basenames: a GGUF of any filename qualifies, as does `model.q4nx` beside a
tokenizer or matching a registry name. Bare GGUF tokenizers follow P5 S6;
discovery does not certify kernels, bridge availability or model correctness.
`config.json` with `architecture: "TritEmbed"` and `model.safetensors` identifies
an embedding directory; its basename is its ID. Per-model chat settings and bridge
selection are resolved again on each load using SERVER-SETTINGS and P1 S2.
`--kernels` and `--config-model` require single-model mode; `--bridge` remains an
explicit override for deployments with one compatible bridge.

Image installations need their existing native pipeline configuration because
weights alone do not describe the text encoder, kernels and VAE. Discovery reads
`image_config.json` or `phlegm-*-image.json` in the same directories; external
configurations can be added with `--image-config JSON[,JSON...]`. Images use the
pipeline's existing public model ID. Duplicate IDs at different paths are a startup
error rather than an arbitrary selection; choose one configuration for variants
sharing a public ID. The same canonical config path can be named more than once.
Config assets are validated when that image model is selected. Existing image
installations outside `~/models` therefore need `--image-config` or a config in
the discovery root. Do not copy model weights to expose an existing installation.

`GET /v1/models` lists all catalog entries, sorted by ID, including unloaded
models, with the OpenAI fields and the extensions `capabilities: ["chat"]`,
`["embedding"]` or `["image"]`, `loaded: bool`, and
`status: "unloaded" | "loading" | "loaded"`. `created: 0` means the installation
time is unknown. A failed load leaves its entry installed and unloaded; it is
retryable. For images, loaded means the pipeline config is selected; the existing
sampler creates and releases device resources for each generation/edit.

**Request lifetime.** The request's `model` selects an installed ID. It is required
in discovery mode, except that a catalog containing just one image model accepts
an omitted model for compatibility with `serve-images`. One FIFO lease serializes
all inference, including same-model requests. A different model waits for the
current response to finish, including the final SSE frames, then destroys the old
backend before invoking the new loader. The chat bridge and embedding worker are
killed and reaped before their NPU lock is released. No extra Arc to the old model
is retained by the catalog. A load error returns 503 with its diagnostic, leaves
no resident model, and does not prevent a later load. A stream already running is
never canceled by another model request. Client disconnects follow the existing
generation behavior; the lease remains held until its handler actually returns.

`--queue-timeout SECONDS` defaults to 600, measured from entry to the model queue
after reading the bounded request body. Timed-out waiters leave the queue. It
does not cancel another request's load or generation, nor bound the selected
loader's own startup/device-lock timeouts. There are at most 64 concurrent POST
handlers (including body readers); excess requests return 503. GET discovery and
health remain available while a worker loads or generates. Queue order is the
order handlers enter the lease queue, not TCP connection order.

`--keep-alive SECONDS` optionally unloads a resident model after that much idle
time since its last handler completed; omitted means keep it until a switch or
normal server teardown. A 250 ms sweep checks idle time. Zero releases it on the
next sweep unless a request is active/queued. Single-model modes retain their
existing lifetime; they reject these discovery-only options. Image samplers use
the same `NpuLock` around generation as the resident backends, wait for their child
to exit before replying, and set the existing parent-death behavior on the child.
Images retain their existing size, edit-capability and response-format validation;
JSON/base64 and multipart edits reuse the same parser as the image-only server.

`serve-images --config JSON` is an alias using the same model server with one
image entry and its existing default port 52626. It keeps optional `model`, both
image endpoints, `--host`, and `--port`, and now also accepts the auth/CORS,
keep-alive and queue-timeout options. `image::serve` remains available to callers
of the core library.

Errors: 404 for an unknown model/path; 400 for a model used on the wrong endpoint,
a missing/invalid selector or invalid generation parameters; 401 for failed bearer
authentication; 413 for an oversized body; 503 for queue timeout/full or load
failure. Backend generation errors retain their endpoint's existing 500/SSE-error
behavior. Body limits are 16 MB for chat/completions/embeddings, 64 KiB for image
generations and 32 MiB for edits. Invalid endpoint parameters are checked by the
existing handler after selection, so they may still cause a model switch.

**Offline evidence, 2026-11-09.** `server_models_tests.rs` exercises two chat
models, an embedder, reuse, load/unload order with an exclusive fake lock,
concurrent streamed chat versus embedding, queue timeout/recovery, failed
load/recovery, installed-versus-loaded listing, discovery/duplicates, idle
unload, auth/CORS, and reuse of image generation/edit validation. The slot test
also checks removal of a timed-out front waiter. These are CPU/fake-loader
checks, not evidence of device inference or real worker cleanup on hardware.
`phlegm-cli/tests/u7_discovery.py` is a read-only smoke against a real CLI child;
it requests only discovery/health and rejected requests, never a backend load.

Final local checks passed: core 130 unit tests plus two integration tests; CLI
five tests; seven prerequisite-dependent tests ignored. The CLI smoke listed
26 installed models (21 chat, four image, one embedding), all unloaded, and
passed four rejection routes plus health without invoking a loader. Exact
commands, results and corrected setup failures:
[u7-validation.json](u7-validation.json). No new crate or download was needed.

**Untested hardware gate.** After integration into the main checkout, run this
in one terminal (no such NPU command was run for U7):

```sh
cd /home/cyrus/code/phlegm
dibs run build --label 'U7: build integrated server' --eta 10m -- cargo build -p phlegm-cli --offline
dibs run npu port-8767 --label 'U7: multimodal switch hardware gate' --eta 90m -- target/npu-dev/npu-run.sh target/debug/phlegm serve --models-dir /home/cyrus/models --image-config /home/cyrus/Documents/ChatGPT/current/work/phlegm-image-models/phlegm-flux2-klein-4b-image.json --embed-python /home/cyrus/code/model-experiments/ternary-embedder/.venv/bin/python --keep-alive 5 --queue-timeout 3600 --port 8767
```

Then, against that freshly started, otherwise idle server, in another terminal
(the newest process named `phlegm` must be the server just started):

```sh
cd /home/cyrus/code/phlegm
python3 -I phlegm-core/tests/check_u7_hardware.py --base-url http://127.0.0.1:8767 --server-pid "$(pgrep -n -x phlegm)" --check-edits
pgrep -af 'phlegm serve|phlegm-reference-bench|embeddings_worker|flux2_native_generate'
test ! -e /tmp/phlegm-npu.lock
```

The script switches KAT Coder -> trit-embed -> Gemma 4 -> FLUX.2 klein, then edits
its generated image when `--check-edits` is supplied. It checks response shapes,
reaping of previous workers, observed descendants (PID plus start time to avoid
PID reuse), idle unload, and the free lock after images/idle. It monitors only
this server's descendants so unrelated sessions' Python/bridge processes are not
misidentified or killed. The final `pgrep` is an additional operator inventory;
the server itself remains running. Lock-free means the lockfile is absent; this
is not a numerical quality gate. The embedding Python dependencies, all selected
model/kernel/bridge bundles and the existing image config's output directory
must be available. Image process-tree behavior on abrupt server termination,
Windows parent-death behavior, and device failures remain untested.

## SERVER-EMBEDDINGS: Trit embeddings

`phlegm serve --embed-model DIR` runs the named model as an embedding-only
server and lists it in `/v1/models`. The default model name is the directory
basename; `--embed-model-name` overrides it. `--embed-python` (or
`PHLEGM_EMBED_PYTHON`) selects a Python interpreter with the model's PyTorch,
Transformers, safetensors and NumPy dependencies. `--embed-source` (or
`PHLEGM_EMBED_SOURCE`) selects the directory containing `trit_npu.py`; the
default is the source tree used to build Phlegm. The model directory must
contain the packed model, tokenizer, CPU reference, and compatible NPU bundle.
The resident Python worker uses the existing `TritNpu` implementation; Rust
holds the shared NPU lock for the worker's lifetime. The embedding-only mode
avoids loading two exclusive NPU models at once; SERVER-MODEL-SWITCHING describes
the lazy multimodal mode.

`POST /v1/embeddings` requires the served `model` and `input`: a string, an
array of strings, token IDs, or an array of token-ID arrays. Token IDs must
include the model's own special tokens and form nonempty sequences of at most
128 in-vocabulary IDs; text is tokenized and truncated by `TritNpu` at 128.
Each request accepts at most 2048 inputs and a 16 MB JSON body.
`encoding_format` is `float` (default) or `base64` (little-endian float32).
`dimensions` is unsupported and returns 400. The response contains ordered
`data[].embedding` values and the actual input-token count in `usage`.

Mock HTTP tests cover request shapes, batching, usage, errors, model listing,
and base64. An opt-in CPU test compares route vectors from a CPU-reference
backend with independent CPU-reference calls:

```sh
dibs run build cpu --label 'U6: CPU reference route test' --eta 10m -- env PHLEGM_TRIT_CPU_PYTHON=/home/cyrus/code/model-experiments/ternary-embedder/.venv/bin/python PHLEGM_TRIT_MODEL=/home/cyrus/models/trit-embed-3l cargo test -p phlegm-core --offline embeddings_route_matches_trit_cpu_reference -- --ignored --nocapture
```

The device acceptance test remains unrun: after setting a
Python environment with the dependencies and building the CLI, run, from the
checkout (replace the Python path as needed):

```sh
dibs run build --label 'U6: build embedding server' --eta 10m -- cargo build -p phlegm-cli --offline
dibs run npu port-8767 --label 'U6: trit embeddings hardware acceptance' --eta 15m -- /home/cyrus/code/phlegm/target/npu-dev/npu-run.sh /home/cyrus/code/model-experiments/ternary-embedder/.venv/bin/python -I phlegm-core/tests/trit_embeddings_hardware.py --model-dir /home/cyrus/models/trit-embed-3l --server "$PWD/target/debug/phlegm" --port 8767
```

It compares the server's text and pretokenized results, binary32 value for
value, with direct `TritNpu.encode_ids` for the same inputs. This is a pending
hardware check, not evidence that the route passed on a device.

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

`phlegm run` and `phlegm serve` default `--kernels` to `<model-dir>/open_kernels`.
Their bridge selection checks `--bridge`, `PHLEGM_REFERENCE_BRIDGE`, an adjacent
`phlegm-reference-bench` executable, then `bridges/index.json` beside the executable
using the kernel manifest family, then the build-time developer checkout's bridge tree.
Explicit paths must exist. `phlegm bench` continues to require `--reference-engine`.

## SERVER-SPECULATIVE: Verified speculative decoding

`phlegm run` and `phlegm serve` accept `--speculative off|auto|lookup|mtp`.
The per-model `speculative` registry field, settings overlay and
`PHLEGM_SPECULATIVE` follow SERVER-SETTINGS precedence. The built-in default
and every shipped registry entry are `off`. No hardware speed or exactness
claim is established by the Rust fake-bridge checks below; default-on requires
the model/method's numerical, serving and paired performance gates.

With a protocol-2 bridge advertising `verify_width > 0`, `lookup` uses the
longest repeated suffix (4 down to 2 tokens), the most recent earlier match
and its extended continuation (OPEN-SPEC-LOOKUP). `auto` selects an advertised
MTP drafter, otherwise lookup. `mtp` requires `draft: {"method":"mtp","max":K}`;
without it the request fails with a stage-4 prerequisite error. Protocol-1
bridges and bridges without verification always use ordinary decode, for all
modes. An accepted mode is not evidence of runtime MTP support.

Between cycles Rust holds one emitted token outside the committed context.
It verifies that token plus at most three drafts, bounded by the advertised
width, remaining output budget and context capacity. With no lookup match,
the pass has one row. After a rejection the draft budget halves; a completely
rejected draft triggers two width-one cycles before another probe. A fully
accepted cycle grows the budget by one, up to the advertised width.

Protocol 2 retains protocol 1's prefill, reset, rewind and decode commands:

- The process greeting accepts `protocol: 1` or `2`. Protocol 2's `init`
  adds `verify_width` (0 means unavailable), `verify_ref` (slow reference
  verifier) and `draft` (null or the MTP capability above).
- `verify_logits {"accept":m,"tokens":[t0,d1,...],"draft":j}` commits the first
  m rows of a preceding pass, then runs the supplied inputs without committing
  them. For j > 0 only t0 is supplied; the bridge appends at most j drafts.
  The reply declares `rows`, the exact input `tokens`, the committed starting
  `position`, and `logits_f32_le_bytes = rows * real_vocab * 4`, followed by
  row-major little-endian f32 logits. Rust validates the row count, input IDs,
  position, payload size and finite values before walking the rows.
- `accept {"n":m}` commits only the first m pending rows and discards the rest.
  This client explicitly accepts after each cycle, so its next verify command
  carries `accept:0`. Reset, rewind, prefill and decode require no pending pass.

Rust samples only valid rows, once each, in order, using the request's sampler,
history, penalties, ramps, RNG and structured-output mask. A masked-out draft
cannot match the sampled token and is rejected. Only sampled, verified tokens
reach `TokenEvent`, with their raw model logits; drafts never reach callbacks.
Bit-identical valid verifier rows imply identical greedy and seeded output
to ordinary decode. Device row identity remains a separate hardware gate.

Stop IDs, output limits and callback `Break` end the walk immediately. Stop
strings continue to use SERVER-GENERATION's callback. On `Break`, Rust starts
no new pass and runs no new token, but sends the final `accept` needed to
discard the pending suffix. The token whose callback broke remains unrun.
Only acknowledged committed input tokens enter the reusable context; the
prefill mark is preserved. Failed verification/commit leaves no reusable
context or mark. Streaming, reasoning splitting and log probabilities use
the same verified callback path as ordinary decode.

CPU acceptance tests (2026-11-09): `speculative::tests` covers all 13 existing
lookup fixtures and policy bounds. `bench::reference::speculation::tests`
checks greedy plus five seeded sampler configurations, adversarial drafts,
U4 masks, stops, early breaks, capacity/output bounds, committed context and
marks, follow-up prefix reuse, protocol-1 fallback and malformed replies.
`server::tests::speculative_bridge_stream_stop_and_reuse_match_plain_http`
checks SSE/non-stream text, a stop string ending inside a cycle, rollback and
subsequent HTTP prefix reuse with the protocol-2 fake. These tests load no
hardware and do not validate C++ verification or MTP implementations.

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
whatever `do_sample` says. Clover's entry: greedy, presence 0.9, frequency 0, repetition 1.05 on
the response, length 10 from 8192 over 8192, think-end 10 from 4096 over 4096 (its model card and
generation_config.json). Tests: `presence_and_frequency_count_the_reply_only`,
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

## SERVER-TOOLS: Chat Completions tool calling (U3)

`/v1/chat/completions` accepts OpenAI function `tools` and `tool_choice` (`auto`,
`none`, `required`, or `{"type":"function","function":{"name":"..."}}`). Bad choices,
unknown named functions, duplicate names and malformed tool lists are HTTP 400.
`none` hides the definitions and disables output parsing. A named choice exposes
only that function. Required/named choices also add a system instruction because
publisher templates generally do not inspect `tool_choice`; this is prompting,
not a guarantee that the model will obey. Invalid model output remains plain text.

Tool definitions and the choice reach the Jinja context. Assistant `tool_calls`,
call IDs, tool messages' `tool_call_id`/`name`, and `reasoning_content` survive the
request. JSON-string arguments are deserialized into objects before rendering;
invalid argument JSON is HTTP 400. GPT-OSS additionally receives `thinking` as
its template's alias for `reasoning_content`.

The model's template selects one of six parsers (`tool_calls.rs`): function XML;
JSON/key-value tool tags; Gemma's `call:name{...}`; Hy-Memory's tool list (including
DeepSeek tokens); LFM2.5's function list; or harmony recipient headers. Installed
Flint and Xing templates actually use `arg_key` and `param_key` tags respectively,
although the November roadmap described this group as JSON. Both are supported.
LFM2.5 also emits a bare function list in the recorded CPU fixture.

Successful calls receive opaque `call_...` IDs, a function name and JSON-string
arguments, and finish with `tool_calls`. Text preceding a call stays in `content`;
reasoning remains in `reasoning_content`. An empty content beside calls is null.
SSE emits indexed `delta.tool_calls` entries and the same finish reason. Tool-enabled
requests stream ordinary content and reasoning as they arrive. Each parser holds
back a suffix that could start its call marker, then buffers from the first call
marker through final validation (one delta per validated complete call). LFM's
recorded bare-list variant treats `[` as a possible opener. Harmony holds incomplete
headers until it knows their recipient, and streams commentary as well as final
content. Malformed, truncated or unknown calls fall back to content, including the
held marker; already streamed prefixes are not duplicated. Text and reasoning
before a call therefore remain available during generation.
Unknown function names also fall back to content. No model output is evaluated.

Verification (2026-11-09): six small CPU-generated fixtures, with exact prompts,
llama.cpp identity and command, live in `phlegm-core/tests/fixtures/tool_calls/`.
They cover KAT Coder, Gemma 4 26B-A4B, GPT-OSS 20B, Flint, Hy-Memory and LFM2.5.
The parser suite additionally exercises documented JSON and DeepSeek variants.
`server::tests::tool_loop_streaming_and_full_and_malformed` covers two-step HTTP
loops, `none` and malformed output in both modes. The ignored local-files test
`installed_templates_round_trip_tools` exercises the installed templates; the
six protocol groups exclude K2's separate, unsupported protocol. Hardware SDK
loops remain untested; offline tests do not establish hardware acceptance.

## SERVER-STRUCTURED-OUTPUT: JSON and schema token constraints (U4)

`response_format` accepts `{"type":"json_object"}` and the OpenAI
`{"type":"json_schema","json_schema":{"name":"...","strict":true,"schema":{...}}}`
shape. `strict` must be boolean when supplied. Both strict and non-strict schema
requests enforce the supported schema constraints; strict never silently relaxes
one. Omitted/null `response_format` and `{"type":"text"}` follow the ordinary
sampler path without constructing a vocabulary, grammar, or mask.

`structured_output.rs` compiles a byte grammar into incremental parser stacks.
A cached vocabulary trie uses the tokenizer's inverse ByteLevel mapping or
SentencePiece replacement/byte-fallback mapping; it preserves partial UTF-8 bytes,
rather than decoding individual tokens with replacement characters. Traversing
that trie against the current parser stacks computes which complete tokens keep
a valid prefix. The sampler masks other logits after penalties and before greedy
selection or temperature/top-k/top-p. EOS is allowed only when a complete value
has been accepted. Invalid UTF-8, unpaired escaped surrogates, special control
tokens and IDs outside the tokenizer vocabulary are excluded. An empty candidate
set is a generation error, never an unconstrained fallback.

Supported schema keywords: `type` (also type arrays), `properties`, `required`,
`enum`, `const`, `items`, `anyOf`, and boolean/schema `additionalProperties`.
Types: object, array, string, number, integer, boolean and null. Nested boolean
schemas are supported. Annotations `title`, `description`, `default`, `examples`,
`$schema`, `$id` are accepted without changing validation. Other features,
including `$ref`, `oneOf`, `allOf`, numeric bounds and string patterns, are HTTP 400
with the offending feature's schema path. Invalid/unsatisfiable schemas are also
HTTP 400. Current construction limits: depth 48, 256 properties per object,
64 alternatives per `anyOf`.

Generation uses a valid subset of the schema language: declared object keys are
emitted in sorted order, optional properties may be omitted, and permitted extra
properties need not be emitted when declared properties exist. Generic objects
have arbitrary keys. Unconstrained numeric productions use finite decimal values
(up to 15 integral and 9 fractional digits); enum/const use their exact JSON
literals. This avoids numeric overflow in consumers. These choices restrict
representation without allowing values outside the requested schema.

Structured responses constrain the entire generated reply and disable reasoning
splitting (and request `enable_thinking: false` from templates). They can coexist
with `tools` in auto/none mode: plain JSON remains content. Argument-only tool
constraints and mixing mandatory tool calls with whole-reply JSON constraints
are not implemented. As for any finite output budget, `finish_reason: length`
can leave an incomplete prefix; clients must not treat a truncated reply as a
complete JSON value.

Verification: grammar tests cover nested objects/arrays, unions, enums/constants,
UTF-8 and escapes, spanning tokens, EOS, invalid schemas and unsupported features;
`constraints_win_over_logits_penalties_and_sampled_candidate_lists` covers greedy
and stochastic sampling. The fake-logits HTTP test exercises both response modes
and tools in auto mode. The ignored `every_registry_tokenizer_random_walks` test
uses all 18 registry entries' actual tokenizer files, samples random valid prefixes,
completes them, parses the result with serde_json, and independently validates it
against the schema; it also checks the actual IDs' decoded output. Run it offline
with the local models installed. Hardware validation remains separate.

### U3/U4 offline evidence, 2026-11-09

Core checks: 96 unit tests passed, four prerequisite tests ignored; the release-report
and CPU tool-fixture integration tests passed. CLI: three tests passed. Explicit
local-files checks passed for all 18 tool-capable installed templates and all 18
registry tokenizers (five schemas, two random walks each, with grammar completion).
Two hardware/native bridge checks remain ignored. The SDK client is
`phlegm-core/tests/check_tools_sdk.py`; its Python syntax was checked, its hardware
loop was not run. No NPU execution or new dependency crate was used.

Dibs-isolated CPU development microbenchmark, CLEAN before and after, Gemma's
262,144-token vocabulary, required `city` string property, parser positioned just
inside the string: the first mask at that state took 87.65–101.12 ms (three samples,
median 95.95 ms); cached repeated-state masks took 0.1223–0.1449 ms (nine samples,
median 0.1278 ms). These timings include applying the mask to a full logit vector,
exclude tokenizer/vocabulary loading, and are not end-to-end HTTP or NPU timings.
New parser states pay a cache miss; this is not a universal per-token latency.
The vocabulary is shared per tokenizer, transitions are memoized per request,
and up to 32 completed masks are cached per request.

Retained negative experiment: the first, uncached implementation spent
2194.38–2287.69 ms constructing the string-state mask (nine samples, median
2231.03 ms, without the subsequent logits mask). It was replaced after measurement.
An initial timing attempt never reached CLEAN because host CPU stayed at 76–91%;
it was canceled and its reservation released. All completed measurement series
reported CLEAN before and after. Host: Strix Halo Ryzen AI MAX+ 395, Linux
7.2.13-200.fc44.x86_64, `powersave` CPU governor, rustc 1.99.1, optimized Rust build;
base revision `7742985b53e464a1a8171321fe4e33281905ec8a` with these uncommitted changes.
The pre-fast-path sampler comparison exposed a greedy regression, retained as a
failed development experiment; the final unconstrained path is separately compiled
and stores the inactive constraint behind an optional pointer.

The final paired sampler comparison also reported CLEAN before and after. Both
samplers were called through identical out-of-line wrappers, with black-boxed
state and logits, one warmup pair discarded, and 14 retained alternating pairs
of 100 tokens each. On the same 262,144 logits, unconstrained greedy took a median
0.132 ms/token versus the pre-change sampler's 0.170; temperature 1 / top-k 64 /
top-p 0.95 took 0.407 versus 0.443 ms/token. Every retained final pair favored the
new sampler. No slowdown was observed in these two sampler cases. The bare greedy
path avoids copying logits when penalties are inactive; non-greedy and penalized
paths retain their transformations. These are CPU component measurements, not an
end-to-end HTTP performance or hardware acceptance claim. Raw samples, provenance,
commands and retained failures: [u3-u4-validation.json](u3-u4-validation.json).
