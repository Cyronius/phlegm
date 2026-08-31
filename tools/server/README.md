# tools/server — OpenAI-compatible HTTP server for the open NPU engine

A zero-dependency (stdlib-only) HTTP server that exposes the open Qwen3.6-MoE NPU
executor behind an **OpenAI `/v1/chat/completions`** API — streaming (SSE) and
non-streaming — plus `/v1/models` and `/health`.

It wraps the resident `decode_driver.exe` (serve mode) that runs the interval-3
model on the Ryzen AI NPU (the config FLM's closed engine NaN-collapses on). The
HTTP layer is backend-agnostic: a **mock** backend (default, no device) makes the
whole API testable with zero hardware, and an **npu** backend drives one resident
driver process.

## Launch

```bash
# Mock backend (default) — no NPU, fully functional API for testing/integration
python -m tools.server

# Real NPU backend — drives the resident decode_driver.exe on the shared NPU
FLM_BACKEND=npu python -m tools.server

# Auto: npu iff the driver + built buffers exist on disk, else mock
FLM_BACKEND=auto python -m tools.server
```

Server listens on `127.0.0.1:52625` by default (FLM's own port). Override with
`FLM_HOST` / `FLM_PORT`.

On Windows PowerShell, set env vars first:

```powershell
$env:FLM_BACKEND="npu"; $env:FLM_PORT="52625"; python -m tools.server
```

## Endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET  | `/health` | `{"status":"ok","backend":"mock\|npu"}` |
| GET  | `/v1/models` | OpenAI model list (one model: `FLM_MODEL_ID`) |
| POST | `/v1/chat/completions` | OpenAI chat schema; `stream:true` → SSE |

Request fields honored: `messages[]`, `model`, `max_tokens`, `temperature`,
`top_p`, `seed`, `stream`. `max_tokens` is capped by `FLM_MAX_TOKENS_CAP` (512).

### Example — non-streaming

```bash
curl -s http://127.0.0.1:52625/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-5li3-npu",
       "messages":[{"role":"user","content":"hi there"}],
       "max_tokens":40}'
```

```json
{"id":"chatcmpl-…","object":"chat.completion","choices":[
  {"index":0,"message":{"role":"assistant","content":"…"},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":25,"completion_tokens":40,"total_tokens":65}}
```

### Example — streaming (SSE)

```bash
curl -s -N http://127.0.0.1:52625/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"stream please"}],
       "stream":true,"max_tokens":30}'
```

Emits `data: {…chat.completion.chunk…}` frames with incremental `delta.content`,
a final `finish_reason` frame, then `data: [DONE]`.

### Device-busy (NPU backend)

The NPU is a **single shared device** and the driver holds it exclusively while
resident. The driver starts **lazily on the first request**. If it can't acquire
the device (fails to reach `SERVE READY`, or dies mid-step), the server returns
**HTTP 503** with `Retry-After: 5` and `{"error":{"type":"device_busy"}}`.
Concurrent requests are serialized by a per-backend lock (one decode step at a
time), so the API is safe under parallel clients.

## How the NPU backend drives generation

`backends.NpuBackend` reproduces `tools/kernel-interp/generate_npu.py` (its
serve-driver mechanism is **copied**, not imported, to avoid that script's 2 GB
import-time lm_head build and to keep it unmodified):

1. Lazily launches **one** `decode_driver.exe <serve-config>`. The config lists
   the layer/lm_head xclbins, control ELFs, and the resident BOs (per-layer 512 MB
   pools, 2 MB packs, 6 MB sides, 3 MB states, the 517 MB lm_head pool), then a
   `serve` program of `runlist`/`submit`/`barrier` lines (≤3-layer runlist chunks
   with a cross-context barrier — the ERT queue-depth workaround).
2. Waits for `SERVE READY` (pools + prefill states resident, ~7 s startup).
3. Per token: writes the act buffer `[embed(tok) | model.norm.weight]` to a file,
   sends `step <act> <hidden>` over stdin, waits for `STEP OK`, reads the NPU's
   final hidden, computes **full logits on CPU** (cached dequantized lm_head
   `[248320, 2048]`), samples, feeds the next embedding. ~0.1 s/token.

Streaming is exposed by making `generate()` a Python generator that `yield`s each
token id as the driver returns it; the HTTP handler decodes the growing id list
incrementally and writes one SSE `delta.content` frame per revealed piece.

## Integration points (handoffs)

These are deliberately thin and swappable in one line each:

- **Tokenizer** — `tokenizer.py::get_tokenizer()`. Ships `PlaceholderTokenizer`
  (UTF-8 byte identity codec; real vocab ids render as `⟨id⟩`). Replace the body
  of `get_tokenizer()` with the real tokenizer being built under
  `tools/kernel-interp/`. Interface: `encode`, `decode` (must be
  incremental-prefix safe), `apply_chat_template`, `eos_id`.
- **Sampler** — `sampler.py`. Greedy by default; `temperature`/`top_p`/`seed`
  paths are implemented (numpy nucleus sampling with a pure-python fallback). No
  stub — wire different strategies here.
- **Per-request prefill (the real gap)** — today the NPU backend generates a
  *continuation of a FIXED prefill*. The `state_L*.bin` buffers are built offline
  by `run_5li3_npu.py` for one baked prompt and seeded from `first_token.npy`; the
  request's `messages` flow through the tokenizer and params but do **not** yet
  re-prefill the NPU. Wiring arbitrary-prompt prefill means: tokenize the request
  → run the CPU prefill (the `*_prefill` functions in `run_5li3_npu.py`) to build
  fresh `state_L*.bin` → `load` them into the resident driver before stepping.
  That is the main remaining backend task.
- **Full 30-layer model** — the pool builder and driver config are general.
  Point `FLM_MODEL_Q4NX` / `FLM_NUM_LAYERS` / `FLM_NPU_OUT_DIR` at a full-model
  build; the serve-config generator already chunks any layer count into ≤3-layer
  runlists.

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `FLM_BACKEND` | `mock` | `mock` \| `npu` \| `auto` |
| `FLM_HOST` / `FLM_PORT` | `127.0.0.1` / `52625` | bind address |
| `FLM_MODEL_ID` | `qwen3.6-5li3-npu` | advertised model name |
| `FLM_MAX_TOKENS` / `FLM_MAX_TOKENS_CAP` | `64` / `512` | default & hard cap |
| `FLM_NPU_OUT_DIR` | `npu-engine/m3out/5li3` | built pools/packs/sides/states |
| `FLM_DRIVER_EXE` | `npu-engine/m0/out/decode_driver.exe` | serve driver |
| `FLM_XCLBIN_DIR` | `src/xclbins/Qwen3.6-35B-A3B-NPU2` | kernels |
| `FLM_CAP_DIR` | `C:/caps/m0c` | control ELFs |
| `FLM_MODEL_Q4NX` | `model_5Li3.q4nx` | model file (in q4nx `MODEL_DIR`) |
| `FLM_NUM_LAYERS` | `5` | layer count for the serve config |
| `FLM_DRIVER_READY_TIMEOUT` | `120` | seconds to wait for `SERVE READY` |
| `FLM_DRIVER_STEP_TIMEOUT` | `30` | seconds to wait per decode step |

## Files

```
tools/server/
  __main__.py    python -m tools.server entry point
  app.py         stdlib HTTP server + OpenAI request/response + SSE
  backends.py    Backend base, MockBackend, NpuBackend (resident driver), factory
  tokenizer.py   Tokenizer interface + PlaceholderTokenizer (swap point)
  sampler.py     greedy + temperature/top-p
  config.py      paths + env overrides (importing it is free)
  README.md      this file
```

## Validation

Mock backend, verified with curl:
- `GET /health` → `{"status":"ok","backend":"mock"}`
- `GET /v1/models` → model list
- `POST /v1/chat/completions` non-streaming → full completion JSON + usage
- `POST …` `stream:true` → SSE `chat.completion.chunk` deltas + `[DONE]`
- empty `messages` → 400; unknown path → 404

Real-NPU wiring is code-complete (serve-config generation, lazy driver start,
step protocol, 503-on-busy) but was **not exercised here** to avoid contending
for the shared NPU while other agents use it. To smoke-test on the device:
`FLM_BACKEND=npu python -m tools.server`, then the curl examples above — expect a
~7 s first-request delay while pools load, then ~0.1 s/token.
