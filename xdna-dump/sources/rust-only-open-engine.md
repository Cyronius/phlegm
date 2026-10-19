# Rewrite the open NPU engine in Rust, remove Python from the runtime path

## Decisions made (confirmed with Cyrus)

| Question | Decision |
|---|---|
| Pool-builder / model-conversion step (`build_pools.py`, `l30_build.py`, `pools_only_l40.py`, `run_5li3_npu.py`'s buffer prep) | **Port to Rust.** Same `.q4nx` format `q4nx.rs` already parses; removes the last Python dependency between "downloaded model" and "running server." |
| CPU oracle scripts (`decode_step.py`, `full_forward.py`, `l30_forward.py`, `moe_forward.py`) | **Retire.** `forward.rs` already does this job; no reason to maintain two independently-implemented CPU references. |
| Target model for the rewritten generate loop | **l30 (30-layer pruned model) is primary and must use pool streaming** (30×512MB pools can't stay resident, unlike 5li3's 5). **5li3 stays alive as a fast/small config** for quick iteration. Both loops converge on one generate function, parameterized by layer schedule. |

**Stays Python, unchanged** (research/RE tooling, never touched at runtime):
`crack_q4.py`, `find_weights.py`, `op_table.py`, `hf_fetch.py`, `q4nx.py` (superseded exploratory reader — `q4nx_v103.py`/`q4nx.rs` are the real ones now).

**Stays Python, but needs updating**: `bench_e2e_l40.py` — it shells out to whatever binary you give it; point it at the new Rust binary instead of `generate_npu.py` once that exists, so the "Rust vs FLM" comparison from the last benchmark becomes a real all-Rust fight instead of Rust-driver-wrapped-in-Python-glue.

## Current state (what's already real Rust, verified by reading the code)

- **`q4nx.rs`** — reads the `.q4nx` container format (header, tensor table, bf16/quant raw reads). Used by `forward.rs` and `main.rs`.
- **`forward.rs`** — CPU reference: `embed()`, `linear_attn_prefill/decode`, `full_attn_prefill/decode`, `moe`, `logits()`, `rms_norm`. This is the oracle that makes the 4 Python CPU-forward scripts redundant.
- **`xrt.rs`** — the XRT FFI shim (Device/Context/Kernel/Bo/Runlist over `xrt-shim`). C++ boundary, unavoidable, not in scope to remove.
- **`decode.rs`** — a **faithful Rust port of `decode_driver.cpp`**, not yet a library. Its `Driver::serve()` still speaks the exact same text protocol over **stdin** and does per-step activation/hidden I/O through **files** (`step <act_in> <hidden_out>`). It's spawned as its own subprocess via `main.rs`'s `npu` subcommand, same as the C++ driver. This is the actual target of the port below — the shape doesn't change, but the process/file boundary does.

## What's still Python and why it's the real gap

`tools/kernel-interp/generate_npu.py` (5li3 loop), `l30_run_npu.py` (`gen` mode, l30 loop), and `tools/server/*` collectively do everything `decode.rs`/`forward.rs`/`q4nx.rs` don't:

1. **Tokenizer** (`tools/kernel-interp/tokenizer.py`, `Qwen36Tokenizer`) — real byte-level BPE + chat template.
2. **Sampler** (`tools/kernel-interp/sampler.py`) — temperature / top-k / top-p / repetition penalty.
3. **lm_head** (`_build_lmhead_matrix` / `full_logits` in both `generate_npu.py` and `l30_run_npu.py`, duplicated a third time in `tools/server/backends.py`) — dequantizes the 517MB `lm_head.weight` into a cached ~2GB f32 `[248320, 2048]` matrix, then one matmul per token. This is the ~15-20ms/token numpy cost from the earlier benchmark.
4. **The generate loop itself** — spawns `decode_driver.exe` as a subprocess, writes `gen_act.bin` to disk, sends `step <in> <out>\n` over stdin, polls stdout for `STEP OK`, reads `gen_hidden.bin` back off disk. This file round-trip per token is the other big cost from the earlier benchmark.
5. **The HTTP server** (`tools/server/app.py` + `backends.py`) — OpenAI-compatible `/v1/chat/completions` (SSE), `/v1/models`, `/health`. `backends.py` literally says it copy-pastes pieces of `generate_npu.py` "so we don't trigger its 2GB import-time side effects" — a tell that the Python architecture is already straining.
6. **Model conversion** (`build_pools.py`, `l30_build.py`, `pools_only_l40.py`) — one-time step that packs a `.q4nx` file into the pool/pack/side/state buffer files the driver config references.

## Known functional gap, independent of language

`backends.py`'s `NpuBackend` docstring is explicit: today, per-request prefill of an arbitrary prompt is **not wired**. The server generates a continuation of one fixed, offline-built prefill regardless of what the request actually asked. Porting this to Rust does not fix it by itself — the pool-builder port (item 5 below) has to actually run prefill per request (reusing `forward.rs`'s existing CPU prefill math) for the server to be real. Flagging this now so it doesn't surface as a surprise mid-port.

## Target shape

One Rust crate (today's `open-qwen-npu`), extended with subcommands mirroring `generate_npu.py`'s CLI and `tools/server`'s HTTP surface:

```
open-qwen-npu run    --prompt "..." [--system ...] [--raw] [--no-think]
                      [--max-tokens N] [--temperature T] [--top-k K] [--top-p P]
                      [--rep-penalty R] [--seed S] [--model 5li3|l30]
open-qwen-npu serve  [--port N] [--model 5li3|l30]     # OpenAI-compatible HTTP
open-qwen-npu build  <model.q4nx> --schedule 5li3|l30  # pool/pack/side/state build
```

No subprocess, no file-based act/hidden hand-off — the driver becomes a library called in-process.

## Work items, in dependency order

1. **Turn `decode.rs`'s `Driver` into a library, not a subprocess.**
   Replace the stdin-protocol `serve()` loop with a plain method — e.g. `Driver::step(&mut self, act: &[u8]) -> Result<[u8; 8192], String>` — that writes straight into the resident `act` `Bo` and reads the resident `act` `Bo` back, no temp files, no process boundary. This alone removes the per-token disk round-trip identified as ~38ms of the previous 151ms/token Python overhead.

2. **Port lm_head dequant + matmul** (`_build_lmhead_matrix` / `full_logits`) into a new `lm_head.rs`, reusing `q4nx.rs`'s raw tensor access and `bf16_to_f32`. Keep the on-disk f32 cache (or an equivalent) — re-dequantizing 517MB every process start is wasteful regardless of language.

3. **Port the sampler** (`sampler.rs`) — temperature / top-k / top-p / repetition penalty. Small, no external crate needed.

4. **Tokenizer.** Before hand-porting `Qwen36Tokenizer`'s BPE: check whether the model ships a standard `tokenizer.json`/`vocab.json`+`merges.txt` and whether Hugging Face's `tokenizers` Rust crate loads it directly. If yes, that's a dependency add, not a port. If the chat-template/special-token handling is nonstandard, port by hand from `tokenizer.py`.

5. **Port the pool-builder / model-conversion step** (`build_pools.py`, `l30_build.py`, `pools_only_l40.py`, and `run_5li3_npu.py`'s buffer-prep half) into Rust, reusing `q4nx.rs` for tensor reads and `forward.rs` for the CPU prefill math that produces `state_L*.bin`. This is also where the "known functional gap" above gets closed — make this produce real per-request prefill state, not just a fixed offline prefill.

6. **Assemble the generate loop**: tokenizer → CPU prefill (`forward.rs`) → in-process `Driver::step` (item 1) → `lm_head.rs` (item 2) → sampler (item 3), for both the 5li3 (resident pools) and l30 (streamed pools, load poolA/B/C per group of 3 layers, exactly as `l30_run_npu.py::gen_stream_cfg` does today) schedules.

7. **Port the HTTP server** (`tools/server/app.py` + `backends.py` + `config.py`) into Rust — `/v1/chat/completions` (SSE + non-streaming), `/v1/models`, `/health`, backed by item 6. Pick a minimal-dependency HTTP crate (the Python version deliberately used only `http.server` from stdlib — match that spirit; e.g. `tiny_http` rather than a full framework).

8. **CLI wiring** in `main.rs`: `run` / `serve` / `build` subcommands as sketched above.

9. **Delete** `decode_step.py`, `full_forward.py`, `l30_forward.py`, `moe_forward.py` (superseded by `forward.rs`, confirmed above).

10. **Update `bench_e2e_l40.py`** to invoke the new Rust binary instead of `generate_npu.py`, so the FLM comparison benchmark is genuinely Rust end-to-end.

## Explicitly out of scope

`crack_q4.py`, `find_weights.py`, `op_table.py`, `hf_fetch.py`, `q4nx.py` — one-off reverse-engineering scripts, never called at runtime, left untouched.

`xrt-shim/` and the C++ XRT boundary in general — XRT is a C++ API, this isn't Python and isn't going away.
