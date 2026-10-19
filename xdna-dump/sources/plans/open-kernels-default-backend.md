# Serve on the open kernels by default; closed kernels behind a flag

**Status:** plan, not started (2026-09-03).
**Why now:** the chat server produces gibberish, and the cause is the backend it
runs on. See "Diagnosis" — this is not a UI or a weights problem.

## Diagnosis (measured 2026-09-03)

`open-qwen-npu serve --backend l30` drives FLM's **closed** `layer.xclbin`
([generate_l40.rs:61](../../npu-engine/src/generate_l40.rs#L61)), and that kernel
is the one already shown to contribute nothing from the full-attention block on
`full_attention_interval=3` models (README "Diagnosed (2026-09-02)",
[[flm-capture-oracle]]). The 27B is interval-3, so ten of its thirty layers are
effectively missing.

Greedy, prompt `output 5 paragraphs about pasta`, server vs the CPU replica
(`open-qwen-npu run`) on the identical context:

| context | server emitted | replica top-1 | gap |
|---|---|---|---|
| prompt (16 tok) | `Here` (8160) | `Here` (8160) | agree |
| + `Here's a thinking process:` | `\n` (198) | `\n\n` (271), 27.6 vs 19.8 | 7.8 logits |
| + `\n` | `2` (17) | `1` (16), 24.1 vs 16.6 | 7.5 logits |

With the full-attention contribution dropped from the replica
(`OPEN_QWEN_ZERO_FULL_ATTN=1`, [forward.rs:378](../../npu-engine/src/forward.rs#L378) —
a diagnostic added for this, inert unless set) both divergences collapse to
near-ties: 15.32/14.62 and 20.35/19.92. The server is behaving like a model whose
attention layers contribute almost nothing, which is exactly the documented
defect. End-to-end throughput of that path today: 48 tokens in 11.5 s (~4.2 tok/s).

The open kernels do not have this defect: the 27B step through `layer_x` matches
the replica at **corr 0.999998, same argmax and top-5, every layer's residual
>= 0.999997**, in 165 ms/token (6.1 tok/s) — faster than the closed path as well
as correct.

## Goal

`serve` and the run CLIs use the open kernels by default; FLM's closed kernels
stay reachable behind `--backend closed` (keep `l30` as an alias) for
comparisons and for the base interval-4 40L model.

## What already exists

- **Kernels:** `designs/layer_x` (`lx0/lx1` linear layer, `ax0/ax1` attention
  layer, whole layer in one xclbin context), `designs/ln`, `designs/lm_head_q8`
  (full vocab, 15.6 ms). 62 dispatches over 4 contexts per token.
- **Dynamic KV:** `attnpos <kernel> <pos>` patches the shared `ax0` stream
  (three words + one instruction-BO sync, 0.04 ms); `kv{l}` is one 8 MB BO per
  attention layer, `MAX_CTX` 4096; `ptab` holds the per-position RoPE record.
  Verified over three greedy positions (argmax = replica's at 846, 198, 3710).
- **Driver:** [decode.rs](../../npu-engine/src/decode.rs) already interprets the
  whole config language, and `load_resident` / `serve` / `endserve` already give
  "set up once, run this program per token" (used by `step-test`).
- **Weights:** `pool_L{l}.bin` from `l30-build` bind directly; the rest of the
  per-layer buffers (`consts2_*`, `constsa2_*`, `ptab`, `zstate2`, `lm27.bin`,
  `normw`) are built and sitting in `designs/decode_chain/w27/` (2.4 GB).

So this is wiring, not new kernel work.

## Work

1. **Make the resident step run the real program language.**
   `Driver::step_bytes` ([decode.rs:723](../../npu-engine/src/decode.rs#L723))
   re-implements a five-directive subset (`load`, `runlist`, `layer`, `submit`,
   `barrier`) — the closed-kernel chain's vocabulary, no `run`, `moeroute2` or
   `attnpos`. Replace its body with a loop over `exec_line`, the same
   interpreter the setup phase uses, and expose:
   - `write_buf(name, bytes, off)` / `read_buf(name, len, off)`
   - `exec_prog(prog: &[String])`
   - `set_attnpos(kernel, pos)` (typed entry to the existing directive)

   Keep `step_bytes` as a thin wrapper so `step-test` stays byte-identical.

2. **A serve config.** `make_27b.py --serve` writes `serve_27b.cfg`: the same
   buffer block as `run_27b_x.cfg` (pools, consts, states, kv, ptab, lmpool),
   `xres` declared without an init file, **no `dump` lines**, and a
   `serve` / `endserve` block holding one token's program — 30 x (`run lx0|ax0`,
   `moeroute2`, `run lx1|ax1`) + `run ln` + `run lm`. The per-token `load xres`
   and `attnpos` lines come out; the backend does both itself.

3. **`generate_open.rs` — `OpenBackend`,** implementing `backend::Backend`:
   - open: `decode::load_resident(cfg)`, hold the `Driver` + program resident
     across requests exactly as `L40Backend` holds its BOs.
   - per request: write zeros to every `state{l}` and `kv{l}`, `pos = 0`.
   - per token: `write_buf("xres", embed(tok) as f32[2048])`,
     `set_attnpos("ax0", pos)`, `exec_prog`, `read_buf("logits")` ->
     `f32[248320]`, sample; decode-as-prefill over the prompt first, same shape
     as `L40Backend::generate`.
   - reject `prompt + max_tokens > 4096` (`MAX_CTX`), the way the l30 path
     rejects > 1024.
   - on error, drop residency so the next request reopens.

4. **CLI / server.** Default `--backend` becomes `open`; `--backend closed`
   (alias `l30`) selects the FLM kernels; `mock` unchanged. Add `open-run
   [prompt] [max_tokens]` next to `l30-run`. `/health` reports `open`. Config
   path from `OPEN_QWEN_OPEN_CFG`, defaulting to the generated `serve_27b.cfg`.

5. **Verification** (in order):
   - The regression that started this: `open-run "output 5 paragraphs about
     pasta" 24` must emit `\n\n` then `1.` — the replica's argmax at both
     contexts above (271, then 16), not the closed path's `\n` / `2`.
   - Teacher-forced spot checks against `open-qwen-npu run` at 3-4 further
     contexts along a ~30-token greedy generation.
   - Two requests in one process (state/KV reset), mirroring
     `generates_twice_on_real_hardware`.
   - Soak: 200 tokens in one process, watching for XRT resource accumulation
     and for a context switch resetting the array mid-token.

## Risks / unknowns

- **Only three positions have ever been verified.** At position 1 the logits
  corr is 0.9945 (layer 23's attention output cancels its input, so bf16
  rounding of the delta is 2.7 % of the residual and layers 25/26 amplify it —
  see `open-kernels-phase2-dynamic-kv.md`). Long greedy runs are unproven; the
  soak test is where that shows up.
- **Memory:** ~16 GB of BOs (30 x 512 MB pools + 542 MB lm pool + 30 x 11.9 MB
  consts + 10 x 8 MB kv). The open and closed backends cannot both be resident;
  `serve` picks one.
- **Prefill is still sequential decode** at ~165 ms/token: a 500-token prompt is
  ~80 s. Same order as the closed path, but it caps usable prompt length.
- **Artifacts are still built by WSL Python** (`make_27b.py`). Acceptable as a
  one-time build step now; the follow-up is an `open-build` that writes
  `consts2_*` / `constsa2_*` / `ptab` / `lm27.bin` from the q4nx in Rust, the
  way `l30_buffers.rs` does for the closed path.
- Process teardown segfaulted after 19 contexts in the one-shot harness
  (harmless, after all work). Four contexts here; watch it anyway.

## Not in scope

Deleting the closed path, porting the artifact builders to Rust, and the
remaining phase-2 speed items (router fold, design B, lm_head).
