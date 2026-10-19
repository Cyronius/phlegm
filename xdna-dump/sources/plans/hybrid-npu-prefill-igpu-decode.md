# Hybrid execution: NPU prefill, iGPU decode

**Status:** proposal, not started. Written 2026-09-13.
**Related:** [[npu-prefill.md]] (Phase 2 batch prefill, never built),
[[open-kernels-feasibility.md]] (NPU-vs-iGPU bandwidth analysis),
[[open-kernels-phase3-model-recipes.md]].

## The short version

The phase split is right in theory and backwards in payoff order. Decode is
DRAM-bandwidth-bound and the iGPU shares the same DRAM but pulls more of it, so
moving decode to the 890M is a 2-4x win available with zero new code. Prefill is
compute-bound and is the only place XDNA2's 50 TOPS can beat the iGPU — but
phlegm has no batched kernels, so NPU prefill today is 10-14 tok/s (sequential
decode) against an iGPU that will do hundreds.

So the hybrid is worth building only if NPU batch prefill beats the 890M's
prefill. That is an open question with a cheap decisive experiment in front of
it (Stage 0). Everything after Stage 0 is gated on the number it produces.

And "pull pieces from llama.cpp into phlegm" is the wrong direction. The right
move is to make phlegm a **ggml backend** that llama.cpp schedules onto. The
in-tree precedent for exactly this is `ggml/src/ggml-hexagon/`.

## Three findings that change the framing

**1. llama.cpp already runs this model correctly, at interval 3.**
`src/models/qwen35moe.cpp:24-28` reads `full_attention_interval` from GGUF
metadata and computes `is_recr_impl[i] = (i+1) % interval != 0` per layer — no
special case for 4, none needed for 3. `llama-model.cpp:2303` then builds a
`llama_memory_hybrid` with attn/recurrent layer filters off that same flag. The
FLM NaN bug has no analogue there. The GGUF is already on the box
(`C:/Users/josha/.flm/models/Qwen3.6-27B-A2.8B-open/qwen36-27b-a2.8b-mtp-Q4KM.gguf`,
`general.architecture = qwen35moe`, `full_attention_interval = 3`,
`block_count = 31` = 30 layers + 1 MTP).

This removes the correctness reason for phlegm on any iGPU path. What is left is
the performance question, which is a different project.

**2. Decode on the NPU is at its architectural floor.**
`docs/lm-head-npu-bottleneck-instrumentation.md:447` already concluded this:
per-token device time is ~108 ms, ~5 GB of weight traffic at ~50-60 GB/s
effective, memory-bound, and "an iGPU on the same SoC shares the same DRAM, so
iGPU-class means this same bound." The 890M reaches a larger fraction of the
LPDDR5x-7500 peak than the NPU does, and llama.cpp's Vulkan MoE path is mature.
Expect a straight multiple, not a rewrite.

**3. There are no batched kernels to prefill with.**
`tools/open-kernels/designs/` has `gemv_q4` and no GEMM — every design is
decode-shaped. `docs/npu-prefill.md` Phase 2 (true batch prefill) was scoped at
weeks and never started. The cited prior art is not encouraging: TileFuse got
XDNA2 prefill only "competitive with the iGPU", not ahead of it.

## Stage 0 — measure the iGPU (gate for everything else)

Nothing below is worth planning in detail until this number exists.

- Get a Vulkan-capable llama.cpp. The checkout at `C:/code/llama.cpp` is Aug
  2026 and built CPU-only (no `ggml-vulkan.dll`, no `glslc` on PATH, no Vulkan
  SDK installed — the `vulkan-1.dll` runtime is present). Either install the SDK
  and rebuild with `-DGGML_VULKAN=ON`, or download a prebuilt Vulkan release new
  enough to carry `qwen35moe`.
- Run the 27B GGUF on the 890M. Record: prefill tok/s at 512 and 4096 prompt
  tokens, decode tok/s, peak RSS.
- Quiet box required — covenant's llama-server shares this machine, and
  bandwidth contention already corrupted one round of measurements
  (`docs/npu-prefill.md:60-67`).
- Also confirm the output is coherent, i.e. that interval 3 really is fine in
  llama.cpp and not just nominally supported.

**Decision rule:**

- If iGPU prefill lands in the low hundreds of tok/s, the NPU has to beat that
  with kernels that don't exist yet. Estimate the ceiling from the bandwidth and
  MAC budget before writing any of them.
- If iGPU decode is >= 15 tok/s, the NPU decode path stops being the fast path
  and becomes the reference implementation. That is fine, but it should be said
  out loud rather than discovered later.

## Stage 1 — the seam: phlegm as a ggml backend

Do not transplant llama.cpp code into `npu-engine/`. Register phlegm as a
backend that `ggml_backend_sched` can offload to, and let llama.cpp keep GGUF
loading, the tokenizer, chat templates, sampling, the hybrid KV + recurrent
memory, and the server.

The template is `ggml/src/ggml-hexagon/ggml-hexagon.cpp`, which solves the same
shape of problem for Qualcomm's NPU:

- **Op subset with fallback.** `ggml_backend_hexagon_device_supports_op`
  (line 4062) claims a list of ops; the scheduler routes everything else to the
  other backends. phlegm claims `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID`
  **only when the batch dimension is above a threshold**. That one condition is
  the entire prefill/decode split: prefill batches land on the NPU, decode's
  n_tokens=1 matmuls fall through to Vulkan. No hand-written phase handoff.
- **Weight repack at load.** Hexagon registers a `repack_buffer_type` (line 277)
  whose `set_tensor` rewrites q4_0/q4_1/q8_0 into its tiled layout
  (lines 501-830). q4nx is already a repacked q4_1 with 16-lane row interleave,
  and `tools/q4nx-convert/` already does GGUF Q4_K -> q4nx. Same move, with code
  that mostly exists.

**Why this and not the session-blob design.** The alternative — run the whole
prefill on phlegm's own kernels, then hand llama.cpp a state via
`llama_state_seq_set_data` — requires matching llama.cpp's KV cell layout, its
recurrent-state layout, and its IMROPE convention (`llama-model.cpp:2751`
returns `LLAMA_ROPE_TYPE_IMROPE` for qwen35moe) exactly, then re-verifying that
two independent implementations agree. All that buys is keeping phlegm's fused
whole-layer dispatch during prefill. Not worth it at this stage; revisit only if
Stage 2 shows dispatch overhead actually dominates.

**What the backend seam costs us, honestly:**

- **Fusion is thrown away.** The 1239 ms -> 165 ms decode win came mostly from
  fusing ops into one xclbin context (62 dispatches). Op-granularity offload
  reintroduces per-op dispatch at ~3.3 ms fixed cost. For prefill that amortizes
  — ~300 dispatches for a 30-layer prefill is ~1 s fixed, ~2 ms/token on a
  512-token prompt. For decode it is fatal, which is why the backend must never
  claim decode ops. OllamaAMDNPU is the recorded cautionary tale: naive per-op
  offload, 0.65 tok/s decode (`docs/open-kernels-feasibility.md:60`).
- **Weight duplication.** ggml assigns each weight tensor to exactly one buffer.
  NPU-resident prefill weights plus Vulkan decode weights is ~33 GB of the box's
  88 GB. Tolerable here, ugly in general. A single copy would need host-memory
  import on the Vulkan side (`VK_EXT_external_memory_host`), which ggml's Vulkan
  backend does not expose today. Do not design for that.
- **phlegm's host shrinks a lot.** `server.rs`, `sampler.rs`, `tokenizer.rs` /
  `placeholder_tokenizer.rs`, `q4nx.rs` and the three `generate_*.rs` backends
  all become redundant on the hybrid path. That is the actual repurposing: the
  repo's value narrows to the kernels plus a backend shim, and the rest stays as
  the correctness oracle.

## Stage 2 — batched prefill kernels (the real work, gated on Stage 0)

Net-new designs under `tools/open-kernels/designs/`: a q4 GEMM (as opposed to
today's `gemv_q4`), batched DeltaNet prefill (the chunked recurrent form, not
the per-token update), batched full attention with causal masking, and batched
MoE with per-token expert routing. This is the `docs/npu-prefill.md` Phase 2
scope, now with a concrete consumer.

Only worth starting if Stage 0's iGPU prefill number leaves room, and if the
MAC-budget estimate says XDNA2 can clear it by enough to justify weeks of kernel
work plus 16 GB of duplicated weights.

## Open questions

- Does the MTP layer (`block_count 31`, `nextn_predict_layers 1`) matter here?
  phlegm's converter drops it; llama.cpp has an MTP context type
  (`llama-model.cpp:2270`). Probably orthogonal, worth one check.
- Is `ggml_backend_sched`'s split granularity willing to put a whole prefill
  layer on one backend, or will it ping-pong per op? Measure on a toy backend
  before committing.
- Does the NPU's XRT context survive being driven from inside llama.cpp's
  threading model? phlegm's own note that a context switch resets the cores
  suggests care is needed.
