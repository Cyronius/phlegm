# Port NPU prefill (decode-as-prefill) to L30Backend

**Status:** Phase A DONE + superseded by the RESIDENT path (2026-09-01) — see
the update section at the bottom. Phase B (group-major streaming) is CUT.
**Follows:** [[npu-prefill.md]] (Phase 0/Phase 1, DONE for `L40Backend` only,
2026-09-01) and [[rust-only-open-engine.md]]. This is that same idea applied to
the pruned/interval-3 30-layer schedule — the config that's the actual point of
[[pruned-qwen36-flm-project]], not the base 40L model Phase 1 shipped against.

## Current state
`L30Backend::generate` ([generate_l30.rs:157](../../npu-engine/src/generate_l30.rs#L157))
does CPU prefill today: `l30_buffers::build` calls `forward::run_prefill` (full
numpy-equivalent forward in Rust) to compute each layer's post-prompt state,
writes those to `state_L{l}.bin`, then decode streams from there — one token at
a time, reloading all 30×512MB pools off disk per token (`poolA/B/C` rotation,
3 layers per disk load). Only decode is on the NPU; prefill and the final
logits projection (`model.logits`, called every decode step) are CPU.

**This isn't just slower than it could be — it's on math already flagged
unreliable.** [[flm-capture-oracle]]: `forward.rs` (the CPU model L30's prefill
and lm_head both depend on) is KNOWN DIVERGENT, 0.57-0.68 corr vs verified NPU
paths. Porting L30's prefill to NPU decode-as-prefill isn't only an
NPU-purity/speed goal — it removes a standing correctness risk from every L30
request today.

## Why this is a well-grounded port, not a fresh RE effort
Phase 0 already proved the mechanism on hardware, against **real FLM ground
truth**, on the base 40L (interval-4) model: "sequential decode IS exact
prefill" (bit-for-bit the same final state as batch prefill, because the decode
kernel self-tracks its own KV position from zeroed state with no host input).
`L40Backend` ships this today. L30 uses the **exact same** `layer.xclbin` /
`elf_000005.bin` kernel — M3's finding that it's a "config-agnostic universal
layer executor" (same ELF drives both linear-attention and full-attention
layers) means the interval-3 schedule isn't new kernel territory, just a
different arrangement of the same op. Mechanism risk is low; the open questions
are about L30's specific resource shape, not correctness of the core idea.

## The one real new problem: L30 streams pools, L40 doesn't
`L40Backend` holds all 40×512MB pools resident, so decode-as-prefill's ~92-108
ms/prompt-token cost ([[npu-prefill.md]] Phase 0) is basically free — it's just
more decode steps against buffers already on-device. L30 reloads 30×512MB =
15GB off disk **per token** today (that's inherent to why it streams at all —
15GB doesn't fit resident like L40's ~21GB does at l40's per-layer size, or
rather does fit but the design chose not to hold it — see
[l30_buffers.rs:1-20](../../npu-engine/src/l30_buffers.rs#L1-L20)). A naive port
— just feed prompt tokens through the existing per-token streamed step, same as
decode does — pays that 15GB reload **T times** for a T-token prompt, on top of
whatever it already costs once per generated token.

**Two-phase approach, mirroring how the base project staged itself:**

### Phase A (ship first, days): naive decode-as-prefill, direct port
Swap `l30_buffers::build`'s CPU `forward::run_prefill` call for writing
**zero-filled** `state_L{l}.bin` files (3,145,728 zero bytes each — same size,
same "starts from zero by construction" as `L40Backend::Resident::zero_states`,
just file-backed instead of an in-memory `Bo.init(&[])` since L30's states load
from disk at `load_resident` time). Then, before the existing generation loop
in `L30Backend::generate`, run one `driver.step_bytes(&prog, &act)` per PROMPT
token (not just per generated token), same call the decode loop already makes,
discarding the hidden dump except for the last prompt token. This is a small,
mechanical change — `decode.rs` needs **no changes**: state buffers are loaded
once at `load_resident` and never touched by the streamed program's `load`
lines (those only reload `poolA/B/C`), so they already persist and accumulate
correctly across repeated `step_bytes` calls — that's exactly what sequential
decode-as-prefill needs.

Correct, verifiable, matches L40's proven pattern exactly. Cost is real but
bounded: for prompts the size this project actually exercises (chat prompts,
tens of tokens), T extra streamed decode steps is the same order of magnitude
as generating T extra tokens — not free, but not obviously disqualifying either.
Measure it before deciding Phase B is needed.

### Phase B (follow-on, gated on measurement): group-major streaming prefill
If Phase A's measured prefill cost is bad enough to matter (long prompts
especially), restructure so each layer-group's pool is loaded **once** and run
across **all T prompt tokens** before moving to the next group, instead of
reloading pools per token. This is still exactly decode-as-prefill
mathematically — nothing about batch kernels or new tiling (that's Phase 2 in
[[npu-prefill.md]], a much bigger, still-unstarted project) — just reordering
the loop nest from token-major (all layers, one token, repeat) to group-major
(all T tokens, one layer-group, repeat), which is valid because a layer's
output for token t depends only on the previous layer's output for token t and
that layer's own state from token t-1 — never on another layer's state for a
different token. Cuts prefill I/O from O(T × 15GB) to O(1 × 15GB): the same
single pass the model already pays once per generated token, just amortized
over the whole prompt instead of one token.

This needs new plumbing `decode.rs` doesn't have today: `step_bytes` assumes
one act in, one hidden out; group-major prefill needs a variant that holds T
acts alive across a group boundary and produces T hiddens per group before
advancing. Scope as its own follow-on, not part of Phase A.

## Secondary/stretch scope: real NPU lm_head for L30 (optional, not blocking)
L30's streamed config already invokes the lm_head kernel (`klm`) at **every**
layer-group boundary (`barrier klm logits lmpool act` in
[`build_stream_config`](../../npu-engine/src/generate_l30.rs#L89), ~10 times
per token for 30 layers) — but purely for its cross-context-barrier side
effect; the `logits` BO it writes is never read back
([decode.rs:337-342](../../npu-engine/src/decode.rs#L337-L342) only dumps
`act`/hidden). Real logits still come from CPU `model.logits` — the same
flagged-divergent `forward.rs` math, called every decode step. Reading back
`logits` after the *last* group's barrier (mirroring `L40Backend::step`'s
explicit separate lm_head run) would let L30 drop `model.logits` entirely,
matching L40's "no CPU forward math on the request path" bar
([[npu-prefill.md]] Phase 1). Worth doing, but a separable change from prefill
— track as Phase C, don't block Phase A on it.

## Verification (no FLM oracle exists for interval-3 — this is the real gap)
Phase 0's verification recipe for L40 was "compare against FLM's own captured
prefill boundary states" — that doesn't exist for L30, because **FLM's own
engine NaN-collapses on `full_attention_interval=3`**
([[pruned-qwen36-flm-project]]) — there is no working closed-engine run to
capture. Three fallback layers, weakest to strongest:

1. **Mechanism confidence, inherited, not re-earned.** Same kernel, same
   zero-state sequential-decode mechanism Phase 0 verified byte-exact/
   functionally-correct against real FLM ground truth on interval-4. The
   interval-3 schedule only changes which layer indices are `full_attention`,
   not the kernel or the mechanism.
2. **State-buffer diff against the CURRENT CPU-prefill path** (same method
   [[npu-prefill.md]]'s Phase 0 write-up specifies for L30 explicitly): run
   both on the same prompt, compare per-layer device state dumps. Useful as a
   sanity check and to catch gross bugs (a wrong layer, a byte-order mistake),
   but **do not treat agreement as proof** — the CPU path is the same
   known-divergent `forward.rs` math, so this comparison's ceiling is "as good
   as a compromised reference," not ground truth. Disagreement in the same
   0.6-0.7 ballpark [[npu-prefill.md]] found for L40 vs its CPU model would be
   consistent with "L30 is fine, CPU is still wrong" — not distinguishable
   from "L30 has a new bug" without a third source.
3. **The actual oracle: `tools/golden-ref/reference.py`.** Independent of
   both FLM and our own `forward.rs` — real `transformers` forward on real HF
   weights, already used to prove interval-3's math is sound in the first
   place (finite, well-scaled logits where the closed engine overflows).
   Currently only validates small slices (3LiF/4Li3/5Li3, ≤5 layers) over
   original-model layer indices, not the full 30-layer pruned schedule
   end-to-end. Extending it to the full 30L config (or at least a
   representative deeper slice) is the trustworthy check for this port, at the
   cost of another RunPod GPU session (~24GB CUDA, per the tool's README).

**Recommendation:** ship Phase A behind layer 1+2 (cheap, fast, catches gross
bugs), but don't call L30 NPU-prefill *verified* — as opposed to *plausible* —
without running layer 3 at least once against the full schedule.

## Other gaps to close alongside the port
- **No position-capacity guard.** `L40Backend::generate` rejects
  `prompt_len + max_tokens > MAX_POSITIONS` (1024, the 3MB state buffer's KV
  capacity) before starting
  ([generate_l40.rs:304-310](../../npu-engine/src/generate_l40.rs#L304-L310)).
  `L30Backend::generate` has no equivalent check today — add one; the same 3MB
  state-buffer format applies to L30's `full_attention` layers.
- **Zero-state file writes belong in `l30_buffers.rs`**, replacing the
  `forward::run_prefill` call at
  [l30_buffers.rs:55](../../npu-engine/src/l30_buffers.rs#L55) — but keep a
  function that still does the CPU version (renamed, not deleted) since
  verification layer 2 above needs it as a comparison point.

## Work items, in dependency order
1. Add zero-state buffer writer to `l30_buffers.rs` (or a sibling function);
   keep the existing CPU-prefill path available under a different name for
   verification use.
2. Change `L30Backend::generate` to: build zero-state + prompt-independent
   buffers, open the resident streamed driver, loop `step_bytes` once per
   prompt token (discarding hidden except the last), then continue into the
   existing per-generated-token loop unchanged. Add the `MAX_POSITIONS` guard.
3. Verification layer 2: run both prefill paths on 2-3 known prompts, diff
   per-layer state dumps + first-token argmax, same script shape as
   `compare_flm_boundary.py` used for L40's Phase 0 check.
4. Verification layer 3: extend `tools/golden-ref/reference.py` to the full
   30-layer schedule (or a deeper representative slice), compare logits/argmax
   against the NPU-prefill path's output on the same prompt.
5. Measure Phase A's real per-token prefill cost on hardware (the `l30: prefill
   N tokens in T s` style log line `L40Backend` already prints — add the same
   to `L30Backend`). Decide whether Phase B (group-major streaming) is
   justified from real numbers, not a guess.
6. (Stretch, Phase C) NPU lm_head readback for L30, dropping `model.logits`
   from the request path entirely.

## Risks
- Verification layer 3 (golden-ref) requires spinning up a RunPod GPU session
  again — real cost/time, not just engineering.
- Phase A's prefill cost on L30's streaming design is currently a guess (no
  measurement exists) — could be fine for chat-length prompts and bad enough
  at length to make Phase B non-optional; won't know until step 5 runs.
- If verification layer 2 shows the NPU-prefill and CPU-prefill states
  disagreeing WORSE than L40's CPU-vs-NPU gap, that's a signal of a real L30
  bug, not just "the CPU oracle is bad again" — don't wave it away by
  pattern-matching to the L40 precedent without checking layer 3.

## Update 2026-09-01: Phase A landed, then replaced by going resident

**Phase A (decode-as-prefill on the streamed loop) — done.**
`l30_buffers::build` no longer takes prompt ids or calls
`forward::run_prefill`; it writes zero-filled `state_L*.bin` and only the
prompt-independent weight buffers. The CPU path survives as
`build_cpu_prefill_reference` (verification layer 2 only). `L30Backend::generate`
feeds the prompt through `step_bytes` token-by-token from zero state, samples
the first token through the real sampler (it was always greedy argmax before,
regardless of `GenParams`), and got the `MAX_POSITIONS` guard.

**Then the premise of this whole plan fell over.** The streaming design (and
Phase B) exists because of "30 x 512MB pools = 15GB — cannot stay resident"
(`l30_run_npu.py` docstring, 2026-08-30). Nothing ever measured that. Facts:

- `L40Backend` holds 40 x 512MB = 20GB of pools resident, verified on hardware
  the same day (docs/npu-prefill.md Phase 1). 30 pools is strictly less.
- The machine has ~88GB RAM (OS reports 91.9M KB total, 42GB free at the
  time), not the 48GB the project notes carried.
- `m3out/l30/gen_resident_cfg.py` (2026-08-31) already generated a resident
  30-pool config for the C++ driver — the assumption was half-abandoned
  before the Rust port re-inherited it from `rust-only-open-engine.md`.

Per-token cost of streaming is 15GB of disk reads + H2D per token, for BOTH
prefill and decode. Resident, the 30L model should land near the 40L's
measured 7.4-7.7 tok/s (Python) / ~9 tok/s projected (Rust in-process),
scaled by 30/40 layers — i.e. the "9 tok/s" number the project has been
chasing, and ~2s TTFT for chat-length prompts.

**What changed (resident path):**
- `L40Config::l30()` — `L40Backend` is layer-count agnostic (same fused
  layer kernel; pack/side/pool carry the layer type), so the 30L schedule
  runs on the resident backend unchanged: NPU decode-as-prefill, NPU decode
  (ping-pong runlists), NPU lm_head (full-vocab bf16). That also delivers
  Phase C for free: no `forward::Model::logits` (a 500MB q8 dequant matvec
  on the CPU per token) anywhere on the request path.
- `L40Config::required_files()/missing_files()` + an early, explicit error in
  `Resident::open`; the CLI auto-runs `l30_buffers::build` when files are
  missing (`l30_resident_config_ensuring_buffers`).
- CLI: `l30-run` and `serve --backend l30` are now the resident path;
  `l30-build` writes the buffers; the streamed loop is kept as
  `l30-stream-run` / `--backend l30-stream` (low-memory fallback, second
  implementation for cross-checks).

**Verification status is unchanged** (layers 1-3 above still apply: no FLM
oracle for interval-3). Decode-as-prefill on 30L inherits the 40L mechanism
verification; golden-ref on the full 30L schedule is still the only
trustworthy correctness check and is still not run.

**Result of the first resident runs (2026-09-01, same prompt "why is the sky
blue?", greedy, chat template with thinking):**

| model | pool load | prefill | decode | output |
|---|---|---|---|---|
| `model_30L.q4nx` (30L, interval 3) | 15 s | 71 ms/tok (16 tok in 1.14 s) | **11.6 tok/s** (64 tok) | gibberish |
| `model.q4nx` base 40L (same code path) | ~25 s | 97 ms/tok | 9.1 tok/s (32 tok) | coherent ("The user is asking for the reason why the sky looks blue…") |

Speed goal met: 30L resident is ~1.3x the 40L rate, TTFT ~1.1 s for a
16-token prompt, no CPU math on the request path.

**Correctness: the 30L file we have been testing is NOT Josh's model.**
`model_30L.q4nx` is the raw `slice_keep.py` cut of the BASE 35B model from
2026-08-26 (a structure test for the closed engine: "does it accept 30
layers / interval 3"). Nothing was healed. Josh's published
Cyronius/Qwen3.6-27B-A2.8B removed the same 10 layers **and then LoRA-healed**
(rank 32, ~16M tokens) — and even so reports GSM8K below base and perplexity
"most but not all of the way" recovered. A 25%-layer-removed hybrid MoE with
no healing producing garbage is the expected outcome in ANY correct engine,
so this run is not evidence of an interval-3 engine bug. It is also not
evidence of correctness: the interval-3 schedule still has no oracle.

**What settles it: run Josh's actual weights.** Converter already exists
(`tools/q4nx-convert/convert.py`, Q4_K_M GGUF -> 1.0.2 q4_1 q4nx, tensor
mapping verified at the quant bound against `model_3LiF.q4nx`, never yet run
on the real 27B file). Steps: download `qwen36-27b-a2.8b-mtp-Q4KM.gguf`
(16.5 GB) into `C:/Users/josha/.flm/models/Qwen3.6-27B-A2.8B-open/` (tokenizer
files staged there from the HF cache), convert, `OPEN_QWEN_L30_MODEL=…/model.q4nx
OPEN_QWEN_L30_BUF_DIR=…/bufs open-qwen-npu l30-run`. Coherent output =>
engine handles interval-3 with healed weights (and Josh's model runs at
~11 tok/s on the NPU). Garbage => bisect converter vs engine: `run` (CPU
forward, ~HF-parity) on a `--layers 3` conversion is the cheap first check.

## Update 2026-09-01 (evening): Josh's real 27B runs coherently on the NPU

**Converter had three bugs, all in the linear-attention block** (found by
cosine-matching the converted tensors against the base 35B's corresponding
layers — LoRA-healed weights stay ~0.99 vs their source; see
`tools/q4nx-convert/README.md` "Linear-attention v-head regroup" and
`compare_vs_base.py`): llama.cpp writes v-head axes group-major (g q) where
HF/FLM want (q g) — qkv v-rows, z-gate, ssm_out cols, alpha/beta, ssm_a,
dt.bias, conv1d v-cols were all scrambled (cos 0.06–0.6); `ssm_a` was −exp'd
twice; the MTP block (`blk.30`, with `nextn.*`) was being emitted as a 31st
decoder layer. Fixed; re-converted; validation worst cos 0.9908, no mismatches.
The prune kept base layers 0,2,3 / 4,6,7 / … (2nd DeltaNet of each block dropped).

**Result — `OPEN_QWEN_L30_MODEL=…/Qwen3.6-27B-A2.8B-open/model.q4nx`,
resident backend, greedy, chat template with thinking:**

| prompt | prefill | decode | output |
|---|---|---|---|
| "why is the sky blue?" (16 tok) | 102–159 ms/tok | 6.4–6.6 tok/s | "…The user is asking a simple question: "Why is the sky blue?" I need to provide a clear, concise, and accurate answer. I should explain why the sky appears blue…" |
| "What is the capital of France? Answer briefly." (20 tok) | 102 ms/tok | 7.7 tok/s | "Here's a thinking process: 1. **Analyze User Input:** The user is asking for the "capital" of a "France"…" |

Coherent, deterministic across runs, healthy thinking-mode text. This is the
first time the published interval-3 model has run correctly on the NPU
(FLM's closed engine NaN-collapses on it). Still not "verified" in the plan's
sense — no oracle — but "produces the model's expected behavior" is now a fact.

**Open: throughput on the real model is lower than on the same-shape slice.**
Unhealed `model_30L.q4nx` ran 11.6 tok/s / 71 ms prefill-tok; the healed 27B
runs 6.4–7.7 / 100–160 (both 27B conversions, bad and good, ~7). Same layer
count, same kernels, same buffer builders, same resident path, 40L base does
9.1 on the same day. Not yet understood. Candidates: (a) data-dependent kernel
cost — MoE expert fetch locality (a garbage model may route to few/adjacent
experts), (b) denormal bf16 scale/min values from the q4_1 re-quantization
slowing AIE math, (c) residual machine contention during these runs. Measure
per-layer kernel time with a fixed act vs real acts, and scan the converted
pools for denormal bf16 scales, before guessing further.

**Artifacts:** GGUF + q4nx + tokenizer in `C:/Users/josha/.flm/models/
Qwen3.6-27B-A2.8B-open/`; resident buffers `m3out/l30_27b` (rebuilt in place
from the corrected file). Reclaimable: the sliced test models and the
`bo_*`/`mag_*`/`m0`/`m0b` capture dirs (~6 GB) in `C:/caps` — deletion is
permission-blocked for the agent.
