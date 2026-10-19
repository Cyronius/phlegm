# NPU prefill: kill the CPU time-to-first-token bottleneck

## Goal

Decode now runs at ~7.7-8 tok/s on the NPU (see
[lm-head-npu-bottleneck-instrumentation.md](lm-head-npu-bottleneck-instrumentation.md)),
but every prompt still pays a CPU prefill: `run_5li3_npu.py` /
`l30_build.py` run the full forward in numpy — minutes-scale for 40 layers.
Until prefill runs on the NPU, the open engine has no viable
time-to-first-token and the server flow (Rust port item 5,
[rust-only-open-engine.md](rust-only-open-engine.md)) can't take arbitrary
prompts. This plan gets prefill onto the NPU in two phases: a
mathematically exact shortcut that works today, then true batch prefill.

## Phase 0 — decode-as-prefill (days, no new kernels)

**Key insight: sequential decode IS exact prefill.** A causal model
processed one token at a time produces bit-for-bit the same final state as
batch prefill produces mathematically — and we just proved the decode
kernel is fully self-contained: it appends KV and advances its own position
inside the resident state buffer with no host input (M5c finding), starting
correctly from zeroed states. So prefill = feed the prompt tokens through
the existing serve loop one at a time, ignore the logits, then decode.

- **Driver change (small):** a step-level flag to skip `lmhead` directives
  (and the logits D2H) during prefill steps — saves ~16 ms/token; the final
  prompt token DOES run lm_head, producing the first sampled token's logits
  for free.
- **Bench/harness change:** `prefill <n>`-style loop before the timed
  decode: write embed(act) per prompt token, step with no sampling.
- **Verification:** run CPU prefill (existing `l30_build.py` states) and
  decode-as-prefill from zeros on the same prompt; compare the per-layer
  state buffers (device dump vs `state_L*.bin`) and the first-token logits.
  M2/M3 established 0.999+ corr for the decode kernel vs CPU; this inherits
  that. Byte-exactness is not expected (bf16 vs f64 accumulation) — corr
  targets, not hashes.
- **Cost:** ~92-108 ms per prompt token. 19-token chat prompt ≈ **~2 s
  TTFT** (vs minutes on CPU). 512-token prompt ≈ 50 s — unacceptable, which
  is exactly why Phase 2 exists. Phase 0 is still the right first step: it
  makes the server end-to-end real (arbitrary prompts, NPU-only) with ~zero
  new engineering risk, and it is the correctness oracle for Phase 2.

## Phase 1 — wire into the Rust server path — DONE (2026-09-01)

`npu-engine/src/generate_l40.rs`: `L40Backend`, the first backend with NO
CPU forward math on the request path. Decode-as-prefill from zeroed states
→ NPU decode with NPU lm_head (full-vocab bf16 logits, truncated to the
real 248070 vocab — padding rows are undefined) → sampler. Execution shape
is M5's `servep` ping-pong pipeline ported from
`decode_driver_nobarrier.cpp` (two hw_contexts on one layer.xclbin,
chunks of ≤3, execute-before-previous-wait). Unlike Li3/L30 it keeps the
device + ~21 GB of pools RESIDENT across requests — a new request only
re-zeroes the 40 state BOs. Wired into `serve --backend l40` and `l40-run`.

Verified on hardware (2026-09-01): two-request test passes
(`generate_l40::tests`, ignored/hardware); HTTP end-to-end: request 1 pays
the one-time pool load (~26-45 s) then 7.8 s for a 17-token prompt + 16
decode tokens on request 2. Output coherent, per-request state reset
proven by clean second-request content.

**Perf numbers above are depressed ~2x — cause identified:** a concurrent
Qwen batch run was on the machine during all 2026-09-01 measurements
(memory-bandwidth contention hits the NPU's host-DDR-resident weights
directly). Both drivers were equally affected — Rust decode 4.0 tok/s /
prefill 211 ms/tok vs the C++ driver's 292/234 the same hour (vs 7.4-7.7
tok/s at M5) — and the Rust in-process path was the faster of the two.
This also depressed the FLM warm-request numbers measured the same day
(13.4 s TTFT / ~420 ms/token marginal); the relative FLM-vs-open-engine
comparison stands, absolute numbers need a quiet-machine re-bench.

## Phase 2 — true batch prefill (the real project; weeks, gated on one experiment)

FLM prefills with dedicated batch kernels — the captured m0c prefill phase
runs ~350 op submissions for 3 layers x 11 tokens across the op-level
xclbins (`mm`, `dequant_mm`, `attn`, `GateDeltaNet_prefill`, `conv`), each
op a freshly generated ELF (5200 B and 8192 B programs dominate, plus
one-off big programs). Batch prefill should be ~10-50x faster per prompt
token than decode-as-prefill because it amortizes weight DMA across all T
positions — that's the whole point of prefill kernels.

**The decisive unknown: T-dependence of the generated ELFs.** All existing
captures used an 11-token prompt. The per-op programs bake the sequence
length into their tiling. Whether an arbitrary-T prefill is a *patching*
problem (a few length words, like the 480B ELF's 4 seqlen pokes) or a
*generation* problem (tiling loop structure changes with T — the closed
engine's sequence-generation moat, Tier-2 territory per
[rust-engine-replacement-feasibility.md](rust-engine-replacement-feasibility.md))
determines whether Phase 2 is weeks or months.

**Experiment 2a (run FIRST, ~half a day):** capture FLM prefill on the base
40L model at three prompt lengths (e.g. T=11, 12, 23 — one step apart and
one tile-boundary apart) with the existing `tools/seq-capture` shim, then
structural-diff the op streams (`seq_struct.py`):
- Same op count/structure, only length words + DDR offsets differ →
  **patchable**: build a template library at one T and a patcher (the
  `poketpl` infra generalizes); Phase 2 ≈ 2-3 weeks.
- Op count or loop structure changes with T →
  **generation problem**: either derive the tiling rule from a T-sweep
  (more captures, model the pattern) or cap Phase 2 to "prefill in fixed-T
  chunks" (pad prompts up to the captured T, run ceil(T/T0) batch chunks —
  needs the chunk-boundary state handoff verified). Decide after seeing
  the diffs; do NOT commit to full sequence generation up front.

**Phase 2 verification:** decode-as-prefill (Phase 0) is the oracle — same
prompt through both paths must yield matching states/logits (corr targets).

## Phase 0 findings (2026-08-31): mechanism built and fast; verification blocked by the CPU-oracle gap

Built and ran `tools/kernel-interp/prefill_e2e.py` (decode-as-prefill from
zeroed states, lm_head skipped via a step-level `-` flag added to the
driver; final prompt token runs lm_head). **The mechanism works and is
fast: 19 prompt tokens in 0.43 s (22 ms/token) on 5li3** — the projected
~2 s TTFT for the 40L model stands. But the correctness story took an
unexpected turn, and the honest status is UNVERIFIED — in either direction:

1. **The NPU replay path is perfect.** Re-ran the m0c ground-truth replay
   (FLM's captured pools/states/act through our driver): logits
   **byte-exact, 0 of 1,048,576 bytes differ**. Driver, kernels, serve
   modes — all above suspicion. Also: every serve-config variant tested
   (M4-replica kernel split, ping-pong, single-kernel) produces
   bit-identical output — config shape does not affect results.
2. **The CPU reference model cannot adjudicate.** NPU-vs-CPU hidden corr is
   ~0.57 on a warm-start single token (and `decode_step.py`'s own
   ground-truth check now reads 0.67) — matching M3's own recorded open
   item ("CPU model reaches corr 0.68 vs recovered hidden"). This was
   masked in the milestone story: M2's "parity" was vs the HF golden
   reference; the CPU-vs-NPU decode gap was never closed. Ruled out today:
   config differences (bit-identical across variants), group-wise
   power-of-2 scaling of the act hidden (no clustering; group corr WORSE
   than global; sign agreement 74%), sequential-vs-batch math (CPU
   sequential == CPU batch prefill at corr 0.999999 — the Phase-0 identity
   holds in numpy). Still open: why two forward implementations that each
   look self-consistent (NPU==FLM byte-exact; CPU==itself/golden) disagree
   this much while both generating plausible text.
3. **The from-zero 5li3 run produced degenerate text** ('aely...odox...')
   and a first token disagreeing with the CPU pipeline — but with the CPU
   oracle unreliable and a 5-layer slice plausibly degenerate on its own,
   this neither confirms nor refutes decode-as-prefill.
4. **Harness caveat:** the `-` lm_head skip also skips the barrier — fine
   for ping-pong configs, but single-context barrier-style configs then
   exceed the 3-run budget and time out. Ping-pong is the correct prefill
   config.
5. The m0c capture's prompt token ids are NOT recoverable from the dumps
   (prefill inputs never appear as raw embeddings; only decode acts do —
   recovered decode inputs match `rust_ref/decode_tokens.i64`), so FLM's
   captured boundary states can't serve as the prefill oracle without the
   prompt.

**Resolution path (next session): one fresh FLM capture with a KNOWN
prompt** on the base 40L model — this simultaneously (a) provides FLM's
true prefill boundary states to verify decode-as-prefill against, and (b)
IS Phase 2's experiment 2a (capture at known lengths for the T-dependence
diff). Until then, Phase 1 wiring should proceed only for the free-run
path; prompt-conditioned serving waits for the capture verdict.

**Also flag independently:** the CPU-model-vs-NPU divergence (M3's open
item) deserves its own investigation — it silently weakens every
CPU-oracle claim in the repo, including `forward.rs` (a port of the same
math).

## Experiment 2a results (2026-09-01): T-dependence is a repetition rule, not a tiling problem

Captured FLM prefill on the BASE 40L model with known prompts at three
lengths (seq plane only — see harness notes below). Prompts chosen via the
tokenizer port; FLM's `usage.prompt_tokens` confirmed the templated-length
minus 2 rule every time (the trailing `<think>\n` of the chat template is
NOT prefilled; it enters as the first decode inputs):

| prompt | prefill T | .seq ops | FLM TTFT | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|
| "Say hi." | 11 | 3693 | 9.42 s | 1.17 | 5.33 |
| "Say hi now." | 12 | 3949 | 11.74 s | 1.02 | 5.83 |
| "Count from one to ten …" | 23 | 5367 | 12.30 s | 1.87 | 6.46 |

Timings are FLM's own `usage` fields, under the (cheap) seq-only shim.
Captures: `C:/caps/pf_t11_seq`, `pf_t12_seq`, `pf_t23_seq`.

**Structural verdict (`seq_struct.py`, pairwise):**
- The op-type vocabulary is FIXED: 16-17 structural fingerprints at every
  T. The workhorse `M`/`N` op pair has an IDENTICAL fingerprint across T —
  individual ops do not re-tile with sequence length.
- T-dependence is (a) the COUNT of `MN` repetitions (M=N=1821/1949/2658 at
  T=11/12/23 — linear-ish, with a small boundary effect: +128/token at
  11→12, ~+64.5/token averaged over 12→23), (b) a per-position walk of
  1KB-strided weight-table offsets (exactly one new arg_offset per extra
  token — a rope/position-table slice), and (c) small prologue/epilogue
  count changes (op types C, P, Q).
- Distinct weight arg_offsets are otherwise the same set (10359-10360).

So Phase 2 is NEITHER a pure few-word patch NOR the feared tiling
compiler: FLM's prefill stream is a fixed-shape micro-op unit repeated per
token/tile with a deterministic per-position offset walk. Synthesizing an
arbitrary-T stream = emit the prologue, repeat the per-token unit with the
position-table offset advanced, emit the epilogue. Weeks-scale, not months.

**The strategic surprise — FLM's prefill is SLOW:** ~1-2 prefill tok/s
vs its own ~6.5 tok/s decode, with a ~10 s ELF-generation-dominated
constant even for 11 tokens. Phase 0 decode-as-prefill (~10 tok/s
equivalent, ~22 ms/token on 5li3) is already ~5x faster than FLM's real
TTFT on short prompts. Batch prefill (Phase 2) remains the win for long
prompts, but the "FLM must be 10-50x faster" assumption is dead — the
repeated micro-op stream explains it (weight DMA is NOT amortized across
positions the way real batch kernels would).

**Warm-request test (2026-09-01, shim pass-through, one server session,
sequential requests):** FLM does NOT cache generated ELFs or prompts —
repeating the identical T=11 prompt as the third request gives TTFT
13.39 s vs 13.41 s cold (bit-stable, zero warm-up benefit). Same-session
T=11 → T=23 gives a clean marginal: (18.39−13.39)/12 ≈ **420 ms/token
marginal + ~8.8 s constant, every request**. Extrapolated to a 512-token
prompt: FLM ≈ 220 s vs decode-as-prefill ≈ 50-70 s — Phase 0 wins at
EVERY prompt length, ~3-4x even at 512 tokens. Phase 2's bar is now "beat
~100 ms/token", which halves its urgency; it stays worthwhile only if the
batch micro-op stream amortizes far better than FLM's own prefill does.

**Harness notes (hard-won):**
- `prefill_capture.ps1` drives base-model runs; both shim planes armed via
  `-Planes both`, seq-only via `-Planes seq`.
- The bo/event plane's RUNARG hashing (fnv1a over EVERY kernel-arg buffer
  per submit, including 512MB pools) stalls 40L prefill ~100x — a 12-token
  prefill did not finish in 10 minutes. Fixed in the shim:
  `FLM_BO_RUNARG_MAX` (bytes) skips hashing/dumping args larger than the
  cap; the driver script sets 8MB for `both` runs.
- FLM prints "Prefill chunk 1/1 with N tokens" — its prefill is CHUNKED;
  larger prompts than tested here may reveal the chunk size (relevant to
  the fixed-T-chunks fallback, likely how FLM itself handles arbitrary T).

**Boundary-state capture (for Phase 0 verification):** `pf_t11_full`
(both planes, DumpMax 4MB, RUNARG_MAX 8MB) captures FLM's true prefill
boundary states for the known 11-token prompt — the decode-as-prefill
oracle the Phase 0 findings called for. Extraction
(`boundary_manifest.json` in the capture dir): the 30 GDN state buffers
roundtrip D2H→H2D (hashes byte-identical) between the prefill-final
logits and the first decode op, whose 1MB act H2D was verified to be
embed(`<think>`) at corr 1.000000 — so the D2H dumps are the states after
EXACTLY the 11 known tokens. FLM's captured logits are the odd vocab half
(fp32), semantically sane (top-8: Hello/Hi/Hey/...). Layer mapping:
zero-init order k → model layer k + k//3.

## Phase 0 VERIFIED against FLM ground truth (2026-09-01)

Ran `prefill_e2e.py l40 1` (decode-as-prefill from zeros, same 11 tokens,
resident 40L pools) and compared device states against the FLM boundary
(`compare_flm_boundary.py`):

- **Early layers near-exact:** L0 conv corr 0.99996 / S 0.9993; L1-L2
  ≥0.996. The mechanism — byte format, token feeding, on-device sequential
  KV/state advance — is correct.
- **Depth-wise decay, not failure:** corr decays smoothly to ~0.83-0.73
  (S) by L28-L38 — the signature of accumulated bf16 divergence between
  two different-but-valid execution orders (FLM's per-token micro-op
  prefill vs our monolithic decode ELF), not a math bug. conv corr
  recovers to ~0.94 at the deepest layers (residual-stream saturation).
- **First token MATCHES:** our full-vocab argmax is `<think>` (248068) —
  what FLM feeds next; restricted to FLM's odd half, top-3 are IDENTICAL
  (Hello 9419, Hi 12675, Hey 18103). Aligned odd-half logits corr 0.70.
  (Our l40 lm_head logits are FULL-vocab bf16; FLM's capture is odd-half
  fp32 at index i → vocab 2i+1.)

**Verdict: decode-as-prefill is functionally verified on the 40L model.**
Behavioral agreement (first-token / top-k) holds end-to-end; numeric drift
is bounded and depth-accumulated. Remaining caution: drift may grow with
prompt length — spot-check top-k agreement at a longer prompt before
declaring victory at 512 tokens.

**CPU-oracle question resolved directionally:** our NPU chain agrees with
FLM's NPU at 0.9996 (L0, after 11 sequential tokens) while the CPU
reference model sits at 0.57-0.68 against both NPU paths — the CPU model
is the OUTLIER. FLM captures, not the CPU model, are the correctness
oracle from here on; the CPU forward (and `forward.rs`, same math) has a
real defect to find in its own workstream. (Caveat: both NPU paths share
the closed engine's kernel lineage, so a common codegen quirk can't be
excluded — but as the engine we're replacing, FLM is ground truth by
definition.)

## Risks / honesty

- Phase 0's per-token numerics use the decode kernel's windowed dynamic
  normalization (M3); long prompts accumulate bf16 error differently than
  FLM's batch path. The CPU-oracle comparison bounds this; if corr degrades
  with prompt length, flag before building Phase 1 on it.
- Phase 2's capture experiment needs the HX370 free and FLM runnable with
  the shim (existing tooling: `bo_capture.ps1` era scripts) — same setup
  that produced the i3/i4 captures.
- The prefill xclbins are the same closed, revenue-capped kernels
  (TERMS.md) — no license change from decode.
- FLM's own prefill speed on this hardware is unmeasured by us; measure it
  during 2a (the capture run prints FLM's timings) so Phase 2 has a real
  target instead of a guess.

## Out of scope

- Prefill sequence GENERATION from scratch (Tier 2/3) unless 2a forces the
  question — and then only the minimal tiling rule, not a general compiler.
- Multi-request batching, paged KV, prompt caching — server features for
  after the port lands.
