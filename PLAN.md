# Phlegm integration and kernel work plan

Updated: 2026-10-26. Status: Linux NPU development in progress; current evidence and remaining gates live in `.plans/`.

This is the project handoff plan for continuing on another machine. Paths are
relative to the repository root. Check the current checkout and later result
entries before resuming: several older plan headers predate completed work.

## Direction and priority

Keep C++ as the inference engine while completing normal generation, model
onboarding and the tray. The native Rust inference engine port comes last.
The CLI/server now use the resident C++ process adapter for NPU inference.
The next milestone is a real hardware run with a compatible combined kernel set,
followed by fixing the resulting integration or performance failures. Hardware
validation of the combined kernel tree needs a compatible C++ runtime.

The implementation order above follows the 2026-10-22 override in
[the monorepo plan](.claude/plans/phlegm-monorepo.md). Its older P1/P2/P3 order
does not override that decision. The kernel items below are separate workstreams;
their order after combined-tree validation is a proposed priority, not a claim
that they all block CLI/server integration.

## Starting point

- The C++ reference benchmark adapter runs real models. `phlegm run` and `serve`
  now use it for NPU inference; `serve --mock` remains an explicit test path.
  The tray currently provides its shell. Device inference with this Phlegm
  revision has not yet been verified.
- The 2026-10-23 kernel sync and worktree recovery are recorded in
  [UPSTREAM.md](open_kernels/UPSTREAM.md). The audited kernel imports are present.
- That ledger records an offline suite result of 1793 passed, 58 skipped and
  8 deselected. This is historical evidence, not a check performed by this plan.
- No fresh NPU build or hardware acceptance has been established for the combined
  tree. Earlier measurements remain tied to their original revisions/artifacts.
- C++ runtime companion patches are retained in
  [open_kernels/upstream-patches](open_kernels/upstream-patches/). They are separate
  diffs against recorded source heads, not a combined runtime or a patch series.

## 1. Connect CLI and server generation to C++

- [x] Extend the resident process adapter with prefill/decode logits for normal generation.
- [x] Connect `phlegm run` and `phlegm serve` to that adapter.
- [x] Per-model runtime defaults (sampling, thinking, context, prefill) in OpenFlowLM's
  `model_list.json` + user `settings.json` format, with `phlegm config` (SERVER-SETTINGS).
- [x] OpenAI `reasoning_effort` translated to each template's own switch (SERVER-REASONING-EFFORT); the
  bridge dies with phlegm (PR_SET_PDEATHSIG).
- [x] Long prompts (2026-11-04, `.plans/long-prefill.md`): the block routes' host attention vectorised and
  tiled (bit-identical), 16 bridge threads by default, block prefill for Hy-Memory and Flint. Through
  `phlegm serve` an ~11.5k-token prompt now answers on LFM3 (48 s), LFM2.5 (231 s), Flint (236 s) and
  Hy-Memory (204 s); a ~2.9k-token prompt takes 10-50 s instead of 32-657 s.
- [x] Prompt reuse (2026-11-05, SERVER-PROMPT-REUSE): the bridge keeps its context between requests and a
  turn prefills only what it adds (recurrent state saved at a block-aligned mark where the conversation ends);
  bit-identical to the same prefills from a reset on all seven model families; a 1800-token second turn
  sharing 1500 tokens prefills in 2.3-9.3 s instead of 6.4-26 s.
- [x] Sample in Rust, load external/inline model chat templates, and derive EOS IDs.
- [x] Stream server output, reset for each request, and bound generation by context capacity.
- [x] Surface load/worker errors without silently using a placeholder tokenizer.
- [ ] Verify actual OpenAI responses, finish reasons, streaming, and NPU
  hold/release on a device with matching model and kernels.

Completion: real prompts work through both CLI and server, including sampled,
streamed and multi-turn requests; the mock backend is no longer the normal
inference path. Preserve C++ as the reference for the later Rust port.

Sources: [current direction](.claude/plans/phlegm-monorepo.md),
[adapter build/protocol notes](benchmarks/reference/README.md),
[project status](README.md).

## 2. Validate the combined kernels and runtime

2026-10-24 host inventory: see [baseline readiness](specs/open-engine/artlite-baseline-readiness.md).
The current boot initializes the Strix Halo driver with IOMMU. Local Linux
tools, XRT/plugin, the C++ bridge, an isolated ArtLite comparator and fresh Llama
1B kernels are built, and Q4_1 weights are available. An ArtLite arithmetic NPU
smoke passes. The authenticated memlock adjustment now permits Phlegm XRT
enumeration, resident model inference and reset. Experimental GGUF block
prefill and an explicit Q8 head pass a 300-token quality gate; the artificial
4096-window gate still fails. See
[Linux GGUF prefill](specs/open-engine/gguf-prefill-linux.md) for fixes and
diagnostic survey progress. These routes remain opt-in.
The measured development baseline supports versions 1–3 and local DIT7;
a separate PR209 runtime candidate supports1–7 while preserving DIT7 and
Llama hi/lo precision. Bonsai/Hadamard integration and broad family acceptance
remain open. Real-prefix GGUF quality and active4k flash checks pass; diagnostic
the corrected probability-hi/lo flash route reaches770/714/585 tokens/s at
1k/4k/8k, still far behind ArtLite. All9 diagnostic pairs pass controls; the
artificial zero-history failure remains, so the route stays opt-in. Decode
compute/DMA and core-layout experiments are next. Every accepted parity cell remains
open; these development measurements are not clean-release benchmark results.

- [ ] Prepare an external C++ reference checkout at a recorded revision.
- [ ] Reconcile the required companion changes against that checkout. Use
  [sources.json](open_kernels/upstream-patches/sources.json) for patch bases.
- [ ] Ensure the runtime implements every selected manifest feature, including
  the local version-7 `bfp16_dit` mapping.
- [ ] Preserve the paired precision/state/KV contracts and Bonsai/Hadamard ABI
  choices documented in the sync ledger. Llama batched attention requires the
  companion high/low softmax-probability fix.
- [ ] Build fresh kernel sets for the selected validation models and run the
  applicable recipe/manifest, numerical, full-model and serving checks.
- [ ] Record source revisions, build settings, artifact hashes, model identity
  and hardware conditions with the results. Report untested routes explicitly.

Completion: the selected combined kernel/runtime routes pass their documented
acceptance checks on hardware. Offline success alone does not close this task.

Source: [sync ledger and recovery notes](open_kernels/UPSTREAM.md).

## 3. Finish model and kernel onboarding

- [ ] Derive `ModelSpec` and `spec_hash` using the Python recipes.
- [ ] Find matching local kernels and fetch published sets through an index
  mapping model specs to build keys, artifact URLs and checksums.
- [ ] Complete model download and loading flows against the C++ engine.
- [ ] Add kernel list/fetch/build commands and clear progress/error reporting.
- [ ] Follow with direct GGUF integration where needed, local builds on a cache
  miss, and validation against the family's reference before accepting a new
  parameter point.

Completion: a supported model with a published matching set can be added and
run without manual kernel installation. Local compilation and self-validation
are later onboarding milestones.

Source: [monorepo plan, P4](.claude/plans/phlegm-monorepo.md).

## 4. Complete tray and visualization integration

- [ ] Implement the tray's model library, download progress, deletion and
  load/unload flows, with one model on the NPU at a time.
- [ ] Add chat, an OpenAI server toggle, logs and start-on-login behavior.
- [ ] Run inference in a managed `phlegm serve` child process so an NPU fault
  can be recovered by restarting that process.
- [ ] Expose the existing decode visualization through the Phlegm CLI.
- [ ] Later add prefill visualization and optional measured timing traces.

Completion: the tray can add/load a model, chat and manage its inference server;
the CLI can open a model's decode visualization. Track prefill and measured
timing as separate follow-ups.

Sources: [monorepo plan, P5](.claude/plans/phlegm-monorepo.md),
[visualization plan](specs/viz/plans/flow-viz.md).

## GPT-OSS 20B

Runs on the NPU from ggml-org's `gpt-oss-20b-MXFP4.gguf` through `recipes/gptoss.py` and the
open runtime (`upstream-patches/gptoss-runtime.patch` + `gptoss-prefill-runtime.patch`):
`phlegm run` and `phlegm serve` answer through the bridge, with the harmony analysis apart
from the answer. Decode streams the experts as the GGUF's own MXFP4 (`gemv_mx4.h`, decoded on
the core into the q8 route's exact codes: byte-identical logits, `gx_moe` 4.06 -> 2.49 ms a
layer, 12.5 GiB of pools instead of ~26). Prompts prefill on a block route (OPEN-PREFILL-GPTOSS):
q | k | v and o as bfp16 GEMMs, each expert's GEMMs reading MXFP4 natively (decoded to bfp16 in
L1), every operand hi + lo to ~16 bits, the rest on the host in fp32. 663 tokens in 2.9-3.3 s
(94 s one token a step), 2000 tokens in ~10 s, 16 tokens in 1.0 s; accuracy against an fp32
reference at least the sequential route's; the chat prompt's greedy tokens equal llama.cpp's.
Development timing on a loaded machine. Open: a clean benchmark, NPU attention products for
long prompts, one shared copy of the experts (the prefill GEMM keeps its own 10.8 GiB), the
Safeguard variant. Plan: [.plans/gptoss-native.md](.plans/gptoss-native.md); spec:
OPEN-FAMILY-GPTOSS, OPEN-PREFILL-GPTOSS.

## Recursive Flint (40B-A2.5B MoE)

Runs on the NPU from RecursiveSI's `flint-int4-q4_1.gguf` through its own recipe `recipes/flint.py`
(`designs/flint/fl.py`: q4_1 GEMVs, Qwen3 q / k norms and GPT-OSS's learned sink in one attention
program -- NoPE full layers as an identity position table, the 513-row window on the sliding ones --
and an 8-slot expert dispatch: 6 sigmoid-routed experts plus the ungated shared expert as two
slots) and `upstream-patches/flint-runtime.patch` (`fl_q4` pack op, `flint_route` host step). The
GGUF's Q4_1 scales are bf16 values, so the pools are exact. `phlegm run` / `serve` answer (the
`<think>` split, SERVER-REASONING, is not checked live on this model). Greedy tokens equal llama.cpp's; an 815-token prompt
sits at the bf16 rounding control's distance from the fp32 reference at every position. Decode
~190-230 ms a token. Block prefill (2026-11-04, the default; `FLINT_BLOCK=0` for sequential): LFM2.5's route --
q4 GEMMs, host attention with the sink, the window and NoPE (runtime kind `flint`,
`upstream-patches/flint-block-runtime.patch`), the experts and the shared expert through LFM2.5's `mbl` reading
the pools in place -- prefills the 815-token prompt in ~14 s against ~163-178 s sequential (development, loaded
host); through `phlegm serve` (bridge `bridges/flint-blk`) a 2,891-token prompt in 70 s against 620 s. Quality
is the sequential route's class (per-position KL median 9.5e-3 vs 9.1e-3 sequential, 8.0e-3 bf16 control) with a
heavier tail; 29 / 32 greedy tokens of the long prompt equal llama.cpp's (a near-tie flip). Open: the host
attention past a few thousand tokens, the tail error, one context per layer, a live check of the reasoning
split, a clean benchmark. Plan: [.plans/flint-native.md](.plans/flint-native.md); spec: OPEN-FAMILY-FLINT.

## Tencent Hy-Memory-30B-A3B (hy_v3 MoE)

Runs on the NPU from our own pure Q4_0 GGUF (tencent's BF16 safetensors through llama.cpp's converter
and `llama quantize --pure`, Q8_0 head; no published file packs exactly) through its own recipe
`recipes/hyv3.py` (`designs/hyv3/hy.py`: Flint's attention program without the sink, NEOX RoPE on every
layer; a 9-slot expert dispatch whose 768-wide experts -- the stored width, not the config's 1536 -- run
gate | up in slot pairs over 4 + 4 cores; the dense layer 0 as nine 768-wide slices of the same program)
and `upstream-patches/hyv3-runtime.patch` (`hy_route`: sigmoid router, selection-only expert bias,
renormalised x 2.826, the ungated shared expert at weight 1). Every Q4_0 scale is a bf16 value, so the
pools are exact. A chat prompt's greedy tokens equal llama.cpp's up to the EOS both rank first; an
836-token prompt sits at the bf16 control's distance from the fp32 reference at every position, and 32/33
teacher-forced continuation positions agree with llama.cpp. Sequential prefill ~226 ms a token, decode
~220-240 ms a token (development, loaded machine). `phlegm run` / `serve` answer. Block prefill
(2026-11-04, the default; `HYV3_BLOCK=0` for sequential): LFM2.5's route over the same pool -- q | k | v and o
as q4 GEMMs, host attention and norms, every FFN (routed, the shared expert at weight 1, the dense layer's
slices) through LFM2.5's `mbl` expert kernel unchanged, since Hy's pool already has its byte law; runtime
`hyv3-block-runtime.patch` (route `shared` / `dense`). The 836-token prompt in 17.6-17.9 s against 145-186 s
sequential (development, loaded), every position inside the bf16 control's spread, short-prompt and
256 / 257 / 300-token greedy tokens equal to sequential. The model dir's kernels and `phlegm` use it (bridge
`bridges/hyv3-blk`). Open: NPU attention for the block (host attention dominates long prompts), one context
per layer, a clean benchmark. Plan: [.plans/hymemory-native.md](.plans/hymemory-native.md);
spec: OPEN-FAMILY-HYV3.

## LiquidAI LFM2.5-24B-A2B (lfm2_moe)

Runs on the NPU from our own pure Q4_0 GGUF (LiquidAI's BF16 GGUF through `llama quantize --pure`,
Q8_0 tied embedding; no published file packs exactly) through its own recipe `recipes/lfm2moe.py`
(`designs/lfm2moe/lm.py`: LFM2's short conv in fl.py's fabric with f32 taps, HY V3's attention at
head dim 64, Flint's expert program at 1536 = 3 bands a core, the dense MLP as 23 slots of 512) and
HY V3's `hy_route` host step, unchanged (no runtime patch of its own). 51 greedy tokens of a chat
prompt and all 32 after an 843-token prompt equal llama.cpp's; every prompt position sits inside the
bf16 control's / llama.cpp's distance from the fp32 reference. Decode ~183 ms a token, sequential
prefill ~151 ms a token (development, loaded machine). `phlegm run` / `serve` answer. Open: block
prefill, the reasoning split, the QAD-Q4_0 checkpoint (a Q6_K head), a clean benchmark. Block prefill
(2026-11-03, token-batched `mbl` experts + q4 GEMMs, host elementwise middles and attention): the
843-token prompt in 7.8 s against 105-116 s sequential (development, loaded), 32/32 greedy identical, every position inside
the bf16 control's spread. The default (`LFM2MOE_BLOCK=0` for sequential): the model dir's kernels and `phlegm run` use it
(bridge `bridges/lfm2moe-blk`). Plan:
[.plans/lfm2moe-native.md](.plans/lfm2moe-native.md); spec: OPEN-FAMILY-LFM2MOE.

## NVIDIA Nemotron 3.5 Lightning 30B-A3B (nemotron_h MoE)

Runs on the NPU from our own pure Q4_0 GGUF (NVIDIA's BF16 safetensors through llama.cpp's converter, `--no-mtp`,
and `llama quantize --pure`, Q8_0 head; every scale bf16-exact) through its own recipe `recipes/nemotronh.py`: the
NPU streams the weights (q4 GEMVs for the Mamba in / out and q | k | v / o projections, an 8-slot expert program for
the six routed experts and the shared expert's two halves, the Q8_0 head) and the host runs the rest between the
dispatches (the norms, Mamba-2's conv and selective-state update, attention without positional encoding, the
router, the residual) as `nh_*` program steps of its own runtime patch (`nemotronh-runtime.patch`, over the
gemma4-moe chain). Every block dispatch is a stream of one image (the expert program with runtime-parameter loop
counts also runs the GEMVs), so only the head changes context. 16/16 greedy tokens of a raw prompt and 32/32 after an
841-token chat prompt equal llama.cpp's (a chat prompt's 64: the first 46, then a near-tie llama.cpp resolves
differently itself); every position of the 841-token prompt sits at the bf16 control's distance from the fp32
reference (827/841 argmax, KL median 1.2e-3; control 825/841, 9.0e-4), and the reference agrees with llama.cpp CPU
(810/841, partial KL median 6.6e-4). Decode ~103 ms a token, sequential prefill ~109 ms a token (development, loaded
machine; 145-174 ms on the first one-image-per-projection route). `phlegm run` / `serve` answer (the model dir's
template tolerates phlegm's `tools` = none). Open: block prefill, the head on the block image, overlapping the host
steps, a clean benchmark, a llama.cpp Vulkan control. Plan: [.plans/nemotronh.md](.plans/nemotronh.md); spec:
OPEN-FAMILY-NEMOTRONH.

## LiquidAI LFM3-1.7B (lfm3)

Runs on the NPU from LiquidAI's QAD-Q4_0 GGUF (derived: Q8_0 head, per-head gate rows repeated to a q-width
projection) through its own recipe `recipes/lfm3.py` (`designs/lfm3/lg.py`: the LFM2 hybrid with a 7-tap conv, sandwich norms,
gated attention in two kinds -- full with a half-rotated head, sliding 256 -- and a dense MLP on every layer; two dispatches a
layer) on the existing LFM2.5 runtime, no patch of its own. 16/64/32 greedy tokens of a raw, a chat and an 843-token prompt
equal llama.cpp's; every prompt position sits at the bf16 control's distance from the fp32 reference (835/843 argmax-equal,
KL median 8.7e-5). `phlegm run` / `serve` answer. Decode ~100 ms a token (dispatch-bound: 48 dispatches and context
switches; fusion runs one dispatch a layer: 49 ms a token, bit-identical logits; the shared image puts conv and attention in one xclbin with two streams: ~32-38 ms a token at load ~30, bit-identical logits; Q4_0-native pools and bf16 head scales, exact, ~37 -> 34.6 ms at load ~16), sequential prefill 60-102 ms a token. Block prefill (2026-11-03, the LFM2.5 `lfm` route generalised:
q4 GEMMs on the decode pool's bytes in place, host conv / attention middles, sandwich norms): the 843-token prompt in 2.6 s against
50 s sequential (development, loaded), 32/32 greedy equal to llama.cpp, mean KL 2.46e-4 against 2.43e-4 sequential and 2.42e-4 for the
bf16 control. Every one of these is the default (2026-11-03; `LFM3_FUSED` / `LFM3_SHARED` / `LFM3_Q40` / `LFM3_BLOCK=0`,
`LFM3_HEAD=f32` select the slower routes): the model dir's kernels are the `lfm3-q40h-blk` set and `phlegm run` uses block prefill
through `bridges/lfm3`. Open: the head's context switches (~2 ms a token) and a Q6_K-native head, the 16-row gate, a quiet-machine timing, attention products on the NPU for prefill,
a clean benchmark. Plan:
[.plans/lfm3-native.md](.plans/lfm3-native.md); spec: OPEN-FAMILY-LFM3.

## Gemma 4 26B-A4B (MoE)

Runs on the NPU from Google's QAT Q4_0 GGUF through `recipes/gemma4.py` (the 12B's dx.py
attention and dense MLP, with `designs/gemma4_moe/gm.py`'s expert cores in the same image as a
second stream, `dxm.py`; the router and FFN close on the host) and the open runtime
(`upstream-patches/gemma4-moe-runtime.patch` + `gemma4-moe-decode-runtime.patch`). On a
26- and a 1450-token prompt, the 32 greedy tokens equal llama.cpp's on the same GGUF. Every
logit's KL against an fp32 reference matches a bf16-rounding control. `phlegm run` and
`serve` answer through the bridge. Development timing (2026-11-01, paired against the 10-31
build): decode step 0.80x (106 against 132 ms median at load 8-20; 82 ms a token on a quieter
run), sequential prefill 88 ms a token, logits byte-identical. Phlegm returns Gemma's thought
channel as `reasoning_content` (SERVER-REASONING). Block prefill (2026-11-03, the default since then: the model dir's kernels and `phlegm run` through `bridges/gemma4moe-blk`; `G4_MOE_BATCH=0` drops it;
`gemma4-moe-block-runtime.patch`, `designs/gemma4_moe_batch`): 1450 tokens in 13-15 ms a token against
74-99 sequential (paired, alternating, load 20-30; development timing); short prompt 32/32 equal to
llama.cpp, long prompt 17/32 then a 0.374 / 0.371 near-tie flip, prompt log-KL within the bf16
control (mean 0.064 against 0.065) but 147 flips against the sequential route's 133. Teacher-forced decode over the same prompt: KL mean 9.0e-4 against the bf16 control's 1.0e-3, max 8.6e-3, one
flip (the control's one). Open: the experts' weights are rounded to bf16 on the core (a hi/lo split would
remove that); bfp16 dense GEMMs; the ten local / global context changes a token; a clean benchmark. Plan:
[.plans/gemma4-moe.md](.plans/gemma4-moe.md); spec `OPEN-FAMILY-GEMMA4-MOE`.

## DiffuSpace Clover-35B-A3B-Preview (block diffusion on Qwen3.6-35B-A3B)

Autoregressive Clover runs on the NPU (2026-11-05): `/home/cyrus/models/Clover-35B-A3B-Preview-Q4pure-NPU2`,
a pure Q4_0 GGUF built from the BF16 weights (@ `4b0402b6`) by the Ornith / Nex recipe, on KAT's kernels
unchanged (Nex's spec). The converter and the engine read a mapped config (`qwen3_5_moe_text`); the 120
preview-only tensors are dropped. Text RoPE is the same as Clover's plain RoPE (measured: llama.cpp gives
bit-identical logits for the same tensors under either label). Pack check ALL OK (Q4_0 rel L2 0). On the
491-token prompt the NPU agrees with llama.cpp CPU at 464/491 (block, median KL 4.9e-3), the CPU-vs-Vulkan
control at 462/491. `phlegm run` and `serve` answer, with the thinking split. Development timing, paired
with Nex: decode 66 ms a token at ~500 tokens (15 tok/s) and 76 ms at ~3000; prefill 3.0 s / 13.5 s.
Registry: `clover:35b-a3b-q4pure` with Clover's recommended sampling (greedy; on the reply only: presence 0.9,
repetition 1.05; +10 ramps on the stop tokens from 8192 reply tokens and on `</think>` from 4096; SERVER-SETTINGS).
Block diffusion is not on the NPU. Estimate from measured parts: a 32-row pass on today's block route is
padded to 256 rows and costs 1.13-1.24 s (1.44-1.57 s with every row's logits), about 17 decode steps.
LowConfidence at 0.95 took 18.5-29.9 preview passes per 32-token block on three prompts (llama.cpp PR head,
CPU), so it would run at 0.7-1.2 tok/s. Self-speculative w=16 accepted 3.3-6.7 tokens per two passes,
1.3-2.6 tok/s. Both beat AR only if a W-row pass costs a few decode steps (a narrow route, the coreset's
smaller expert union, a batched head). Plan: [.plans/clover.md](.plans/clover.md); spec `OPEN-FAMILY-CLOVER`.

## Underdog Saluki 27B (Qwen3.8-27B, IQ2-mix GGUF)

Saluki runs on the NPU from its unmodified GGUF (2026-11-06): `/home/cyrus/models/Underdog-Saluki-27B-1.0-NPU2`,
registry `underdog-saluki:27b`. No kernel reads i-quant codes and the file mixes ten projection types, so the
model streams every projection at q8: a block with one scale and integer codes keeps them (bf16 scale), the
rest are requantized to int8 (`saluki-runtime.patch`, the all-q8 Qwen3.5 dense GGUF view). Pack check ALL OK
(kept codes 1.6-1.7e-3, requantized 2.4-6.6e-3). Against llama.cpp CPU on the same file: 488/491 (sequential)
and 487/491 (block) on the 491-token prompt at partial KL median 3-4e-4, 64/64 and 48/48 greedy tokens; against
an fp64 reference over the GGUF's values the NPU is closer than llama.cpp CPU is (KL 7e-6 vs 4e-4). The block
route needed the 27B's decode state ABI (paired KV rows, f32 conv history) in this runtime lineage. `phlegm run` and
`serve` (streamed and not) answer, with the thinking in `reasoning_content`. The gap is
speed: 0.91 s a decode token and 27-32 s for a 491-token prefill, against llama.cpp CPU's 0.25 s and 9 s.
Open: a CPU-vs-Vulkan control, a quiet-machine timing, kernels that keep the i-quant grids (or a smaller law).
Plan: [.plans/saluki.md](.plans/saluki.md); spec `OPEN-FAMILY-SALUKI`.

## Qwen3.5 MoE 35B-A3B (Ornith 1.5, Nex N2.5 mini, KAT Coder V2.5 Dev)

KAT Coder V2.5 Dev runs on the NPU from bartowski's unmodified Q4_0 GGUF. It uses the qwen36moe
q4_1 kernels and block prefill route through the manifest's new `gguf` view, with `q35_*` pack
ops from `upstream-patches/qwen35moe-gguf-runtime.patch`. `phlegm run` and `serve` answer
correctly. Against llama.cpp on the same GGUF, the 491-token prompt has partial KL median
1.9e-2; against a control holding the NPU's weights, 2.1e-3. Requantizing the GGUF's Q8_0 /
Q5_0 / Q6_K tensors to q4_1 is the gap. Development timing: 4.6 s for the 491-token prefill
and 84-86 ms per decode token. Ornith 1.5 and Nex N2.5 mini now run the same way, from their own
pure Q4_0 GGUFs built from BF16 (`/home/cyrus/models/{Ornith-1.5-35B-A3B,Nex-N2.5-mini}-Q4pure-NPU2`),
on KAT's kernels. Each agrees with llama.cpp at the level of llama.cpp CPU against Vulkan: Ornith
413/491, Nex 451/491. `phlegm run` and `serve` work for both. Open: a BF16 comparison for those two,
Ornith's heavier KL tail, a clean benchmark. The `<think>` block goes to `reasoning_content`
(SERVER-REASONING; checked on KAT). Plan: [.plans/qwen35moe-gguf.md](.plans/qwen35moe-gguf.md);
spec: OPEN-FAMILY-QWEN36MOE-GGUF.

Decode stream (2026-11-06, OPEN-DECODE-STREAM-UX): the decode layers were bound by their DMA issue
order, not by core arithmetic (a null-compute build walked as fast). dux.py's issue order, memtile
weight rings, early weight fills and a two-channel router W are now the default for this geometry,
with no main-core change and bit-identical logits on every gate. On a loaded box the decode step is
~10% shorter at ~500 tokens and ~5-10% at ~3000 (paired ratios; a quiet-box absolute is open). The
eight 35B-A3B model folders carry the new set as `open_kernels_stream` beside the old `open_kernels`;
switching is in [.plans/clover.md](.plans/clover.md).

## Native image model support

Qwen Image 2.1 and FLUX 3 Image Klein are requested for text-to-image and
editing through the CLI and image API, with native NPU inference. MOSS Image
SFT is also requested for its published text-to-image path. They are
new model families: the existing FLUX.2 Klein NPU schedule and C++ image
engine cannot load them by alias. The FLUX 3 port also needs its separate
prompt upsampler and the published reference inference behavior. Acquisition,
architecture differences, validation gates and current blockers are tracked
in [the native image model plan](.plans/native-image-models.md). Do not register
Qwen Image 2.1 or FLUX 3 as supported until actual NPU generation and editing
pass. MOSS's pinned checkpoint, DiT weight packer, SiTU and q/k RoPE kernel
checks are recorded in [the MOSS plan](.plans/moss-image-native.md). Its
headless Qwen3.5 text package, matching 40-set NPU kernel export, and
all-token encoder path are implemented. A 17-token native NPU run agrees with
the exact packed-weight CPU reference within 0.52% relative L2 and 0.999986
cosine across all 32 layers. Its 18.5% difference from the BF16 Transformers
reference is therefore attributable to the Q4/Q8 text package. A 290-token
prompt crosses the 256-token block boundary but has 57.9% final relative error
against BF16; the published CPU chunk and recurrent references also differ
substantially there, so long-context quality still needs a pinned optimized
reference. The MOSS VAE decoder's 96/192/384-channel native schedule and
weight packing are implemented. A 96-channel GroupNorm ran on the NPU within
1.04% of its CPU reference; the real 47.1 MB decoder pack covers 37 weights
and 30 norm blocks, and a decoded convolution matches its checkpoint within
0.66%. The full 102-operation 512-pixel VAE decoder ran on the NPU and matched
the FP32 reference at 0.77 mean absolute RGB levels and 47.7 dB PSNR from
identical random latents. The real-weight MOSS joint-block path now runs
RMSNorm/modulation, four GEMMs, q/k RoPE, 24-head attention, SiTU, and both
gated residuals on the NPU. One 512-pixel block matched its BF16 checkpoint
reference within 2.53% relative L2. All 42 main blocks completed 420 NPU
operations on a 512-pixel random embedded fixture, with 8.99% final relative
L2 and 0.995943 cosine against the published BF16 block stack. The native
text projection and two refinement blocks completed 32 operations from actual
Qwen3.5 NPU hidden rows and matched the BF16 checkpoint reference within
1.34% relative L2. Native latent embedding and timestep conditioning also
passed checkpoint comparisons. A conditioned 42-block forward from those
native stages completed 420 NPU operations at 512 pixels; its final velocity
was 2.97% relative L2 and 0.999755 cosine from the full one-step BF16
reference. `moss_native_sample.py` now stages the native text, DiT and VAE
passes with FP32 Euler/CFG updates and the negative branch's block-22 skip.
A default 50-step, guidance-3 run completed all positive and negative NPU
forwards plus VAE decoding and produced a coherent 512-pixel red block image
for `a red cube on a white table`. Its latent state remained finite. The
`phlegm image` CLI and `/v1/images/generations` endpoint each completed a
separate real NPU request; the API returned an encoded 512-pixel PNG. A
13-token prompt ran with verified RTP-derived kernel variants, whose changed
instruction streams exactly matched an independent direct export. Supported
asset configurations can now serve ordinary text-to-image prompts in the
validated 512-pixel bucket. The 1024-pixel and long-prompt quality gates
remain open; the API rejects resolutions absent from its configured kernels.
For FLUX 3, a native 1536-row half-scale RoPE fixture passed a 0.26% relative
L2 gate, and one real-weight per-mode image block completed nine NPU operations
at 512 pixels. Its output was 2.30% relative L2 and 0.99970 cosine from the
pinned BF16 block reference. The prompt upsampler and editing path still
need integration.
Five consecutive checkpoint-weighted image-mode blocks now complete 45 NPU
operations at 512 pixels. Their output differs by 4.61% relative L2 from the
published BF16 five-block reference, with 0.99890 cosine. The FLUX 3 FP32
autoencoder decoder weights are packed for the native VAE schedule. All 102
decoder operations completed on the NPU at 512 pixels; the resulting RGB
image differed from the published FP32 decoder by 0.604 mean pixel values
and 49.08 dB PSNR on identical random latents. Sampling and editing
integration remain open.
The five early text-mode blocks also execute on the NPU for an 80-token
fixture. All 45 operations completed; their final state differs by 4.22%
relative L2 from the published BF16 reference, with 0.99912 cosine.
The full 28-block shared text/image stack runs 308 NPU operations for 80 text
tokens plus a 32x32 image grid. Its final state differs by 6.25% relative L2
from the published BF16 shared-stack reference, with 0.99802 cosine.
Native image entry, 80-token text entry, and final image projection now run
with the checkpoint's packed weights. Their relative L2 errors versus the
published BF16 projections are 0.93%, 2.52%, and 1.41%, respectively.
Flow-map timestep, reference-time, early/single mode modulation, and final
modulation complete 13 NPU operations. A connected one-pass DiT validation
now runs all of those stages with 5+5 early blocks and 28 shared blocks.
Its early image/text states differ by 2.89%/3.56%, and its final shared state
by 4.15%, from the published BF16 full-model pass. The final velocity differs
by 19.92% relative L2 because the output head amplifies those state changes.
This remains a synthetic 80-token conditioning fixture, not prompt inference.
The FLUX 3 VAE encoder weights now cover all 28 native weight slices and 22
normalization blocks. Its 50 compiled streams execute the full 89-operation
512-pixel encoder on the NPU. On a local generated image, native reference
tokens differ by 10.51% relative L2 from the published FP32 encoder, but
decoding both token sets through the published VAE yields only 0.494 mean
RGB pixel difference and 50.02 dB PSNR. A synthetic checkerboard has 21.04%
token drift; traced stages locate its growth mainly in the first two down
blocks. Reference-image DiT conditioning and the prompt upsampler still need
native integration. A diagnostic 16-step flow-map sampler now runs every
FLUX 3 DiT stage and the VAE decoder on the NPU and produces a coherent
512-pixel red cube for a published CPU text-embedding fixture. Its final
latent remains finite. This validates the DiT sampler loop but does not
establish native prompt support. The checkpoint's Qwen3-VL language tower
has been packed as a headless 2.9 GB Q4NX encoder with all 398 tensors
mapped. The native text engine captures the eight FLUX 3 layer taps in one
pass and applies the published 80-token pad mask and position ID. Its 40-set
matching NPU kernel export completed. On a 15-token red-cube prompt, native
real-token taps are 5.56% relative L2 and 0.9988 cosine from the published
BF16 text reference; all native pad rows are identical as required, and the
overall 80-row error is 5.91% with 0.9986 cosine. A full native 16-step run
through this text encoder, all DiT blocks, and the VAE decoder produced a
coherent 512-pixel red cube on a white table with finite latents. The prompt
was passed directly; the published upsampler, editing path, and 1024-pixel
validation remain open before FLUX 3 CLI/API registration.

2026-10-27 FLUX 3 text-to-image update: the published prompt upsampler's 760
BF16 tensors were packed into a 475-tensor native Q4NX language/LM package.
The local NPU generated a complete 138-token red-cube caption with the
reference's T2I system prompt, 1.5 presence penalty and stop IDs. That caption
needs 145 Qwen3-VL text tokens; the native text encoder pads to the published
160-token bucket. Matching NPU text-entry, text-block and shared-block kernel
sets were exported. A full 16-step native upsampler → text encoder → DiT → VAE
run produced a coherent 512-pixel red cube on a white table with finite
latents. `phlegm image` completed a separate full native route with one
denoising step. `/v1/images/generations` completed another one-step request
and returned a valid 512×512 PNG, while `/v1/models` listed the configured
FLUX 3 model. The local configuration currently supplies 80- and 160-token
DiT buckets; longer upsampled captions require matching kernel exports.

2026-10-31 FLUX 3 editing update: one-reference editing runs natively. The
reference is prepared as published, encoded by the native VAE encoder, and
conditions the DiT through the `image_cond` per-mode and shared blocks
(reference RoPE `l=-1` in `qk_flux3`). A 16-step native edit matched the
unchanged published CPU pipeline at MAE 2.33/255 and PSNR 35.86 dB; `phlegm
image --image` and `POST /v1/images/edits` each completed a real NPU edit.
Edits pass the prompt as typed (the published `pup=False` path) because the
upsampler's vision tower has no NPU route yet. A 240-token text bucket also
passed on hardware; longer buckets wait on text-encoder pad masking across
prefill blocks. Qwen Image 2.1 editing needs a native Qwen3-VL vision tower,
LM image-embedding input and the Qwen VAE encoder; the server rejects it.
Neither model is registered as supported. Details:
[`specs/open-engine/flux3-image-native.md`](specs/open-engine/flux3-image-native.md).
Reference-image editing still needs the upsampler's vision tower and native
conditioning path; 1024-pixel kernels and validation remain open.

Qwen Image 2.1 update: Q4 lost a large text-tower activation and was rejected.
An all-Q8 package now runs the 36-layer text tower on the NPU; its 15 × 4096
red-cube embeddings differ by 7.589% relative L2 from BF16 CPU with 0.99712
cosine. The full 32-block image transformer, checkpoint entry and final head
run on the NPU. Layer 31 differed by 4.259% relative L2 from the independent
BF16 block stack, and the final head differed by 0.867%. A four-step native
FlowMatch Euler diagnostic produced finite latents. Native VAE probes have
passed the first real 64→1152 convolution and padded 144-channel RMSNorm.
All 82 active first-frame decoder parameter groups are now packed, and its 20
unique convolution streams compile. Four real-weight native convolution
probes, including fused upsampling and a 512-pixel width change, pass within
1.12% relative L2 of decoded packed-weight CPU arithmetic. The 144/288/576
RMSNorms also pass in convolution-padded physical widths 256/384/640.
The native 1152-wide single-head VAE attention kernel passed a checkpoint-QKV
probe at 3.624% relative L2 and cosine 0.999379; the shared 128-wide
attention regression also passed after that kernel change. The connected
150-operation VAE decoder executes 105 NPU kernels and decoded those four-step
latents into a coherent 512-pixel image. Against the official FP32 decoder
on the same latents it differs by 0.671% relative L2, cosine 0.999985.
`backend=qwen21` is wired into `phlegm image` and the image API, and prompt
length 1..64 is handled by checked RTP instruction patches. The published
40-step native route with raw Qwen3-VL text completed and decoded to a clean
red cube on a light table. A 21-token prompt passed the CLI end to end, and a
separate 16-token API request returned HTTP 200 with a valid PNG. One-step
requests are rejected because the published terminal stretch yields NaN there.
Native editing, 1024-pixel output, and longer prompts remain open.
See [the Qwen Image 2.1 native spec](specs/open-engine/qwen-image21-native.md)
for the checkpoint contract, validation artifacts and remaining gates; the
working implementation log is `.plans/native-image-models.md`.

## 5. Kernel correctness and performance backlog

### Qwen3.8-27B prefill accuracy

- [ ] Evaluate combined entry/output/up-gate precision in a private CLI with a
  fresh post-merge control and independent full-depth references.
- [ ] Resolve inherited cross-layer drift and rerun full-model/long-prefix gates.
- [ ] Promote arithmetic changes only after the existing acceptance bars pass.

Latest recorded checkpoint: isolated projection improvements pass, but L53/P17
still exceeds the 0.005 residual bound and retained full residual acceptance is
3974/4032, FAIL. The 2026-10-23 host/build merge checks do not resolve that failure.

Source: [Qwen3.8 validation report](specs/open-engine/plans/qwen38-main-rebaseline.md).

### Prefill pipeline stages 2-4

- [ ] Overlap expert gather/scatter and execution while preserving per-token
  expert order and safe instruction-buffer patching.
- [ ] Measure GEMM efficiency by shape; optimize output layout, split readback
  and q8 arithmetic where the measurements justify them.
- [ ] Then evaluate CPU/NPU placement for stages with available kernels and
  independent accuracy gates.

Stage 1 is done. Preserve byte-identical logits when arithmetic order is
unchanged; use the route's documented numerical gate when arithmetic changes.
Measure prefill before and after under comparable conditions.

Source: [prefill pipeline plan](specs/open-engine/plans/prefill-pipeline.md).

### K2 Uno speed

- [ ] Add exact/fast mode and block-size selection with documented fallback.
- [ ] Prototype cheaper multirow GEMV before committing to a full pass layout.
- [ ] Build and validate fast 4/8-row passes; consider the 16-core exact variant
  if the prototype measurements justify it.
- [ ] Measure short and long outputs and longer contexts before choosing defaults.

Base K2/Uno and row-parallel attention are already implemented. The remaining
speed plan covers phases 3-7; projected gains are not measured results.

Source: [K2 Uno speed plan](specs/open-engine/plans/k2-uno-speed.md).

### MoE speculative decoding and expert quantization

- [ ] Finish and hardware-validate the MoE verify pass, feature taps and block
  head, followed by the NPU drafter and serving accept/rollback loop.
- [ ] Validate greedy/distribution correctness and measure end-to-end speed
  before enabling speculation by default.
- [ ] Extend ternary and 2-bit affine formats to routed expert gate/up projections
  in decode and batched kernels; keep down projections at q4_1 per the plan.
- [ ] Validate conversion/packing, logits and model/tool quality. Resolve the
  ternary weight source and any training requirements before that model work.

These are distinct from the imported dense Bonsai ternary/DFlash work. Recheck
current implementation and upstream status before following old PR milestones.

Sources: [speculative decoding](specs/open-engine/plans/spec-decode.md),
[expert quantization](specs/open-engine/plans/expert-quant-2bit.md),
[drafter/model work](specs/open-engine/plans/dflash-drafters.md).

### trit-embed-3l sentence embedder

- [ ] Cut core compute, which now bounds the dispatch (0.48 ms, against 0.26 ms with every kernel
  stubbed): the projection epilogue, and attention's repeated bfp16 operand conversions.
- [ ] Hide the remaining host work per dispatch (0.53 ms an abstract, against 0.48 ms back to back).
- [ ] Add an embeddings route to `phlegm serve`, and run the encoder on Windows.
- [x] Ship the built graph and host library in the model folder (`bundle.py` -> `npu/`), so XDNA2 owners
  run it with only XRT and amdxdna (2026-11-04; staged, not yet uploaded to Hugging Face).

`open_kernels/designs/trit_embed/` runs all three ternary layers (2-bit weights, int8
activations) in one dispatch per 128-token slot. On a quiet machine (2026-11-01) it encodes 1,888
SciFact abstracts/s and 14,926 packed STSB sentences/s, 1.6x the 2026-10-31 build; its vectors
score with the CPU reference (four-task nDCG@10 33.81 against 33.85).

Source: [design README](open_kernels/designs/trit_embed/README.md).

## 6. Port inference to Rust last

- [ ] Port container reads/packing, manifest handling, XRT dispatch and stream
  patches, kernel lookup, decode and block prefill in testable slices.
- [ ] Match reference packing bytes and greedy tokens on the MoE model and a
  dense family before broadening coverage.
- [ ] Verify request isolation, state reset, errors and performance against C++.
- [ ] Switch user-facing flows after the required parity and performance gates.

This work does not gate the C++-backed CLI, server, onboarding or tray milestones.

## Resume on another machine

1. Transfer this file with the matching repository revision. It is a tracked-file
   candidate; creating it locally does not commit or push it.
2. Read this plan, the latest sync ledger and the result entries for the chosen
   workstream. Do not treat old "not built" headers as authoritative status.
3. For C++/NPU work, obtain the external reference checkout, model weights and
   compatible XRT/toolchain dependencies. Generated binaries and model data are
   not included in the source import.
4. Start with CLI/server integration and combined-tree hardware validation.
   Record new evidence beside the relevant spec/report and update this checklist.
5. Keep published benchmark reports restricted to matching clean, optimized
   release tags, as required by [Phlegm Bench](specs/bench/spec.md).


2026-10-25 performance continuation: experimental compensated-BFP and rotated
64x64 prefill pass full8192-position/model/bridge gates. Preliminary engine
trials suggest1.42x at1K and1.31x at8K, but only2/3 and1/3 stable pairs; no
accepted parity/default promotion. Dibs-isolated rerun waits for other sessions.
See [.plans/artlite-performance.md](.plans/artlite-performance.md) for exact
quality results, resource wait and next command.

2026-10-25 clean Dibs continuation: five of five1K paired engine trials pass,
prefill790 to1203 tokens/s (1.5204x); decode unchanged. Zero of five8K pairs
pass controls, so no accepted8K figure. Full8192 model correctness remains
validated; no ArtLite parity claim. Details in active performance plan.

2026-10-25 turbo continuation: run/serve --turbo selects an explicit HH-only
Llama1B companion image. Default arithmetic unchanged. Full300 and8192 turbo
quality profiles pass;47-token4096 boundary fails relative NLL (+3.55%) and
remains an experimental limitation. CLI generation and16-output bridge checks
pass. Clean1K turbo timing queued through Dibs; see active plan for waiter.

Turbo timing update: cancelled our pending measurement runner before launch to
avoid holding the machine while waiting on other sessions. No turbo benchmark
is queued/running. Exact retry command and known quality failure in active plan.

2026-10-26 clean turbo result:5/5 valid1K pairs,Dibs CLEAN before/after.
Compensated1136 versus turbo1111 prefill tokens/s (.97844x); no speedup.
Continue dataflow/producer investigation; no default promotion. Active plan
records exact evidence and retained boundary quality failure.

2026-10-26 continuation: compact producer/activation gather/double-weight-buffer
turbo variants built and match original turbo all8192 outputs plus16decode;
selected W2 also matches300partial-block outputs. No new speed claim. Device
SiLU/pack standalone diagnostic passes exactencoder plus separate arithmetic
checks while original9byte CPUcandidate mismatch remains. C++integration not
performed; shared CPUwork still prevents clean timing. Activeplan has evidence.
