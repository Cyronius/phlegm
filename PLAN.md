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
channel as `reasoning_content` (SERVER-REASONING). Open: block prefill; the ten local / global
context changes a token; a clean benchmark. Plan:
[.plans/gemma4-moe.md](.plans/gemma4-moe.md); spec `OPEN-FAMILY-GEMMA4-MOE`.

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

- [ ] Keep the weights resident or replicated: today all 2.09 MB stream from DDR on every
  dispatch, and data movement is 54% of the 0.83 ms dispatch.
- [ ] Fill the 4096-byte weight packets, which are 56-81% full.
- [ ] Add an embeddings route to `phlegm serve`, and run the encoder on Windows.

`open_kernels/designs/trit_embed/` runs all three ternary layers (2-bit weights, int8
activations) in one dispatch per 128-token slot. On a quiet machine it encodes 1,199 SciFact
abstracts/s and 9,928 packed STSB sentences/s; its vectors score with the CPU reference
(four-task nDCG@10 33.80 against 33.85).

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
