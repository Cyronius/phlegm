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

## Native image model support

Qwen Image 2.1 and FLUX 3 Image Klein are requested for text-to-image and
editing through the CLI and image API, with native NPU inference. They are
new model families: the existing FLUX.2 Klein NPU schedule and C++ image
engine cannot load them by alias. The FLUX 3 port also needs its separate
prompt upsampler and the published reference inference behavior. Acquisition,
architecture differences, validation gates and current blockers are tracked
in [the native image model plan](.plans/native-image-models.md). Do not register
either model as supported until actual NPU generation and editing pass.

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
