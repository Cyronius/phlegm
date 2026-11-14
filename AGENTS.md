# Working in Phlegm

## Start and resume

- Read `PLAN.md`, `.plans/README.md`, and the active plan before changing code.
- Local plans and handoffs live in ignored `.plans/`, independent of any agent
  product. Keep durable shared evidence in tracked specs; do not add plans to Git.
- Current performance work: `.plans/artlite-performance.md`.
- Preserve existing uncommitted work. Inspect `git status` and the relevant diff
  before editing; do not reset, overwrite, or commit unrelated changes.

## Direction

- Reach or exceed ArtLite performance while preserving model correctness. Treat
  parity as a measurement goal, never as an inference from offline tests.
- Keep the inference engine in C++ permanently. Rust owns the CLI, server,
  tokenizer, sampling and tray, as described in `PLAN.md`.
- This repository is the canonical home of `open_kernels/` and
  `specs/open-engine/`. Consult `open_kernels/UPSTREAM.md` before kernel work.
- The C++ engine lives in `src/open_qwen36/`, derived from OFLM. Companion patches in
  `open_kernels/upstream-patches/` have separate source bases; they are not a
  ready-to-apply series. Preserve the manifest/runtime ABI, local version-7 DIT
  mapping, and the Llama high/low probability contract.

## Research and optimization

- Read relevant recommendations in `~/code/xdna2-notes/recommendations/` before
  proposing kernel optimizations. Follow their finding IDs to evidence, stack
  versions and open questions. If that checkout is absent, record the gap.
- Start with R-0001 (budgets), R-0003/R-0008 (contexts and dispatch), R-0004
  (reference), R-0006 (measurement), R-0011 (buffers), R-0012/R-0013 (rates).
- Keep measured, documented, inferred and hearsay claims distinct. Do not carry
  absolute speeds between Strix Point, Strix Halo, operating systems or stacks.
- Profile dispatches, context switches, DMA floors, KV traffic and allocations
  before tuning arithmetic. Record hypotheses and failed experiments too.
- Check program memory, L1 plus stack, DMA/BD budgets and compiler output before
  accepting a new kernel shape. Rebuild every stream belonging to an image.

## Verification and evidence

- Run focused regression checks for changed routes; use `pytest.ini` and the
  relevant spec for prerequisites. Missing hardware/native dependencies are
  untested coverage, not passes. Run `git diff --check` for edits.
- Arithmetic changes need independent numerical and model-quality gates,
  including long-prefix/window boundaries; retain known failures explicitly.
- Benchmark matched models, quantization/quality, actual token sequences,
  context capacity, sampling, output count and cache/speculation settings.
  Record engine and HTTP timing separately, with raw samples and provenance.
- Use warmups, alternating paired trials and controls; record power modes,
  host load, other NPU clients and the complete driver/toolchain/hardware stack.
- Follow `specs/bench/spec.md` for published benchmark reports: clean optimized
  release tags and complete provenance. Keep development experiments separate.
- Never commit weights, generated kernels, captures or bulky logs. Link evidence
  by path and hash; check in small summaries or fixtures where appropriate.

## Production bar for OFLM kernel sets

Do not add q4nx support for any model: q4nx is being deprecated. New support
loads the model from GGUF.

Every OFLM PR that adds or changes support for a model also updates its entry
in `src/model_list.json`: the entry exists, names the files the engine loads,
and states the real quant, family and context length. A model with no correct
registry entry is not supported, whatever the engine can do.

A top-tier supported model ships finished and production-ready. Before you export
its kernel set as done, or open an OFLM issue or PR that calls it supported,
show each of these items with measurements:

- Prefill uses the block route (usually). State the reason for any exception.
- Decode and prefill are near optimal. Profile them and name what is left.
- No unusual context-length limits. Test up to the model's real context, not
  only the 4096 default.
- Everything runs on the NPU (usually). Name each host stage and its cost.
- Speculative decoding was explored: MTP, DFlash, DFlash2, prompt lookup or
  another drafter. Ship one, or reject it with the measured reason. We
  generally want one.
- `oflm add` and `oflm serve` run the model, not only `open_qwen36_cli`.
- The model ships quantized, usually q4. Prefer Unsloth's GGUFs when the
  engine supports their quant types; state the reason for any other source.
- Benchmarks are recorded: `oflm bench` results (prefill and decode against
  context length, with the machine and stack named) on the model's page in
  `docs/docs/benchmarks/`, and a row in the README's "Recommended models"
  table.
- Most of the model's own features are exposed: its vision and audio towers,
  MTP heads (as the drafter), reasoning and tool calling, and for image models
  every supported resolution and image editing. Name each feature left out
  and the reason.

If an item is not done, the PR stays a draft and lists the item.

## Handoffs

Update the active `.plans/` file with the date, completed work, exact commands
and results, remaining failures, prerequisites and next action. Link durable
technical results from their owning spec and keep `PLAN.md` consistent.
