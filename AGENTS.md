# Working in Phlegm

## Start and resume

- Read `PLAN.md`, `.plans/README.md`, and the active plan before changing code.
- Shared plans and handoffs live in tracked `.plans/`, independent of any agent
  product. Do not rely on conversation history or ignored `.claude/` files.
- Current performance work: `.plans/artlite-performance.md`.
- Preserve existing uncommitted work. Inspect `git status` and the relevant diff
  before editing; do not reset, overwrite, or commit unrelated changes.

## Direction

- Reach or exceed ArtLite performance while preserving model correctness. Treat
  parity as a measurement goal, never as an inference from offline tests.
- Keep C++ as the reference inference engine for CLI/server, onboarding and tray
  work. Port inference to Rust last, as described in `PLAN.md`.
- This repository is the canonical home of `open_kernels/` and
  `specs/open-engine/`. Consult `open_kernels/UPSTREAM.md` before kernel work.
- The OFLM C++ runtime is external. Companion patches in
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

## Handoffs

Update the active `.plans/` file with the date, completed work, exact commands
and results, remaining failures, prerequisites and next action. Link durable
technical results from their owning spec and keep `PLAN.md` consistent.
