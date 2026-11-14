# Compact FLUX conditioning and decoder research

Decision date 2026-11-09. Updated 2026-11-11 after reconciling the companion
Claude work. The experimental projected-conditioning input passes offline
checks, and a vector-diffusion enricher prototype has been trained and rendered
through the CPU reference. Compact encoder training, native image replay and
production image-quality acceptance remain open.

## Scope and ownership

Two independent workstreams:

1. Compact text conditioning, with **encoding first, enrichment second** as
   separate sets of layers and separate experiments.
2. VAE decoder distillation, preserving the existing latent interface initially.

Keep the FLUX.2 Klein 4B image denoiser fixed during these experiments. A later
workstream may distill it toward 2B or 1B; these are targets, not established
quality-preserving sizes. Model training, data and checkpoint experiments belong
in the sibling `model-experiments/flux-conditioning/` checkout directory. Phlegm owns NPU
kernels, packaging, runtime integration and hardware validation. This document
records the interface shared by those projects.

## Conditioning boundary

```text
Existing tokenizer
    -> compact encoder
    -> optional, separately trained enricher
    -> replaceable output head
    -> frozen FLUX image denoiser
```

The compact representation between encoder and enricher is the durable boundary;
its width is independent of the image denoiser's hidden width. Enrichment follows
encoding and operates on vectors, without requiring intermediate prose generation.
Preserve the original encoded representation as a reference during enrichment.
Tokenizer research is deferred; retaining token IDs does not require retaining
Qwen's large embedding table.

For the current Klein 4B teacher, Qwen3-4B hidden states at layers 9, 18 and 27
are concatenated, giving 7,680 features per position. The standard encoder uses
512 positions, including template and padding positions. FLUX's existing
`txt_in` projection maps 7,680 features to its 3,072-wide hidden representation.
Sources: [Qwen configuration](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/text_encoder/config.json),
[encoder](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/text_encoder.py),
[FLUX model](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py).
Pin source revisions and target tensor provenance before producing training data.

Train the compact encoder and its output head against the teacher's **projected
3,072-wide conditioning**, bypassing the original `txt_in` when consuming the
student output. The native runner now has an opt-in experimental input at this
boundary, pending live hardware acceptance. Preserve position metadata and sequence semantics;
matching dimensions alone does not establish compatibility. Existing native
support and its quality limitations are recorded in [the Klein spec](flux2-klein-native.md).

Use teacher-compatible 512 output positions for the first encoding control.
The working-representation target is 256 slots; reducing the denoiser-facing
sequence to 256 is a separate ablation. The initial control must specify how
internal slots map to output positions. Do not infer useful enrichment from input
token count or force generated prose to fill a token budget.

For the combined student experiment, initially freeze the encoder and output head and train
the enricher against projected conditioning from enriched teacher prompts, with
the original prompt as input. Compare against the same encoder with enrichment
bypassed. A future smaller denoiser gets its own conditioning head; neither 7,680
nor 3,072 is a permanent width requirement for the compact representation.

The companion design decisions recorded on 2026-11-10 select diffusion over
vectors for the enricher, with append-form teacher prompts. The original user
text supplies an exact token prefix. Copy its conditioning prefix and generate
the changed rows after it, including the shifted chat-template ending. Offline
LLM curation provides the enriched text used to produce training targets. The
enricher generates vectors. All teacher taps for a target describe the same
enriched prompt. The teacher supplies the encoder output for the first prototype.
The decisions are recorded in
`model-experiments/.claude/plans/flux-enricher-decisions.md`.

## Provisional experiments

Architecture and size remain hypotheses. Start around width 512, four distinct
blocks and 2/4/8 refinement cycles, comparing shared bidirectional attention,
a Gated DeltaNet/attention hybrid and an untied control. Distinguish sequence
recurrence from repetition across depth. A DeltaNet candidate needs an explicit
bidirectional design; causal recurrence alone does not expose the whole prompt.
Expand the width search toward 384/768 after the first comparison. Account for
embeddings, resampling, gates and output heads separately from core parameters.

These are research starting points, not a mandated architecture or a measured
NPU optimum. The compact encoder's refinement objective remains open. The
companion enricher uses the vector-diffusion design described above.
Do not equate additional recurrence with
additional stored knowledge or assume improvement beyond trained iteration counts.

Ternary matrix weights are a deployment target. Establish a higher-precision
control, then evaluate quantization-aware training early enough to influence the
design. Precision for gates, normalization and recurrent state remains open.
Storage savings do not establish NPU speedups.

The VAE work starts with a decoder student consuming the teacher's latent format,
trained and evaluated on both encoded-image and actual denoiser-produced latents.
Replacing the VAE encoder for reference-image editing is a separate follow-up.

## Acceptance and next action

Production image-quality thresholds and training budget remain open. Define held-out
prompts and acceptance thresholds before the bakeoff, including explicit
constraints, attributes, counts, spatial relationships, rendered text and long
prompts. Conditioning error is diagnostic; evaluate generated-image quality and
instruction following with fixed teacher controls and exact initial noise.
Evaluate decoder detail, text, faces and reconstruction independently.

Measure each stage and end-to-end latency, memory and model size. Separate
architecture effects from enrichment, position-count and quantization changes.
Use matched hardware trials with warmups, alternating pairs, load controls and
full provenance; do not transfer absolute rates between devices or stacks.

## Encoding harness (2026-11-09)

The companion `model-experiments/flux-conditioning/` directory now contains an
encoding-only teacher-target exporter, draft held-out prompts and conditioning
comparison harness. It reads an existing local diffusers-layout checkpoint,
retains all 512 projected rows, token IDs, attention masks and text positions,
and records the actual checkpoint file hashes and package versions. The declared
teacher revision is the native Klein control's
`e7b7dc27f91deacad38e78976d1f2b499d76a294`; local hashes are recorded without claiming
that the local files have independently been verified against the Hub revision.
Predictions bind to exact target artifacts, with real-token and padding errors
reported separately. No conditioning-error threshold is treated as image-quality
acceptance.

Validation: 15 focused checks pass on Windows, including a tiny Qwen3 checkpoint
export through the installed Transformers API. These exercise harness plumbing
and numerical diagnostics, not the real teacher or any trained student. The
environment was torch 2.13.0+cpu, transformers 5.14.1 and safetensors 0.8.0. Those
initial checks did not exercise the real teacher. The expanded suite has 17 checks.

### Real teacher export control (2026-11-09)

The supplied `C:/Users/josha/.flm/models/FLUX.2-klein-4B-NPU2` is an OpenFlowLM
BFP-packed runtime bundle. Original teacher components were downloaded separately
with the user's authorization to
`C:/code/model-experiments/artifacts/flux-conditioning/teacher/`. The 14 selected
files (15,812,009,114 bytes) match the pinned Hub LFS SHA-256/Git blob IDs at the
revision above. The VAE and complete diffusers pipeline were not downloaded.

All 12 BF16 exports match the pinned upstream text methods and published context
projection exactly: zero maximum absolute conditioning error over all 512 rows,
including padding, and identical token IDs, attention masks and text position IDs.
The cases comprise nine prompt controls plus diagnostics with exactly 511, 512
and 513 input tokens; only the last truncates. This is export-path correctness
evidence, not a student/image-quality acceptance or performance measurement.

Reference source: diffusers commit
`c60830ee365d520ab52b110dda562dd26f7b4d7f`, pipeline SHA-256
`8afefd06451d2bed0f34b59e930e60383c6a47fc8cebbec74ea30bfcdc7b79a9`.
The comparison executes the two upstream static text methods extracted by AST,
with a standalone CPU device helper, and loads the projection into `nn.Linear`.
It uses `Qwen3ForCausalLM`; the exporter uses `Qwen3Model`. Both use the same
Transformers Qwen3 backend, BF16 eager attention and eight CPU intra-op threads.
Windows build 26200, Python 3.11.9 and the package versions recorded above; no
NPU dispatch, denoiser, VAE or training was exercised.

Evidence root: `C:/code/model-experiments/artifacts/flux-conditioning/`.
`teacher-control-20261109/commands.json`, `export.log`, `reference.log` and `source/`
retain exact commands, raw output and source snapshots. Key SHA-256 values:

- `teacher/verified-upstream.json`:
  `3ae69c4397ff6db52c42c72ce37719b6d7bda740dcdb961928c871733ceae224`.
- `teacher-control-20261109/targets-bf16/manifest.json`:
  `4b8eba82bfc733c082d03c1610d0d8bffbac900d7853c800b169a3f2cb4a18bb`.
- `teacher-control-20261109/reference-bf16/report.json`:
  `0f3fb825b3a93562120a7e7aea562601d654b55ccef66bbf9e5f963bf86f2886`.

The companion `flux-conditioning/RESULTS.md` summarizes case counts, provenance
and reproduction. Weights, targets and reference captures remain ignored.

### Trashcan worker (2026-11-09)

An isolated worker is staged at
`/home/cyrus/Work/flux-conditioning-worker` on the user's Linux Mac Pro. Inventory:
Xeon E5-2697 v2, 12 cores/24 threads, 62 GiB usable RAM (54 GiB initially available),
138 GiB initially free disk, Omarchy 4.5.0/kernel 7.2.15, Mesa RADV 26.2.5. Both
FirePro D700s expose 6144 MiB through Vulkan. The existing Granite experiments use
PrismML llama.cpp at `6447906bf96c1a023980d3453fb59daa6af3bb23`; its device query
detects both GPUs. That does not establish GPU support for this PyTorch harness.

The separate worker environment uses Python 3.14.8, torch 2.13.0+cpu,
transformers 5.14.1, safetensors 0.8.0, huggingface_hub 1.20.1, numpy 2.5.4 and
pytest 9.2.0. All 17 harness checks pass there (7.21 s). No existing runtime or
experiment was modified.

A backend-selection probe measured `512 x 2560 @ 2560 x 4096`, one warmup and
three samples for each cell, in the order FP32 then BF16 at each thread count:

| CPU threads | FP32 median | BF16 median |
|---|---|---|
| 8 | 0.0606 s | 14.24 s |
| 12 | 0.0435 s | 17.29 s |

These are development matrix probes, not model or published benchmark results.
They select FP32/12 threads for the bounded two-prompt export trial. Its targets
remain separate from the Windows BF16 control. Supervisor PID 1592489 ran at
nice 10, downloaded pinned original components, then exported `control-fox` and
`boundary-513` and compared them against the pinned reference at the same dtype.
The completed result was retrieved and verified on 2026-11-10: both cases match
exactly on conditioning, input IDs, masks and text position IDs. Maximum absolute
conditioning error is zero. The supervisor exited successfully; no job remains
running.

| Stage | Wall time | Sampled peak RSS |
|---|---|---|
| Initial download/verification | 610.6 s | 5.15 GiB |
| Two-prompt export | 131.6 s | 17.44 GiB |
| Reference comparison | 125.6 s | 18.54 GiB |

Total job wall time was 867.8 s. Export and reference stages include model loading
and provenance checks. The reference's individual case calls, including projection
and output serialization, took 23.80 s and 22.81 s. These are two development
samples, not a model-speed benchmark or evidence of image quality.

This supports using Trashcan for small teacher-data shards. A first operational
choice is 32 prompts per shard at batch one, amortizing model loading. Extrapolation
from these two cases suggests roughly 14 minutes per export pass, plus a similar
reference pass; verify throughput on the first real shard. No corpus export or
training was started, and the production corpus/acceptance choices remain open.

Remote evidence is under the worker's `artifacts/`: `inventory.json`,
`host-probe.json`, `trial-process.json` and `trial-20261109/status.json`. The last
file records stage commands, elapsed times, sampled peak RSS and source hashes;
each stage has a log. `python3 <worker>/source/status.py` checks progress.
Local small evidence copies are in `.plans/trashcan-evidence-20261109/`:
inventory SHA-256
`1c33dbf42da9ad4e9c9121042ae607f4dd531377435aab8c78212cd8451b0a2a`,
host-probe SHA-256
`7299c32894e62772469f8d84d4ab474728afcc4a585c53338fa7a11a7105a0d0`.
Completed reports/logs are also retained locally in that directory's `complete/`:

- `report.json`: `46ae50e38be0dbcbff940ed8dcbdfdc8da363cbe0359b82c01733f301b8e6ecd`.
- `status.json`: `2e2934c9b4793a97044c800d6b8f1f11167cfb1709fdf28a3063c26ff0a9e805`.
- `target-manifest.json`: `93c5e1671ec6b9eed0900bff9adf8d05e027f5b9390574e2e67e221275b9e7c6`.

The report and target-manifest hashes match the supervisor's recorded hashes.
Phlegm integration remains on the NPU machine.

### Expanded parallel control and runtime input (2026-11-10)

Trashcan completed the remaining ten controls at FP32/12 threads, nice 10,
reusing the verified checkpoint: all ten match exactly. Together with the first
two, this gives 12/12 FP32 controls, including all three token boundaries.
Export took 300.314 s with sampled peak RSS 22,352,650,240 bytes; reference took
336.378 s with sampled peak RSS 22,234,296,320 bytes. These are development stage
times including loading/provenance work. They do not establish a speedup.
Supervisor PID 1625310 completed; no export or training remains running.

Remote evidence: `<worker>/artifacts/remaining-controls-20261110/`. Small local
copies: `.plans/conditioning-input-20261110/trashcan-controls/`:

- `report.json`: `7d80ff393fc0dc941fb30b70e87fd20f6a0e6836f456c32290d56630d0b3d2fb`.
- `target-manifest.json`: `e2d21f6d924d0d4ef6645e95a75d2bf5c99999f5bd76ebc41774ef93f828b3d8`.
- `status.json`: `aa5c9dd4a1ca109ab59320cafbf1740eece378003cb78b1967c1b89d4416a015`.

Phlegm's local CLI now accepts `--projected-conditioning FILE`; the native runner
validates `klein-projected-conditioning-v1`, all 512 rows/positions, finite
BF16/F16/F32 values, and exact prompt token IDs/mask before allocating the NPU.
It skips native text weights, text dispatches and `ctx_emb`. Text rows in X are
mutated by denoising, so the input is restored after draining pending operations
at every step. Full buffer/parameter layout is retained; unused text parameters
are zero. No kernel arithmetic, compiled image or ABI was changed. This does not
establish model-size or memory savings. See [the native contract and gate](flux2-klein-native.md).

Focused validation: 23 Python checks pass on Windows (0.69 s) and the transferred
Linux snapshot (1.05 s); six core image tests and four CLI tests pass. Tests cover
container validation, rounding, scheduling with mutated text rows, request
isolation, text-weight bypass, capture timing and CLI argument forwarding.
All 12 original Windows BF16 artifacts ingest with bit preservation; all 12
Trashcan FP32 artifacts ingest with prompt matching and BF16 bits exactly equal
to independent Torch conversion. The companion prediction adapter now retains
input IDs/mask as well as positions; its 17 harness checks pass.

In `.plans/conditioning-input-20261110/`, ingestion evidence SHA-256 values are:

- `windows-ingestion.json`: `b0741901aaa6f9edbdce9c11accce1730993183c5f78c66fcf87140b2184749d`.
- `trashcan-validation/artifact-ingestion.json`: `abb3549452f1591aa3d68014b8bf75c8055f52ecc43d1a8dc610e0f42b5ce24a`.
- `trashcan-validation/source-sha256.json`: `bd9c509143eee7139c162b9a49082300f7b2e7d6578ad25f1184b77bba924afb`.

These checks execute no NPU denoiser or VAE. The live capture/replay gate is ready
and awaits a complete image checkpoint and matching compiled Klein kernel set.
On 2026-11-11 the existing local Windows IRON Python 3.13 environment imported
`pyxrt`, opened the Strix NPU and passed the official SAXPY execution check.
The prior missing-installation claim was a sandbox access denial misread during
inspection. Activate `. C:\dev\mlir-aie\iron_env.ps1`; the native spec records
exact paths and stack. All 23 conditioning checks pass in that environment too.
Run native capture/replay locally first, then compare projected teacher inputs under
the existing native arithmetic quality gate with exact initial noise. Define
held-out image-quality thresholds and training resources before the architecture
bakeoff. The local handoff is `.plans/flux-conditioning-distillation.md`.

## Companion enricher reconciliation (2026-11-11)

Claude's work is in the same `model-experiments/flux-conditioning/` directory.
`ENRICHER_SMOKE.md` records a 5.6M-parameter prototype on 24 subjects and
16 appended details. Its best whitened PCA-256 x0 diffusion run achieved a
median nearest-target distance of 0.42 D against the predefined 0.35 D gate.
Eight of nine rendered samples were judged clean enrichments in that report.
Unseen subject/detail combinations remain a failure. Padding substitutions
preserved the scene in the limited teacher image checks. These observations
support further experiments with the padding policy and compositional training.

The companion exporter now batches unpadded real rows. Its saved batch-16 report
records 128/128 prompts bit-identical to padded real-row controls, 36.118 s total,
0.282 s/prompt. Evidence is
`artifacts/flux-conditioning/enricher-smoke-20261110/fast-export-check-b16.json`.
This explains why the new export route is faster than the older full-512-row
controls. Its shard protocol is `klein-append-rows-v1`.

At the reconciliation check, Trashcan had 1,596 curation result records for
groups 0–10,000 and an active paired exporter. Five completed FP32 shards held
468 accepted groups, with zero recorded prefix mismatch. The latest export log
reported 6,803 prompts at 1.433 s/prompt. Local saved curation records numbered
111 for the stopped local shard and 834 for the OpenRouter shard. Counts include
curation decisions and can include rejected groups. These are progress snapshots.
Claude's latest user-provided handoff leaves groups 10,000–30,000 awaiting a
curation worker. Its older decisions file still lists a Windows assignment there.

The runtime integration should assemble the copied prefix, sampled changed rows
and selected padding into the full `[512,3072]` input, carrying the original
prompt's token IDs/mask and canonical text positions. Existing stage3 sample
files contain only the conditioning tensor, so an adapter must supply the native
protocol metadata and prompt binding. Keep the 512-position encoding control
available while evaluating the enricher's padding policy.

The VAE and pipeline config are now present in the companion `image-pipeline/`
artifact directory alongside the existing `teacher/` components. Its provenance
file records the pinned revision and VAE LFS hash. Next assemble these components
for the native runner, locate/reconcile matching Klein kernels and run capture/
replay under the verified local IRON environment. The model/data work and native
integration can proceed concurrently using these shared interfaces.
