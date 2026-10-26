# FLM registry kernel coverage

Started: 2026-10-26. Scope: NPU kernel designs and model-family coverage in
Phlegm. Performance parity is tracked separately in `artlite-performance.md`.

## Target

Use `src/model_list.json` in the local `openflowlm-next` checkout as the FLM
model roster. Record its commit and file hash whenever the roster is refreshed.
An exported recipe, a compiled NPU image, and a model that runs correctly are
different milestones. Do not call a family supported from a recipe alone.

## 2026-10-26 inventory

The local FLM registry is `openflowlm-next` commit
`478dbea3f00f868ac58205803cc537b402c6cad5`, `src/model_list.json` SHA256
`27d0be65ae4081df7a325436a867baa84562bf9068cd680e40e706defc6ce834`.
`git ls-remote origin refs/heads/main` returned that same commit on
2026-10-26, so the pinned local roster matches the current upstream head.
`python utilities/audit-flm-kernel-coverage.py --registry
/home/cyrus/code/openflowlm-next/src/model_list.json` inventories all 46 entries
and 27 family labels: 32 have a text recipe source, 9 have a specialized
embedding/audio/diffusion source path, and 5 have no complete NPU recipe. These
are source-level counts; zero new model acceptance claims follow from them.
The full 46-row checklist is
[`specs/open-engine/flm-kernel-matrix.md`](../specs/open-engine/flm-kernel-matrix.md).

With `--xclbins-root /home/cyrus/code/openflowlm-next/src/xclbins`, the audit
also records 38 distinct packaged `.xclbin` **basenames** across registry
models. Basenames are an inventory aid only: identical names across models do
not establish the same geometry, binary, or runtime ABI. GPT-OSS packages
`attn`, `dequant`, `expert`, `layer`, `mm`, and `short_seq_mm`; Gemma4 adds
model-specific global/sliding attention, audio, and vision names. None of
those closed binaries was copied into Phlegm.
Embedding-Gemma's differently named packaged directory contributes twelve
shape-specific matmul assets. Phlegm now exports and numerically checks those
shapes from open source, but has no embedding runtime route.

The five no-recipe models are Gemma4 E2B/E4B/12B and GPT-OSS 20B/Safeguard.
The GPT-OSS reference pins the activation, sink and bias
math, but its existing MoE design is hardcoded to Qwen3.6 geometry. Qwen3-VL,
Qwen2.5-VL and the registry's other vision models have text routes only in this
audit; their tower routes remain to be established. Qwen3.8 27B has a recipe
source but its wide DeltaNet and FFN geometry remains unbuildable. Several
model-label aliases still need their actual container config and packing
checked; the script names each.

## 2026-10-26 GPT-OSS kernel components

Added `open_kernels/designs/gptoss_gate/`: one 64-channel NPU activation with
the GPT-OSS asymmetric clamps, 1.702 sigmoid factor and `(up + 1)` offset.
It consumes already-biased fp32 projections and returns bf16. The source hash
enters its build specialization. It does not make GPT-OSS runnable.

`source target/npu-dev/env.sh && dibs run build --label 'Build GPT-OSS gate NPU
design' --eta 8m -- python open_kernels/build_design.py
open_kernels/designs/gptoss_gate/gptoss_gate.py
target/npu-dev/gptoss-gate-build` passed: core ELF `.text` 3104 B, xclbin SHA256
`a7827ffb92597f67dd93431dcba8b715478983ac736be01a9b892251a25f05a0`,
stream SHA256 `2930a0699f031945a0318fcf456d229a9e7ab74df7a82863318425403d2b0608`.
The build compiled `gptoss_gate` and the shared sigmoid/exp/reciprocal helpers.

The first Dibs NPU numerical check stopped at XRT device initialization:
`mmap(... len=67108864 ...) failed (err=-11): Resource temporarily unavailable`.
This chat's hard and soft memlock limits were 8388608 B. The user raised the
current Codex app-server's limits to unlimited with authenticated `prlimit`;
new commands inherited them. The exact same Dibs NPU check then passed all
four cases: three random inputs and asymmetric clamp edges. Against the fp64
GPT-OSS reference rounded to bf16, worst scaled absolute error was
0.0001220703125 (gate 0.012), worst absolute error 0.0001220703125. Report:
`target/npu-dev/gptoss-gate-check/report.json`, SHA256
`4b5eeffeebd908f0f39375612d04d206acc708199299ad9751750309a6241bff`.
This is standalone arithmetic evidence, not GPT-OSS model validation.

Added `open_kernels/designs/gptoss_mxfp4/`: a standalone 2560-byte expert
chunk decoder producing a 32×128 fp32 tile. The Dibs build produced xclbin
SHA256 `d9c55e0d6f7b067cfc0b57ee7aa2103bfe88b59f50cbc0b1fe8302e25c1cc816`
and stream SHA256
`507a7bbf0ccc12f08da23d4af0f161571039a08ab89718995f09ead4d3c14c3a`.
Four Dibs NPU cases (three random, one all-code/scale edge) matched the
`model/mxfp4.py` decoder bit for bit, with zero maximum absolute error.
The report SHA256 is
`75028871e6c54e531bd0cfbc7baa364fe42a0cb160a2924ad720da670ce27e5c`.
This decodes one chunk only; expert GEMV integration, projection bias addition
and model validation are still missing.

Added `open_kernels/designs/gptoss_router/`: a standalone 2880-to-32 BF16
router with learned bias, stable top-four expert IDs and normalized weights.
Activation and bias share one input buffer to fit NPU2's two input DMA
channels. The Dibs build produced xclbin SHA256
`0627726fe6ec503f2d625b2d86f895cf1b92680f1513899e1ae8dae5945fafd0`
and stream SHA256
`e5c4cc60de9d27451a51128dfb282437a07ae278d201a2989bd28d7a5abb0208`.
Four Dibs NPU cases passed against FP64: three random inputs, including one
where bias moves the expert choices, plus an exact tie selecting IDs 0–3.
Maximum logit error was 5.329435225576162e-6; maximum normalized-weight
error was 6.900765550377219e-7. Report SHA256:
`489cf977e7a9380d882bfbef293c70fd61c83ede92bdbcc950ada68baa293ec8`.
This is a standalone route kernel; it does not make GPT-OSS runnable.

## 2026-10-26 Embedding-Gemma matmul set

Added `npu_offload/matmul/gemma_shapes.py`, `export_gemma.py`, and
`check_gemma.py` for the twelve registry-packaged projection shapes. The
whole-array design's f32 output exceeded one core's 64 KiB local memory with
two output buffers, so the core output FIFO now has depth one. The 1152-wide
shape uses two columns to satisfy the 64-wide tile divisibility constraint.

The Dibs build compiled all twelve shapes from source, then a Dibs NPU run
passed all twelve against FP64 products rounded to f32. The minimum cosine
was 0.9999999999998282; maximum absolute error was 4.5299530029296875e-6,
and maximum relative error (normalized by the maximum reference magnitude)
was 1.5061636576819776e-6. The ignored manifest
`target/npu-dev/embed-gemma-build/gemma_kernels.json` has SHA256
`3aad4c0a63952d1db763bae67abe7a7653c436995d5d1ca2b44e79d743fb5cdf`;
the numerical report `target/npu-dev/embed-gemma-check/report.json` has SHA256
`dc8695344a58be00c58db892fd8bf7c524bf0076f502194f89d1bd2edfda8e32`.
This establishes kernel arithmetic only; model dispatch and quality gates
remain open.

Syntax parsing and `git diff --check` passed. The existing CPU reference
suites `test_gptoss_ffn.py` and `test_attn_sink.py` passed 27 with 9 optional
skips under `target/npu-dev/env.sh`; they do not execute the new kernel.
After the MXFP4 decoder and router were added, the focused MXFP4/GPT-OSS
reference suite passed 73 with 11 optional skips. The three new NPU design
checks above exercise their compiled images separately.

## Next actions

1. Integrate the activation and MXFP4 decoding into a GPT-OSS expert GEMV.
   Wire the validated router, add 2880-wide expert projections and biases,
   connect the existing attention-sink kernel path, and add packing, manifest
   and runtime dispatch. Compare against `replica_gptoss.py` and the
   retained Transformers reference tests before reporting a runnable model.
2. Integrate the Embedding-Gemma matmul set into an embedding runtime and
   validate the model. Add Gemma4 recipes, vision tower routes, and wide
   Qwen3.8 buildability. Refresh the pinned roster and rerun the audit as FLM
   changes.

## Handoff

Preserve the separate ArtLite performance work and its uncommitted files.
Generated xclbins and numerical fixtures stay ignored under `target/`.


## Precommit validation, 2026-10-26

Gate/router checkers now reject nonfinite values, malformed byte counts,
invalid/duplicate expert IDs and weights outside[0,1] or unnormalized weights.
The existing numerical tolerances are unchanged.38 focused checker tests pass;
the combined checker/parser/header/prefill suite passes106 with no skips. No
kernel arithmetic changed, and prior device report hashes remain unchanged.
The inventory was reproduced from the pinned registry and packaged-image
names:46 models,27 labels,38 image basenames; regenerated Markdown matches
the checked-in matrix byte-for-byte. These remain component/source milestones,
not GPT-OSS or Embedding-Gemma model runtime acceptance.
