# Linux GGUF baseline and experimental prefill

Development work on 2026-10-24, Strix Halo MAX+ 395. This is not a published
benchmark or an ArtLite parity claim. Existing Qwen3.8 residual and historical
coding-prompt failures remain open.

**Precision audit:** single-BF16-probability flash measurements below are
historical diagnostic experiments. They do not satisfy Llama's probability
hi/lo contract. A separate paired-probability variant is under validation;
no flash route has been promoted. See the latest continuation below.

## Device and source identities

The authenticated memlock adjustment succeeded: child commands inherit unlimited
soft and hard limits. XRT now enumerates Strix Halo, firmware 1.1.2.65, and
`benchmarks/reference/smoke.py` passes resident inference, decode and reset.
System power mode remains **default**. Attempting `performance` through XRT
failed with `DRM_IOCTL_AMDXDNA_SET_STATE`, `EACCES`; no mode change occurred.

Linux toolchain/source identities remain those in
[baseline readiness](artlite-baseline-readiness.md). The external runtime
development reconciliation is now retained as
[`linux-baseline-runtime.patch`](../../open_kernels/upstream-patches/linux-baseline-runtime.patch),
against OFLM `478dbea3f00f868ac58205803cc537b402c6cad5`. Its manifest support
is limited to versions 1–3 and local DIT 7; it rejects 4/5/6 and does not claim
Bonsai/Hadamard or DIT acceptance. Do not apply it after the other companion
patches. The original external checkout remains unchanged.

## Confirmed loading and route defects

1. Llama GGUF Q/K rows use adjacent rotary pairs, while this runtime rotates
   split halves. The direct packer did not undo that row permutation. Fixed it
   for Llama Q/K projections, with head-geometry checks. A normal CLI prompt
   changed from `##` to `The capital of France is Paris.`, also produced by
   ArtLite. Synthetic Q4_0/Q4_1 packing tests now check the reordered rows.
2. The GGUF-direct manifest deliberately omitted block prefill because its
   6144-byte chunks could not use the existing 5120-byte GEMMs. The actual route
   was sequential despite the complete q4nx projection set. Added a distinct
   f32-scale GEMM format and opt-in dense route (`DENSE_GGUF_GEMM=1`). Its
   scale conversion follows the existing decode kernel's RNE narrowing to BF16;
   it does not claim exact fp16-scale arithmetic.
3. The selected Q4_1 GGUF has a **Q6_K tied embedding/output head**. The ordinary
   Q4 head path silently requantized it to Q4_1, producing much larger logit
   drift. Added explicit `DENSE_GGUF_HEAD_Q8=1` for an existing Q8 head kernel.
   This is labelled Q6_K-to-Q8 head conversion, not identical quantization to
   ArtLite's native source head. It leaves projection weights unchanged.
4. Exercising that Q8 path exposed wrong chunk order in the GGUF head packer:
   it wrote row-quarter-major rather than band/K-tile/quarter order. Fixed it
   in the companion runtime and added a native scale/code regression spanning
   multiple 128-row bands and K tiles.

Both export knobs are opt-in and included in build identity. Default manifest
fixtures remain unchanged. Default-route fixture tests caught an unintended
change to unused nested build descriptions; that regression was corrected.

## Build budgets and numerical checks

The first f32-scale GEMM failed L1 allocation. Its larger, double-buffered
12288-byte band displaced the final RTP buffer past 64 KiB. The fused path
never uses the old two-pass nibble scratch; replacing that 4096-byte allocation
with a 64-byte placeholder made it fit. The f32 format explicitly refuses the
two-pass build. Ordinary q4nx/T2 paths retain their existing allocations.

Fresh complete exports are under ignored `target/npu-dev/kernels/`:
`llama32-1b-gguf-block` and `llama32-1b-gguf-q8head`, capacity 16384. Every
stream was rebuilt together. Their `toolchain.json` files retain compiler and
artifact hashes. F32 GEMM program occupancy is 4928/16384 bytes; the dense
f32 decode image's fullest core is 12016/16384 bytes. No new image is accepted
solely from these budget checks.

Independent dense-product checks in `utilities/gguf-prefill/check_gemm.py`:

| N × K, T=256 | Error vs original GGUF values | Error vs BF16 operands | Gate |
| --- | ---: | ---: | --- |
| 256 × 256 | 0.002596 | 9.87e-8 | Pass |
| 2048 × 8192 | 0.002600 | 9.59e-7 | Pass |
| 8192 × 2048 | 0.002616 | 3.79e-7 | Pass |

These use independently drawn codes/scales and a dense float64 reference,
without an inverse of the pool packer. Gates: relative Frobenius error <=0.005
vs original values and <=0.0001 vs operands. Native GGUF packing CTest passes.

## Full-model quality, including retained failure

`utilities/gguf-prefill/cpu_llama.py` uses the standard GGUF dequantizer and
float32 PyTorch SDPA, without C++ attention or NPU packing. The default reference
uses the original Q6_K head. Optional representation experiments are labelled
separately; they do not replace that source reference.

For the checked-in 300-token quality passage, all positions were dumped:

| Route | Median cosine vs source CPU | Argmax agreement | Next-token loss |
| --- | ---: | ---: | ---: |
| Sequential, original Q4-repacked head | 0.998685 | 278/300 | 2.808912 |
| Block, original Q4-repacked head | 0.998681 | 278/300 | 2.807659 |
| Sequential, Q8-repacked head | 0.999951 | 300/300 | 2.803957 |
| Block, Q8-repacked head | 0.999948 | 297/300 | 2.802656 |
| Source CPU | 1 | 300/300 | 2.800121 |

The Q8-head block route passes the issue-212 gate on this passage: median
cosine >=0.9999, argmax agreement no more than one percentage point below the
sequential control, loss no more than 0.5% worse. The three argmax flips are
retained in `target/npu-dev/quality/gate300.json`. Integer counts avoid rejecting
the exact one-percentage-point allowance due to floating-point rounding.

**Retained failure:** 47 real tokens starting at position 4090 with artificial
zero-filled history cross the 4096 attention window. Median cosine vs source
CPU is 0.999796 sequential and 0.999856 block, both below 0.9999. Argmax is
45/47 and 46/47; loss is 8.161499 and 8.168916. Using the explicit Q8 head target
in the CPU reference still fails (block cosine 0.999862). This does not establish
long-context acceptance, retrieval quality, or a passing absolute numerical
gate. The block route remains opt-in. This artificial case is not a recall test.

## Reproduction and pending measurement

Source `target/npu-dev/env.sh`. Key commands (artifacts/logs stay under target):

```bash
DENSE_GGUF_GEMM=1 DENSE_GGUF_HEAD_Q8=1 python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/llama32-1b.json --out target/npu-dev/kernels/llama32-1b-gguf-q8head --build-root target/npu-dev/kernel-builds/llama32-1b-gguf-q8head --max-ctx 16384 --force -j 2
python utilities/gguf-prefill/check_gemm.py --kernel-dir target/npu-dev/gemm-f32-small --driver target/npu-dev/harness-build/run_kernel --out target/npu-dev/gguf-gemm-small-check
python utilities/gguf-prefill/compare_model.py --cpu target/npu-dev/quality/llama-cpu-300.bin --sequential target/npu-dev/quality/seq-q8head300/logits --candidate target/npu-dev/quality/block-q8head300/logits --ids-file specs/open-engine/tests/fixtures/issue212/llama-quality-300.ids --out target/npu-dev/quality/gate300.json
```

Development timing uses the separately identified `phlegm-engine-survey-v1`
command and `utilities/gguf-prefill/paired_survey.py`. It preserves the existing
`phlegm-bench-v1` trial behavior. Both engines receive identical prompt IDs and
teacher-forced decode input tokens from one saved ArtLite greedy trajectory.
EOS never shortens the output count. Capacity, count and timing boundaries are
matched; resets, compilation, IPC and model loading are excluded. Per-step
latencies and predicted greedy tokens are retained. There is no HTTP TTFT
measurement in this protocol.

The runner saves warmups and the discarded first pair, alternates AB/BA, and
brackets each measured pair with per-engine controls. A control drift above 5%
invalidates the pair without discarding its raw record. Power mode and host
load are recorded for each sample. Quantization differences and the quality
failure above prevent treating diagnostic timings as accepted parity cells.

The completed diagnostic survey has three valid alternating pairs at each
length; all accepted pairs meet the 5% before/after controls. Median tokens/s:

| Prompt tokens | Phlegm prefill | ArtLite prefill | Phlegm decode | ArtLite decode |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 306.88 | 9625.09 | 32.19 | 63.82 |
| 2048 | 258.21 | 9933.22 | 25.55 | 62.63 |
| 4096 | 190.36 | 9308.57 | 18.05 | 59.97 |
| 8192 | 104.57 | 7751.84 | 11.44 | 54.96 |

Raw samples, control timings, exact prompt/decode inputs and power/load records
are in `target/npu-dev/survey-1k-8k/{raw.jsonl,metadata.json,summary.json}`.
This is a development comparison, not a clean-release benchmark or accepted
parity result. Context capacity is 16384, outputs 128 (127 decode steps), host
threads 4, power default. ArtLite source is a07a7b8893cb4a835352669d237609ba017e8ab1;
it uses its separate MLIR-AIE 1.4.5.dev56+g1a647f1 / Peano
22.0.0.2026102001+4f5df13d environment. Phlegm uses MLIR-AIE 1.4.2 /
Peano 21.0.0.2026080301+c9c5ecb7. Neither absolute rates nor quality are
transferred from the Windows/Strix Point reference notes.

### Dispatch evidence and next experiments

`target/npu-dev/profile-baseline-1k.log` records
`OFLM_OPEN_STEP_TRACE=2 OFLM_OPEN_DISPATCH_LOG=1` on 1024 tokens.
The first 256-token block takes 589.1 ms: projection GEMMs 291.4 ms,
per-token attention work 195.6 ms, and host middle work 90.7 ms (overlapping
categories are not summed). Each block has 80 projection GEMM dispatches and
256 score plus 256 PV dispatches across 16 layers, eight KV heads, and hi/lo
passes. Score/PV readback and host softmax grow with context; projection time
stays near 291 ms/block. Decode changes hardware context three times per token.
These observations motivate reducing dispatch/KV traffic before arithmetic;
a DMA-only control is still needed before asserting a bandwidth ceiling.

A separate runtime candidate reconciles PR209 at 814640bb9c58fb4db5d0ecd774952740f2a1c5fb
against its c05b00fd base, retaining the baseline hi/lo attention implementation
and local DIT7. The standalone `linux-runtime-pr209.patch` is against 478dbea3,
not applied after other companion patches. It supports manifest versions 1–7;
Bonsai/Hadamard remain outside this integration. Four native CTests and the
fixture-backed host numerical tests pass. All 300 prefill logits reproduce the
baseline byte for byte. The opt-in `ATTN_KVSLICE=1` image also reproduces all
300 logits and 15 subsequent decode steps exactly. Real-prefix/window and
paired speed checks are in progress; this is not promotion of the route.

Final offline Python checks: 1813 passed, 64 skipped, 8 deselected. Missing
native sources/hardware dependencies remain explicitly skipped coverage.
Serving, real long-prefix quality/recall, 32k/64k, Qwen3 0.6B, broader families,
and full combined-runtime acceptance remain open.

### Follow-up gates and compute ablation

The KV-sliced image passes the source CPU gate across 4137 real prefix tokens:
median cosine 0.999946275, 4135/4137 argmax matches, loss 0.058942562 vs
sequential 0.058935069. The isolated final 47 positions (4090–4136) pass with
47/47 matches and median cosine 0.999949996. The matching unsliced block run
is byte-identical at all 4137 positions and all 15 subsequent decode steps.
This repeated-prose fixture checks cache/window behavior, not recall. Evidence:
`target/npu-dev/quality/{gate-real4137.json,gate-real-tail47.json,kvslice-exact4137.json,kvslice-decode-exact4137.json}`.

`DENSE_FLASH_PREFILL=1` selects an eligible Llama GGUF flash route only with
`DENSE_GGUF_GEMM=1`; default q4nx and measured family tables stay unchanged.
Its first full image uses at most 4096 rows of flash window. It omits the
BFP16 c-tiled prep stream because the f32-scale Q4 GEMM returns host rows.
The runtime refuses unavailable windows instead of taking an unvalidated CPU
fallback. All streams are rebuilt; fullest flash core program is 13616/16384 B,
with the existing 8192 B stack allocation. This does not waive L1/DMA budgets.

Independent fp64 flash checks at 256, 2048 and 4096 rows pass: relative Frobenius
errors 0.001084, 0.001128, 0.001121; worst per-row cosine >=0.999995.
**Corrected route audit:** the first full-model runs of the flash-labelled
image did not execute flash attention. Their logs show per-token dxB because
`host_attn_fits()` required an explicit `head_dim` HF check key, omitted by
this Llama manifest. Their 300/4096 passing outputs validate that fallback only,
not full-model flash. Original logs and gate JSONs remain for audit under
`quality/{gate-flash300.json,gate-real4096-flash.json}`. The runtime candidate
now derives head dimension from checked Q width/head count when absent and
retains the separate CPU-family refusal. Fresh active-route gates are pending.
Standalone flash arithmetic checks above do execute the flash kernel.

The matched 8192x2048x256 `GQP_NULL_MM=1` ablation skips matmul but retains
weight dequantization, DMA, and output protocol. Its output is not inference.
After two discarded runs per variant, alternating AB/BA gives median native
4.350 ms / null 1.6945 ms when the context is already active, and native
6.9385 ms / null 4.2795 ms after changing context. Thus approximately 2.66 ms
is removed by skipping matmul and approximately 2.59 ms is added by context
transition in this experiment. This is **not** a pure DMA bandwidth ceiling.
Raw runs, transition classification and instructions are in
`target/npu-dev/gemm-null-ablation/`; source compilation log is
`target/npu-dev/gemm-f32-null-up-build.log`.

ArtLite's sequential 300-position logits were independently checked against the
same source CPU reference: 293/300 argmax matches, median cosine 0.999891818,
loss 2.80068156. It fails this project's cosine/argmax gate although its
next-token loss is close to the source and better than Phlegm on this passage.
This is a metric-specific observation, not a global model-quality ranking.
`quality/gate-artlite-seq300.json` retains every flip. It reinforces why the
speed survey cannot be treated as matched-quality accepted parity.

The 1k/8k KV-slicing survey is running with three paired rounds, the same
128 outputs and before/after controls. Its metadata additionally includes the
exact ArtLite checkout/packages, bridge build sidecar, kernel toolchain and
runner hashes. No concurrent NPU clients or heavy builds/tests are launched
during its timed window. Evidence: `target/npu-dev/survey-kvslice-1k-8k/`.

The completed KV-slicing survey has three valid pairs at both lengths:

| Prompt | Phlegm prefill | ArtLite prefill | Phlegm decode | ArtLite decode |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 313.84 | 9554.75 | 34.78 | 63.91 |
| 8192 | 106.67 | 7749.87 | 14.26 | 54.62 |

Paired decode ratios are 0.544077 and 0.259765, compared with the earlier
0.504479 and 0.208547. These are two separately controlled diagnostic runs,
not a direct paired unsliced/sliced speedup test. All 4137-prefix and decode
logits were byte-identical in the direct correctness comparison.

The f32-scale BFP16-matmul experiment failed its independent gate:
relative error0.009321872 vs source (limit0.005), and0.008950170 vs BF16
operands (limit0.0001). Its warm run was1.983 ms. No model route is exported or
promoted. The temporary assertion relaxation was reverted; experiment sources,
build output and results are retained in `target/npu-dev/gemm-f32-bfp-up/` and
`target/npu-dev/gguf-gemm-bfp-up-check/`.

A second timing-only ablation skips both matmul and dequant while retaining
the DMA schedule. Same-context native4.704 ms vs null1.562 ms;
changed-context7.279 ms vs null4.2155 ms. The native control differs from the
first ablation by >5%, so do not subtract across experiments or claim a stable
absolute bandwidth ceiling. Raw evidence: `target/npu-dev/gemm-dma-ablation/`.

### Active flash and bridge build audit

The corrected runtime derives a missing dense `head_dim` from the checked Q
width and head count. The active route log explicitly names NPU flash and
layer-major/stage-major scheduling; dispatch counts confirm af256/af512 and
16 af4096 calls in the full-model gates. Fresh active results:

| Fixture | Median cosine | Argmax matches | Next-token loss | Gate |
| --- | ---: | ---: | ---: | --- |
| 300 quality tokens |0.999948988|298/300|2.803782932|pass|
| 4096 repeated-prose tokens |0.999947059|4094/4096|0.059375929|pass|

Evidence: `quality/{gate-flash-active300.json,gate-flash-active4096.json}` and
`quality-flash-active*.log`. These replace the earlier fallback-only model
claims; earlier logs are preserved. A real4137 prompt is explicitly rejected
with `flash attention window unavailable and CPU fallback refused`;
`quality/flash-window-guard.json` records the passing rejection check.
Exports now reject AF_LMAX beyond4096 until that kernel/ABI budget is validated.
AF_LMAX enters the cache identity, including the compiled KV shape; shorter
windows have distinct build directories.

A bridge audit found global `-ffp-contract=off` changed core.cpp's rounding
relative to the CLI. The Linux builder now matches CMake Release: O3/NDEBUG,
AVX2/FMA/OpenMP, with `-ffp-contract=off` only on block_host.cpp. Every compilation
and link command is retained in the existing build sidecar field. This keeps
the host-oracle scalar-tail contract without changing core contraction.
`utilities/gguf-prefill/check_bridge_logits.py` confirms all16 prefill/decode
logit arrays match the CLI exactly (`quality/bridge-release-flash-active300.json`).
The failed global-contraction comparison remains in `bridge-flash-active300.json`.
Earlier survey bridges used their recorded O2 settings; the next flash survey
uses this separately fingerprinted Release bridge, so cross-survey differences
also include that build correction. No matched-release parity claim is made.

Current flash survey: `--out target/npu-dev/survey-flash-1k-4k --sizes 1024,4096
--outputs 128 --rounds 3 --prefill-route auto` with the isolated candidate
bridge and `llama32-1b-gguf-flash` image. The runner records NPU client PIDs/names
before and after each trial and invalidates unknown clients or inconsistent
platforms across the complete paired unit, in addition to the5% controls.

### Probability contract correction

The single-P8k image passed the overall8192 source gate (median0.999950261,
8190/8192 matches, loss0.035404180 vs sequential0.035401045). The first4096
logits are byte-identical between its4k and8k images, and its Release bridge
matches16 CLI steps. However, the real4090–4136 slice fails the loss allowance:
0.014254495 vs0.014176436 (>0.5%) despite47/47 matches and median0.999952236.
The artificial zero-history case also fails (median0.999817537). These local
failures are retained in `quality/gate-flash8k-{tail47,zero4090}.json`.

More fundamentally, the existing Granite flash kernel narrowed P to one BF16.
Llama requires probability hi/lo, so those timings are not contract-qualified
Llama acceptance. The new `AF_PSPLIT=1` variant stores an additional4096-byte
BF16 low plane per core. `exp2v` produces probabilities; both the row sum and
PV consume high plus low. The one-key softmax loop reduces spills; existing
single-P behavior remains the default for other families. The Llama GGUF flash
recipe forces paired-P and records `psplit:true`; the companion runtime rejects
Llama flash without it. Q/K, output and hardware buffer argument ABIs stay intact.

Paired-P standalone4352/8192 checks improve relative Frobenius error from
~0.00107 to0.00004072/0.00003997. Budget evidence in
`target/npu-dev/flash-pair4352/budgets.json`: program12160/16384 B,
placed-buffer end63584/65536 B (stack region reserved),8192 B configured stack,
maximum helper frame2816 B. Long streams retain <=64 repeats per BD, three
queued input BDs/channel, one pass in flight. The complete image is rebuilding
as `llama32-1b-gguf-flashpair8k`; model/long-window gates are pending.

### Paired-probability model gates

The complete paired-P image is `llama32-1b-gguf-flashpair8k`, capacity16384,
with32 attention tiers through8192. Every stream is rebuilt together. Its
manifest declares psplit:true and all attention build records force AF_PSPLIT1.
The runtime's rejection of the older unpaired Llama image is verified in
`quality/llama-flash-single-p-refusal.json`.

| Fixture | Median cosine | Argmax matches | Next-token loss | Gate |
| --- | ---: | ---: | ---: | --- |
|300 quality tokens|0.999948762|298/300|2.801876614|pass|
|8192 real prefix|0.999950368|8191/8192|0.035404766|pass|
|real4090–4136 slice|0.999951085|47/47|0.014180733|pass|
|artificial zero-history4090|0.999850534|46/47|8.169178686|fail|

The real boundary loss regression is resolved (control0.014176436, now+0.0303%
vs the prior single-P+0.5506%). The artificial case remains below the absolute
cosine threshold in both baseline and candidate; do not promote the route or
claim all quality gates pass. The full-prefix fixture remains repeated prose,
not a retrieval/recall test. The Release bridge exactly matches all16 CLI logit
arrays after an8192-token prompt (`quality/bridge-flashpair8192.json`).

Standalone paired-P checks also pass1e-4 error thresholds at4352/8192.
Q/K stress factors16/64 have relative errors0.00001976/0.00002256. The fresh
40-head default single-P regression passes its original gate (rel0.001077).
Default Granite recipe behavior is unchanged. Full Python checks after these
changes:1814 passed,64 skipped,8 deselected;4 native CTests and the pair
bridge handshake pass. Missing native prerequisites remain untested coverage.

ArtLite K-hilo sequential300 has median cosine0.999908882 and292/300 matches,
loss2.803245839: it improves cosine but still fails the current argmax gate.
This is retained alongside the single-K result, with no global quality ranking.
It does not turn the survey into a matched-quality acceptance benchmark.

The corrected variant's diagnostic survey is running under
`target/npu-dev/survey-flashpair-1k-8k/`, with1024/4096/8192 prompts,
128 outputs,3 alternating measured pairs, the existing controls and NPU-client
inventory. The bridge selects layer-major explicitly via --prefill-route auto.
Only this paired-P variant satisfies the Llama probability declaration;
earlier single-P flash timing remains separate historical diagnostic evidence.

### Corrected paired-P diagnostic throughput

The corrected Llama probability-hi/lo survey completed with3 valid pairs at
all3 lengths (9/9). All5% controls pass, power remains default, and no foreign
NPU clients were observed. Both engines use capacity16384,128 outputs/127
teacher-forced decode steps, threads4. Median tokens/s:

|Prompt tokens|Phlegm prefill|ArtLite prefill|Phlegm decode|ArtLite decode|
|---:|---:|---:|---:|---:|
|1024|770.49|9611.70|34.98|64.31|
|4096|713.63|9305.79|21.70|59.99|
|8192|584.63|7770.51|14.38|55.20|

Paired Phlegm/ArtLite ratios are0.080041/0.076687/0.075237 for prefill and
0.543659/0.361604/0.260861 for decode. Relative to the earlier diagnostic
Phlegm baseline, prefill rates are approximately2.51x/3.75x/5.59x. This is a
cross-survey comparison including the documented Release build correction,
not an isolated direct paired speedup estimate. ArtLite remains substantially
faster, quality/settings differ as recorded above, and the inherited artificial
failure remains. No accepted parity cell, default promotion or published score.

Raw evidence: `target/npu-dev/survey-flashpair-1k-8k/` (raw.jsonl, metadata.json,
summary.json, throughput.json). The development source checkpoint and numerical
output/reference hashes are indexed by
`target/npu-dev/evidence-paired-flash-20261024.json`, SHA256
`5a26b44e16e83b02fe527b01faa28740b4c69bd64df33a1d1524a84571383ff7`.
Its source archive includes pre-existing user changes and is explicitly dirty,
not a release. Earlier KV-survey binary hashes/provenance remain in metadata;
that early mutable bridge path was subsequently rebuilt. Later4k/paired bridge
variants have separate artifact directories. Generated sources used by the
paired image are retained in `target/npu-dev/export-scratch-probpair/`.

Next bottleneck work is decode: profile the existing ATTN_NULL compute ablation
before trying a larger attention-core split. This is timing-only, with wrong
answers by design; it must never become an inference route.

### Decode ablation setup

The next probe builds existing ATTN_NULL=1 with KV slicing, preserving FIFO
consumption while skipping cached-row attention compute. It is explicitly not
an inference image. Direct compilation needs the export staging tree (copied
projection sources and parent include paths); two initial attempts failed due
to missing staged gemv/vecmath files. Restoring the archived stage to its
original repository-relative location resolves those prerequisites.
The development CLI now applies --at-position to --bench-kernel and labels it
synthetic zero-cache coverage. Before this correction that option was only
applied in the normal request path, so it could not measure a long-window
kernel benchmark. No normal inference behavior changes.

The decode ablation has stable same-run controls at0/1024/8192. Native vs
ATTN_NULL median milliseconds/layer are0.988/0.987,1.382/1.044,4.381/1.485.
At4096,2/3 pairs are valid (2.596/1.323); the failed5.6% control stays in raw
records. The additional long-window cost is mainly arithmetic in this synthetic
probe. Zero cache and repeated fixed-token state are not a real trajectory;
these are not model throughput figures or a pure memory-bandwidth ceiling.

The new explicit ATTN_G4CORES1 experiment uses8 attention cores,4 query heads
per core, and4 two-KV-head streams shared by adjacent cores. Eligibility is
limited to the tested Llama32/8/64/RB4 shape with KV slicing. No measured default
family is changed. The existing G5 helper supports up to5 heads but its
Granite defaults also change the exponential. The first inherited EXPA2 probe
therefore passed source-model300 gates yet failed exact reproduction. The
corrected default retains ATTN_EXPA0; all300 reference logits are byte-identical.
Full8192 comparison and a fresh complete kernel-set export are in progress.
Program12832/16384 B and helper frame3968 B fit; the original layout probe's
buffer end is57344/65536 B, with12KiB attention stacks. Exact budget JSONs and
both prototype results remain under `target/npu-dev/decode-g4*-dx/` and quality/.

### Eight-core decode results and compact representation experiment

The original-exponential eight-core variant reproduces all8192 sequential
reference logits and all16 full-image long-run prefill/decode arrays exactly.
Its final Release bridge matches the CLI. All9 controlled survey pairs pass:

|Prompt|Phlegm prefill|ArtLite prefill|Phlegm decode|ArtLite decode|
|---:|---:|---:|---:|---:|
|1024|770.58|9591.56|39.26|64.29|
|4096|717.46|9302.10|30.79|60.04|
|8192|585.79|7778.89|23.67|55.10|

Raw source/tokens/controls: `target/npu-dev/survey-g4-flashpair-1k-8k/`.
Prefill is effectively unchanged; long decode improves over the earlier
14.38 tokens/s diagnostic. Cross-survey differences are not direct isolated
paired speedup estimates. Parity remains open; no defaults are promoted.

`DENSE_GGUF_COMPACT=1` is a further opt-in representation experiment. The
existing kernels already round source scale planes to BF16. The manifest can
store those same rounded values in5120-byte chunks and select the existing Q4
projection streams, retaining the GGUF reader/permutation, Q8 head, norms and
probability-hi/lo flash. It requires the explicit GGUF GEMM and Q8-head flags;
parent q4nx fields stay unchanged. Native packing coverage compares codes with
the f32 pool and checks BF16 rounding against adjacent representable values.

The prototype's actual packed weights fall from1018167296 to900726784 bytes
(117440512 saved); prefill_extra stays0. All300 sequential and300 block-prefill
logits plus16 decode steps match the prior representations exactly. All8192
sequential logits match too. Evidence: quality/compact-exact300.json,
compact-block-exact300.json, compact-exact-decode300.json and compact-exact8192.json.
The complete compact image is rebuilding under
`kernels/llama32-1b-gguf-compact-g4-flashpair8k`; full-image bridge and performance
checks remain pending. Python1817 pass/64 skip/8 deselected; native packing/
manifest2 CTests pass. The inherited artificial failure remains unaccepted.

### ArtLite prefill source audit

Reusable implementation techniques and precision caveats are recorded in
[the pinned source audit](artlite-prefill-code-audit.md), also contributed to
xdna2-notes raw/phlegm. Both engines batch prefill; fast matrix operands and
device-resident execution are the concrete architectural differences to test.

### 2026-10-25: idle recheck and compensated BFP feasibility

Idle1k survey completed: Phlegm786.005 prefill/43.318 decode tokens/s;
ArtLite9642.256/64.230.2/3 pairs valid; round1 rejected for11.526% Phlegm
prefill control drift. Raw measured load1=0.97–1.37, clients known. Thus the
~12.3x prefill gap survives low host load. Keep all invalid raw samples.
Evidence: `target/npu-dev/survey-compact-g4-flashpair-idle-1k/`.

Diagnostic CLI profile (same compact image,1024 IDs,2 outputs, no full-logit
dumps) completed in1299.1ms prefill. Projection dispatch totals115.5+252.3+112.5+
550.8=1031.1ms; blocked GEMM894.3, host385.0. Timers overlap and cannot be
summed as disjoint percentages. AF tier dispatches44.8+50.9+23.9+34.5=154.1ms.
Command: `OMP_NUM_THREADS=4 OFLM_OPEN_DISPATCH_LOG=1 OFLM_OPEN_STEP_TRACE=2
target/npu-dev/reference-next-build/open_qwen36_cli --model target/npu-dev/models/llama32-1b
--kernels target/npu-dev/kernels/llama32-1b-gguf-compact-g4-flashpair8k
--ids-file target/npu-dev/profile-1024.ids --max-ctx16384 --max-tokens2
--decode-route npu --pmode none --gemm-block` (space between flag/value in
actual shell). Log `target/npu-dev/profile-compact-idle-1k.log`, SHA256
58f9d9a7fd6b369473bcb111abaa455ee3b10c99a5a3be8ed6eb03fb725f12ec.

Added CPU-only `utilities/gguf-prefill/study_bfp_compensation.py`, importing
ArtLite's host encoder instead of copying LGPL code. Three products HH+HL+LH
vs four including LL, accumulating FP32 every64K. Three256x2048x64 seeds
pass existing source.005/operand.0001 thresholds with compensation; single
BFP fails. Full8192x2048x256 seed212 reproduces previous single-BFP hardware
error0.008950169588/0.009321871954 (operand/source), compensated3-product
0.000039846658/0.002616452592,4-product0.000007663827/0.002616102848.
CPU feasibility is not a hardware/model pass or speed claim. Result SHA256
559098d352037e7864e7b494bd6ecd78b09e28063a4565ceebe8ccf552e7c12c
at `target/npu-dev/bfp-compensation-cpu-up.json`.

Started isolated NPU prototype compile in `target/npu-dev/bfp-compensation-prototype/`.
Uses SDK identity-product decode for BFP residuals and one live C accumulator,
three correction MACs; this extra conversion cost must be timed. Experimental
recipe alone relaxes f32+BFP assertion; canonical assertion stays intact.
Next: compiler/budget checks, independent check_gemm gate, paired timing if
valid. Retain failed experiments; no model/default promotion.

### NPU compensated-BFP prototypes: correct, slower (2026-10-25)

Both isolated8192x2048x256 prototypes built every stream and passed
`check_gemm.py`: source relative error0.002616452197, BF16-operand error
0.000039846900 (unchanged limits0.005/0.0001). One output accumulator had
13.118/10.369ms cold/warm diagnostic dispatches; two output accumulators
reusing the A split had10.509/7.922ms. These are two-run diagnostic values,
not paired accepted timing. Both are slower than prior native-BF16 ~4.3ms.
Program .text is4464/4656 bytes on representative core0,2, under16KiB;
declared stack4096. Complete stack/frame and L1 acceptance remains unperformed
because neither candidate merits promotion. Independent check reports and
sources under `target/npu-dev/bfp-compensation-prototype/{check,check2,build,build2}`.
Canonical source assertions/defaults were not changed.

The SDK decodes BFP through a matrix product with identity; residual splitting
inside the hot MAC loop adds this cost and repeats conversions across outputs.
Two-output reuse helped this diagnostic but did not beat native. Next experiment
should move hi/lo conversion out of the MAC loop into producer stages, retain
packed panels across consumers, and check all L1/stack/DMA budgets before
compiling the image. This is a hypothesis, not an accepted optimization.
No full-model route, model-quality pass or speedup is claimed.

### Cached-panel variant and resource rejection (2026-10-25)

The cached-panel attempt with two weight buffers failed sequential L1 allocation:
end0x11048 exceeds64KiB. Preserved failed build3 log/project. Experimental
build4 reduced weight L1 FIFO depth2->1 and kept a12KiB stack. It built all
streams, passed the same gate (operand0.000039846900/source0.002616452197),
but two diagnostic cold/warm runs were9.002/6.441ms: still too slow.
Representative .text5424 bytes; disassembly gqp_comp frame0x2840=10304 bytes,
under declared12288. Representative core's mapped buffer end0xe048=57416
including stack, under65536. No changed repeat/BD geometry; reducing FIFO
depth is a local buffering tradeoff, not a new DMA layout. Kernel source SHA256
4e88718ab2292fd73935de08d4a53621094be332f5152418f3f58eebac65ec53;
check4/result.json SHA256ecf664f6d5b0846edabba169f76df68773e6920990cb8faae3daa670d1ec820a.
Artifacts: `target/npu-dev/bfp-compensation-prototype/{build3.log,build4,check4,disassembly4.txt}`.
No stable comparative timing or full-model acceptance claimed. Stop this
naive compensation loop: arithmetic is feasible, but its conversion/panel
traffic overhead is not competitive. Need producer-resident reuse or a
different exact decomposition before attempting another full-image route.

### 2026-10-25: transferred streaming/fusion techniques, isolated1.51x

Continued reading ArtLite's npu-prefill and aie-kernel-perf-optimization guides
(bank-conflicts, rejected, restructurings, numerics, data-movement) and actual
GemmQ pair workers. Both engines compile for AIE2P; compiler specialization is
not an established explanation for the gap. Same native projection rebuilt
with Peano22 vs21 passes its gate;3 valid paired trials show only1.0083x median.
Artifacts `target/npu-dev/gemm-peano21-vs-22-paired/`.

Implemented standalone compensated-BFP probes, leaving production flags and
manifest/runtime ABI untouched. New owned source:
`open_kernels/designs/gemm_q4_prefill/gemm_bfp_compensated_probe.cc` and
`gemm_bfp_compensated_emit.h`; reproducible builder and documented commands in
`utilities/gguf-prefill/{build_bfp_stream_probe.py,README.md}`.
`check_gemm.py` adds explicit bfp-hilo and bfp-hilo-pairs input checks (ArtLite
host encoder imported only for those); old BF16 check remains default.
`compare_gemm_trials.py`:60 samples/process, first4 discarded,3 alternating
pairs bracketed by5% reference controls, raw data/platform/clients/load/hashes
retained. Actual CPU load0.70–0.77 in the first positive paired probe.

All kernels below are8192x2048x256 unless stated. NativePeano21 ~4.27ms.
Moving conversion to producer and consuming high/low streams with2 independent
chains:3.90–3.93ms,1.0893x median,3/3 stable. Fusing Q4 dequant directly to
panels (no intermediate BF16 tile write/read) and using weight-row pairs:
2.818–2.851ms,1.50965x median vs native,3/3 stable. Source error0.002616452197
and BF16-operand error0.000039846900 pass the unchanged0.005/0.0001 limits.
Repro builds at3072x2048x512 and2048x8192x256 also pass: operand errors
0.000039235415/0.000039613703. No full-model gate or speedup claim yet.
Fused representative .text5296B, stack4096 with observed128/64B frames,
L1 buffer end0xf848=63560B including stack, under65536. Every stream rebuilt.
Artifacts: bfp-stream-compensation/{paired-native-vs-stream2,check2-qkv,check2-down};
bfp-fused-compensation/{paired-native-vs-fused,check};
bfp-fused-repro{,-qkv,-down}/{build,check} under target/npu-dev.

Negative results retained: neighboring-column dequant sharing is numerically
correct but0.89865x the fused baseline (3/3 stable), not a speedup; communication
and coupling attribution remains a hypothesis. Full inner-K unroll plus outer
pipeline hint passes numerics but warm diagnostic3.180ms, frame3072B versus
64B in the rolled MAC, so not promoted. Guide's register-pressure warning
applies. Artefacts bfp-fused-compensation/{build-shared,check-shared,paired-fused-vs-shared},
bfp-loop-probes/{build,check,disassembly.txt}.

Activation residency: canonical recommend_b_reuse currently alwaysFalse.
Proved a different arrangement builds and runs: L3 mega-strip FIFO forwards at
the MEMTILE with repeat_count=NRB, one initial fill held alive across every
output group; repeat is not on shim. All math identical, but paired full kernel
2.871–2.877ms vs2.819–2.826ms (0.98158x,3/3 stable). Lower DDR traffic is
not an accepted speedup. Static repeats and resident size may also make images
shape-specific, so shared-context compatibility remains unverified. Artifacts
bfp-resident-compensation/{build,check,paired-fused-vs-resident}.

Timing-only ablations of fused kernel, all3 pairs stable: full~2.82ms, noMAC
1.752–1.781, noDQ1.608–1.650, neither1.3195–1.3555. Invalid outputs explicitly
labelled in builder and timing metadata. These costs overlap and cannot be
added as disjoint wall time shares. Artifacts bfp-fused-null-{mm,dq,both}/paired.

Host AVX2 feasibility probe `utilities/gguf-prefill/experiments/pack_bfp_hilo.cpp`
produces exactly the host encoder's bytes for256x2048,256x8192 and a32x64 edge
fixture (zero, signs, exponent rollover, BF16 ties and tiny values). It has no
runtime hook and only a bounded finite input range.20 alternating CPU calls:
native/hilo median0.06122/0.15005ms (K2048),0.21880/0.56962ms (K8192).
These are CPU diagnostic timings, not independently bracketed engine trials.
Actual integration overhead and model quality must still be measured.

Focused regression command `python -m pytest -q specs/open-engine/tests/test_gguf_quality_gate.py specs/open-engine/tests/test_gguf_compact.py`:4 passed.
Python tools compile; git diff --check passes. User's existing work preserved.
Next: finish isolated profiling, integrate explicitly tagged experimental packed
input format with checked buffer sizing in external C++ runtime, build all
projection streams in a complete image, rerun300 and long-context/model/bridge
gates, then paired engine timing. Preserve probability hi/lo and the known
artificial zero-history failure. No defaults/parity promotion; no paid resources.


## 2026-10-25: completed engine trial and actual ArtLite prefill quality

The idle launcher completed at09:32UTC. Native versus compensated64 development
engine results (same source GGUF, Q8 head, token trajectories, capacity16384,
threads4; engine timing only):1024 accepted2/3 pairs,760.161 versus1076.689
prefill tokens/s (1.41642x),decode39.574 versus39.164;8192 accepted1/3,
578.617 versus755.154 (1.30510x),decode23.680 versus23.456. Insufficient valid
pairs for acceptance; rejected pairs remain in raw data. This run predates Dibs
coordination and must not be presented as a Dibs-isolated result. Evidence:
`target/npu-dev/survey-native-vs-hilo64-idle/{summary.json,raw.jsonl}`.

Actual ArtLite prefill300 through its own NPU Q6_K head: CPU-reference median
cosine0.9995212360605257,minimum0.9924211767146753,argmax290/300,
next-token loss2.801516981795352. Own sequential:median0.9998918179979034,
argmax293/300,loss2.8006815596276406. Own-sequential gate fails cosine;
common Phlegm sequential gate fails cosine and argmax. Loss stays close, so
this is a precision distinction under the chosen gate, not a general ranking
of model quality. Final captured row exactly matches untouched ArtLite forward.
JSONs:quality/gate-artlite-prefill300-{ownseq,commonseq}.json under target/npu-dev.

Current resource workflow: read ~/.codex/skills/dibs/SKILL.md; no project
.dibs.toml exists. All future shared-resource work uses dibs. A five-round
native-versus-hilo64 rerun to survey-native-vs-hilo64-dibs was blocked before
launch by reservations; `dibs wait cpu network runpod-ternary` is running in
exec session62512. Once FREE, rerun the exact blocked `dibs run --measure npu`
command with existing shared flock,1024/8192 inputs and max-host-load2.
Require initial and final Dibs CLEAN in addition to paired controls. No holds
were acquired by this session. Continue light documentation while waiting.

Source transfer audit: [xdna-engine](xdna-engine-code-audit.md) pins the
shared execution pipeline and persistent-module ideas, separates upstream
measurements from local evidence, and records why symmetric int4 is not a
drop-in replacement for affine GGUF Q4_1. New immutable hardware evidence
is saved in xdna2-notes/raw/phlegm/2026-10-25-compensated-prefill-model-and-quality-controls.md.


## 2026-10-25: completed Dibs measurement and source checkpoint

Dibs reported CLEAN before and CLEAN after five-round native versus hilo64
engine trials. At1024 all5/5 measured pairs pass: prefill789.574411 to
1202.828198 tokens/s (paired median1.520388x); decode39.742217 to39.663339
(ratio .998142). At8192 all0/5 pairs pass stability; no accepted rate or gain.
This supersedes the pending Dibs command above. Reservation was released.
Artifacts: target/npu-dev/survey-native-vs-hilo64-dibs/{raw.jsonl,summary.json,
metadata.json}. Engine timings only, dirty development sources, not release
benchmark/parity. Same Llama3.2 1B Q4_1 model; no other model measured here.

Reproduce: source target/npu-dev/env.sh; dibs run --measure npu --label
'Phlegm native versus compensated64 paired prefill' --eta20m -- flock
 target/npu-dev/.npu-work.lock python utilities/gguf-prefill/paired_phlegm_routes.py
 --bridge target/npu-dev/reference-hilo-snapshot-bridge/phlegm-reference-bench
 --model-dir target/npu-dev/models/llama32-1b
 --reference-kernels target/npu-dev/kernels/llama32-1b-gguf-g4-flashpair8k
 --candidate-kernels target/npu-dev/kernels/llama32-1b-gguf-hilo64-g4-flashpair8k
 --inputs-jsonl target/npu-dev/survey-g4-flashpair-1k-8k/raw.jsonl
 --out <fresh-output-directory> --sizes1024,8192 --rounds5 --threads4
 --max-ctx16384 --max-host-load2.0 (separate option values with spaces).

Focused precommit checks: dibs run cpu --label 'Focused prefill regression
checks before commit' --eta3m -- python -m pytest -q
 specs/open-engine/tests/test_gguf_block_prefill.py
 specs/open-engine/tests/test_gguf_compact.py
 specs/open-engine/tests/test_gguf_decode_g4.py
 specs/open-engine/tests/test_gguf_quality_gate.py:8 passed in0.31s.
Next: diagnose8K controls without weakening acceptance; continue closing the
ArtLite gap. Keep unrelated Rust/server and stream-header work outside this
performance checkpoint. Preserve existing source and generated artifacts.

summary.json SHA256 90ce76514f6e43f7e96bff0ddcbd1f33b3c9b4843e039208639e79f3ad692230
raw.jsonl SHA256 ef97097894cb7aec5c3a3660222c8ae0539f302c6e3a070db32d55560d16ac28


## 2026-10-26: clean turbo result — no engine speedup

Queued session9838 completed normally. Dibs CLEAN before and CLEAN after;
all5/5 retained1K pairs pass controls (discarded warmup pair failed and remains
excluded). Same model/source weights/head/forced128-token trajectory,capacity
16384,threads4,layer-major for both. Compensated64 prefill1135.766474 versus
HH-only turbo1111.275604 tokens/s,median paired ratio .978436703. Decode
39.714537 versus39.309394,ratio .989798614. This is a measured negative result:
fewer matrix products did not improve this implementation's engine speed.
Do not use cross-survey1203tps to compute its gain; the matched control is1136.
Known turbo boundary NLL failure remains. No default or parity promotion.

Evidence target/npu-dev/survey-hilo64-vs-turbo-dibs/ plus launcher.log.
SHA256 summary d043655eb4c23de97195a6839e96375a55229be5e3bcdbf269fb9374919f383e;
metadata846062ce9143f28e11b2138d6dd328f6df85bc07c8219efaaececbddca2f7b68;
raw91f511e038a62881198e0167a73c72c48128be360e6a9b9e82246a7c4fe2436b.
No pending benchmark. CPU shared activity currently busy; NPU free.

Next hypotheses (not measured explanations): HH MAC still emits/reads residual
panels, keeping conversion, buffer and transfer costs. Agent kernel_schedule
owns a compact producer/MAC prototype with current global input ABI preserved;
agent pipeline_audit audits raw matched routes; runtime_audit reviews exact
ArtLite device-resident pipeline transfers. No agent NPU work. Heavy builds
Dibs build; root numerical/model checks Dibs npu; new timing Dibs --measure.


## 2026-10-26 compact turbo and standalone device packing checkpoint

Three unpromoted whole-model variants preserve the original HH arithmetic and
external high/low input ABI. Fresh four-stream images/CDOs verified pervariant:

|Variant|Program/core|L1 incl4096B stack|Validated output|
|---|---:|---:|---|
|High-only weight producer|4640B|55880B|8192 positions+16decode exact|
|Also gather high activation panels|4656B|46664B|8192 positions+16decode exact|
|Also double weight buffering|4752B|58952B|8192 positions+16decode and300partial-block exact|

All equality aggregates for8192+16 are SHA256
89c0f4961c2114439981eaf91b43737028ef5000934fb208574f4843a2b48c20.
These inherit the turbo real47 boundary NLL failure (+3.55% vs1%target), not a
quality repair. IndependentQKV and longK arithmetic gates pass. No timing
claim for these variants; original HH clean result remains .97844x compensated.
Current installed turbo image is preserved, not silently replaced.

Rebuild with compensated64/build_family.py --out <fresh-target> and respectively
--turbo-high-only, --turbo-high-ab, --turbo-high-ab --weight-depth2 (space between
option and2). Builds under Dibs build; actualmemory,BD,compiler checks retained
in turbo64-high-{only,ab,ab-w2}-family/ under target/npu-dev. W2core5BD,
memtilemax24BD with16/8 pooluse,6S2MM/3MM2S. LongK rank-split gather keeps
outerrepeat<=64,wordstrides<2^20,positive strides,72B innerrows and large total
transfers. No hostpacker/runtimeformat change. Prototypebundles under
kernels/llama32-1b-gguf-hilo64-high-{only,ab,ab-w2}-probe/turbo.

New standalone ffn_silu_pack/ prototype computes SiLU(gate)*up from native
F32[F,T] then produces existing packeddowninput,4cores,F8192/T256. Original
CPUcandidate strictbytegate FAILS9lowplanebytes. Diagnosticexposesactualdevice
F32 and packedoutput from samekernel: independentCPUencoder(actualF32) matches
ALLbytes; diagnosticpacked output equalsoriginaldeviceoutput. DeviceF32
relativeerror1.214261629e-6 vsdouble and1.213874056e-6 vsindependentFP32,below
predeclared .02; finalpackedoperatorerror .00166343. Thus tiny prepackarithmetic
rounding changes BF16 decisions atmidpoints; encoder/layout matches for this
fixture. Originalfailure retained, not retrospectively relabelled passed.
No C++integration, fullmodelgate orspeedclaim forSiLUprototype. Extra context
cost and scalarsoftfloat remain concerns; vectorization/chain timing required
before integrating for performance. Diagnostic4cores:text9504B,frame896B
plusmain128/helper64,stack4096B,L1end54272B,2fills/2drains percolumn.

Evidence target/npu-dev/quality/turbo-high-{only,ab,ab-w2}-exact8192.json,
turbo-high-ab-w2-exact300.json,parallel-pipeline/silu-pack-fixture-v{1,2}/,
plus all build/source/resource metadata. Generated data ignored.36 focused
Pythonregressions pass. --reverse-init added to paired_phlegm_routes.py,
records actual creation/init order and preserves controls/defaults; fakeworker
checks pass. Next clean comparison must exercise reverse init to probe worker
placement; no relaxation ofpaired stabilityorquality thresholds.
