# Reach or exceed ArtLite performance

Updated: 2026-10-26. Compensated-BFP development engine result at1K is1.5204x
versus native (5/5 stable,Dibs CLEAN before/after); full300/8192 and real4096
boundary gates pass. Initial HH-only turbo has NO speed benefit: matched clean
1K1136 versus1111 tokens/s (.97844x,5/5 stable). Turbo aggregate quality passes,
but real47 boundary NLL+3.55% fails its1% target; retained as experimental.
Compact high-only producer and activation-gather variants match prior turbo
all8192 logits +16decode exactly. Double-buffered weights fit and match8192/300 outputs exactly; bridge16passes.
Standalone SiLU/pack original9byte golden mismatch is retained; separate device
values prove exactencoding and bounded arithmetic. No runtime integration. No default/parity
promotion. All work local; paid resources need fresh approval.

## Objective and evidence

Reach or exceed ArtLite prefill and decode performance at comparable model
quality, including long contexts. Report each model/context/metric separately;
a short-context win does not establish general parity.

External trigger: [OFLM issue 212](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/212).
Its older OFLM survey reports substantial prefill gaps and context-dependent
decode slowdown. Reproduce comparisons against today's code before attributing
the gaps to this combined tree. Its ArtLite comparisons include different
quantizations, so distinguish product comparisons from kernel-isolation tests.

Local evidence:

- `open_kernels/UPSTREAM.md`: import/recovery status and runtime contracts.
- `specs/open-engine/issue-212-validation.md`: historical Windows/Strix Point
  before/after tests, not current Phlegm or matched ArtLite results.
- `open_kernels/recipes/dense.py`: `BLOCK_ATTN_ROPE` already includes `llama3`;
  eligible manifests emit `attn_block.prep=rope`. Do not duplicate that fix.
- `open_kernels/upstream-patches/issue212-runtime.patch`: matching runtime work,
  including Llama probability precision, Q8 embeddings and shared weight views.
- `~/code/xdna2-notes/`: evidence-backed recommendations; paths below are
  relative to that external checkout and are not vendored into this project.

The historical Llama 1B tests improved prefill by 2.59–3.75x but left decode
essentially unchanged. A short coding case still failed the strict argmax gate.
The combined tree has no fresh hardware acceptance. The Qwen3.8-27B residual
failure in `PLAN.md` also remains open. None of these establish ArtLite parity.

## Applicability audit

| Concern | State here | Next verification |
| --- | --- | --- |
| Llama per-token prefill attention | Recipe fix recovered | Build fresh set; verify runtime selection and high/low probabilities, including split windows |
| Small-model export coverage | Llama 1B/3B and Qwen3 0.6B specs recovered | Export and load complete sets through Phlegm; reject or clearly report missing routes |
| Q8 embeddings | Python reader recovered; C++ companion external | Reconcile runtime and exercise native rows plus full-model loading |
| Duplicated projection weights | Runtime sharing patch retained | Validate layout-compatible views, allocation ledger and unchanged logits |
| Tokenizer/catalog labels/fallback messaging | Primarily external runtime/registry concerns | Check Phlegm's actual loading/template/error paths; do not transplant registry patches blindly |
| Context-dependent decode slowdown | Still open | Profile attention slope and KV traffic separately from weight GEMV |
| Long-context prefill and memory | Still open | Measure tier/window boundaries, padded GEMM work and fixed-capacity KV allocations |

## Execution order

### 1. Establish a runnable combined baseline

- [x] Inventory device, XRT, firmware, compiler, model files and available disk/RAM.
- [ ] Record external C++ source revision; reconcile companion patches using
  `upstream-patches/sources.json`, including local manifest version 7.
- [ ] Build fresh kernel sets and reference adapter. Capture hashes, build knobs
  and actual selected routes; do not silently benchmark fallback kernels.
- [ ] Run focused recipe/manifest tests and model-quality gates before timing.
- [ ] Start with Llama 1B and Qwen3 0.6B; extend to Llama 3B/8B and Qwen3.5/3.8
  as resources allow. Report unavailable routes explicitly.

### 2. Reproduce a matched performance matrix

- [ ] Record current ArtLite commit, build options, backend, device and model
  artifacts. Run both implementations on the same machine in the same sitting.
- [ ] Measure exact 1k/2k/4k/8k prompts first; then 32k/64k where supported,
  with output capacity reserved. Add cases around attention tier/window edges.
- [ ] Match actual templated tokens, context allocation, greedy output count,
  disabled reuse and speculation for attribution. Where formats differ, label
  quantization and verify quality rather than calling containers identical.
- [ ] Keep HTTP TTFT, engine prefill, and decode timing separate. Existing
  `phlegm-bench-v1` is an untemplated 512–4096 engine protocol; preserve it and
  use a separately identified survey protocol for serving/long-context trials.
- [ ] Warm up, alternate paired runs and bracket trials with controls. Follow
  R-0006, including rejecting units whose controls move by more than 5%.
- [ ] Save raw repetitions, medians, spread, per-step latency, peak/idle memory,
  allocation ledger, processed tokens, finish reasons and recall/quality results.
- [ ] Report timeouts and censored runs explicitly. A sweep time limit does not
  prove a context length is unsupported.

Acceptance: every claimed parity cell has prefill throughput and decode
throughput at least ArtLite's, TTFT no worse under matched timing boundaries,
and passing quality gates. Close or repeat overlapping/noisy results rather
than declaring a win. List all losses and unmeasured cells.

### 3. Optimize the measured bottleneck

| Priority hypothesis | Experiment and evidence to consult | Required gate |
| --- | --- | --- |
| Dispatch/context overhead dominates | Count calls and switches; merge layer/tail streams, share GEMM context, group work by context (R-0003/R-0008) | Same math/logits, lower end-to-end time, no queue hazards |
| Decode KV walk limits long contexts | Profile latency versus position; contiguous KV windows per KV head and independent channels (R-0012, F-0203/F-0311) | Attention reference and full-model/long-context checks |
| Weight delivery limits decode | Compare DMA-only floor; test contiguous layout, memtile weight rings and multiple consumers (R-0012, F-0202/F-0303/F-0313) | Packing correctness and model quality; measured bandwidth and token gain |
| Prefill GEMM wastes padded work | Profile actual shapes, DIT padding, block sizes and null-compute DMA controls (R-0012, F-0343 through F-0347) | Independent precision gate plus whole-prefill gain |
| Attention windows dominate prefill | Measure split/merge work and realistic activation distributions; evaluate eligible flash routes (R-0012, F-0306/F-0308) | Causal masks, RoPE, high/low precision and cross-window correctness |
| Allocation/copies dominate footprint | Audit separate quantized, packed, bfp16 and KV storage; share only compatible layouts (R-0005/R-0011) | Lifetime/alignment correctness, finite outputs, measured memory reduction |

Read R-0001 and R-0009 before implementing: program/L1/stack and DMA budgets,
compiler scheduling, rounding and precision can invalidate an apparently faster
design. Treat recommendation rates as stack-specific evidence, not targets
portable from Strix Point to Halo. Resolve disputed findings by experiment.

Speculation, new quantization and CPU/GPU placement are later experiments with
separate quality and end-to-end gates. They must not hide a baseline regression.

## Resume checkpoint

2026-10-24: inspected issue 212, the recovered validation report, recipe family
selection, sync ledger and recommendations R-0001/3/4/5/6/8/9/11/12/13. Created
project instructions and this shared plan. No inference code changed and no
performance claim made. Existing CLI/server changes remain uncommitted.

2026-10-24 continuation: completed host/runtime inventory. Exact commands,
source revisions, scoped searches, results and recovery order are recorded in
[baseline readiness](../specs/open-engine/artlite-baseline-readiness.md).
Strix Halo is present but amdxdna probe fails because IOMMU is unavailable;
no `/dev/accel` exists. XRT/build tools, Python test dependencies and selected
small-model weights are unavailable. External runtime is clean at issue212's
base and only accepts manifest versions 1–3. No inference code changed; all
existing uncommitted work was preserved. Hardware, numerical and performance
checks remain untested. Documentation validation: `git diff --check`.

2026-10-24 unattended setup continuation: the new boot has IOMMU and a bound
NPU driver; `/dev/accel/accel0` is visible outside the sandbox. `sudo -n` requires
a password, so provisioned the tools in ignored `target/npu-dev` instead.
Built matching XRT 2.27/plugin, pinned MLIR-AIE/Peano, an isolated ArtLite
comparator, Linux bridge with provenance, and the complete Llama 1B set at
16384 capacity. Downloaded a Q4_1 GGUF and tokenizer/config with recorded
revisions/hashes. ArtLite's small direct-driver arithmetic test passes on NPU.

Reconciled a development external runtime for versions 1–3 plus local DIT 7;
versions 4/5/6 and Bonsai/Hadamard integration remain explicitly unavailable.
Preserved Llama hi/lo probabilities through the sub-block merge; numerical and
model-quality gates remain untested. Fixed duplicate host row-patch definitions
that prevented the canonical harness from compiling. Added a native regression
and explicit missing-runtime skips. Existing CLI/server changes preserved.

Exact commands, source identities, artifact ledger, failed attempts and results:
[readiness continuation](../specs/open-engine/artlite-baseline-readiness.md#continuation-local-linux-development-stack-2026-10-24).
Python: 1809 passed, 64 skipped, 8 deselected; Rust core: 33 passed, 2 ignored,
plus 1 release-provenance test; compiled bridge handshake/provenance: 1 passed;
reference manifest/GGUF CTests: 2 passed. Canonical harness builds. ArtLite NPU
arithmetic: 1 passed. `git diff --check` passes. Hardware/model tests are not
substituted by these checks.

2026-10-24 hardware continuation: user authenticated the memlock adjustment;
XRT enumeration and resident model smoke now pass. Firmware 1.1.2.65 is read
back. Power mode remains default; selecting performance failed with EACCES.
Normal CLI generation works after fixing Llama GGUF Q/K rotary row ordering
in the external development runtime. Added native row/packing coverage.

Found the selected GGUF route had no block prefill, and its Q6_K tied head was
silently requantized to Q4. Added opt-in f32-scale projection GEMMs and explicit
Q8 head, plus a companion runtime correction for Q8 head band/tile order.
All streams were exported fresh. Independent GEMM checks pass; 300-token
full-model quality passes with the Q8 head. The artificial zero-history
47-token case at position 4090 still fails the absolute cosine gate in both
routes. Preserve it; the new route stays opt-in.

Durable results, exact commands and remaining failures:
[Linux GGUF prefill](../specs/open-engine/gguf-prefill-linux.md).
The consolidated development companion is
`open_kernels/upstream-patches/linux-baseline-runtime.patch`, based at 478dbea3;
it remains limited to manifest versions 1–3/7 and is not a full combined runtime.

Completed diagnostic command (source `target/npu-dev/env.sh`):
`python utilities/gguf-prefill/paired_survey.py --bridge target/reference-bench/phlegm-reference-bench --artlite-python target/npu-dev/artlite-env/bin/python --model-dir target/npu-dev/models/llama32-1b --kernels target/npu-dev/kernels/llama32-1b-gguf-q8head --ids-file specs/open-engine/tests/fixtures/issue212/llama-context-8192.ids --out target/npu-dev/survey-1k-8k --sizes 1024,2048,4096,8192 --outputs 128 --rounds 3`.
This is diagnostic `phlegm-engine-survey-v1`, not accepted parity or a published
benchmark. All 12 measured pairs pass 5% controls. At 1k/8k prompt lengths,
Phlegm prefill is 306.88/104.57 tokens/s vs ArtLite 9625.09/7751.84;
decode is 32.19/11.44 vs 63.82/54.96. Complete table and stack identities are in
the Linux GGUF report. Raw evidence: `target/npu-dev/survey-1k-8k/`.

Profiled the baseline after timing: projection GEMMs ~291 ms per 256-token
block; score/PV have 512 dispatches per block and host readbacks/softmax.
Decode has three context changes/token and a strong KV-length slope.
No DMA-only bandwidth ceiling has been measured yet.

Reconciled an isolated PR209 runtime, saved as standalone
`open_kernels/upstream-patches/linux-runtime-pr209.patch` against base478.
Preserved local DIT7 and Llama hi/lo merge; manifests1–7 compile, Bonsai remains
unintegrated. Four native CTests and fixture-backed host numerical tests pass.
The original 300-position logits reproduce the baseline exactly. A fresh
`ATTN_KVSLICE=1` image matches those logits and all 15 subsequent decode steps
byte for byte. Real-prefix 4137-token CPU/sequential/candidate quality checks
are running before paired KV-slice timing. Logs under `target/npu-dev/quality*`.

Final broad Python rerun: 1813 passed, 64 skipped, 8 deselected (21.91 s).
Next: complete real-prefix and decode checks; measure KV slicing with paired
controls; assess native flash attention and projection DMA floors; preserve and
resolve the artificial-window failure. Serving and other model-family gates
remain open. No parity claim or default-route promotion.

2026-10-24 further unattended continuation: all 4137 real-prefix logits and
15 decode steps are byte-identical between KV-sliced and unsliced block routes.
CPU quality gates pass across the full prefix and its 4090–4136 slice; the
artificial zero-history failure is still retained. Added opt-in Llama GGUF flash
selection and a fail-closed runtime check for unavailable flash windows.
Independent flash checks at 256/2048/4096 rows and full-model 300/4096 positions
pass. Flash fullest program 13616/16384 B; first image limited to4096 rows.

Measured a matmul-null control: same-context native4.350 ms vs null1.6945 ms,
changed-context6.9385 vs4.2795 ms (8192x2048x256). Null retains dequant/DMA;
this is not a pure DMA floor. ArtLite sequential300 misses current cosine/
argmax gates (293/300, median0.999891818), retained explicitly. Full Python
rerun after flash recipe change:1814 passed,64 skipped,8 deselected. Rust core
33 passed/2 ignored, release provenance1 passed; both bridge handshakes pass.

Current command is the same paired runner with `--bridge target/npu-dev/reference-next-bridge/phlegm-reference-bench --kernels target/npu-dev/kernels/llama32-1b-gguf-kvslice --out target/npu-dev/survey-kvslice-1k-8k --sizes 1024,8192 --outputs 128 --rounds 3`.
Do not run heavy builds or other NPU work until it completes. Then evaluate
controlled KV gains, measure flash performance, and test the existing BFP16
matmul mode with f32 scale dequant as an independent arithmetic experiment.
The design assertion now permits that explicit mode, but no BFP16 result or
full-model export is accepted yet. Preserve all gate failures. Details and
commands are in the owning Linux GGUF report.

Route-audit correction: the first flash-labelled full-model300/4096 runs used
per-token dxB, not flash. `host_attn_fits()` rejected Llama's missing explicit
head_dim HF key. Their passing gates are fallback coverage only. Standalone
flash arithmetic results remain valid. Corrected the candidate runtime to
derive head_dim from checked Q width/head count, with division guards and
unchanged CPU-family refusal; CLI/bridge rebuild and fresh active flash gates
are pending. Do not publish earlier full-model flash claims.

KV survey completed: all6 pairs stable. Decode34.78/14.26 tokens/s at1k/8k vs
ArtLite63.91/54.62, paired ratios0.544077/0.259765. Prefill313.84/106.67.
BFP16-matmul f32-scale arithmetic gate failed (rel0.009321872 source vs0.005
limit); reverted the temporary assertion relaxation. No model promotion.
DMA-null experiment native control drifted >5% vs earlier experiment; raw
same-run results retained without cross-run subtraction.

Active flash correction is now verified: dispatch logs show af kernels and
layer-major stages. SourceCPU gates pass at300 positions (298 matches,
median0.999948988) and4096 (4094 matches,0.999947059). A4137 prompt is rejected
without unsupported CPU fallback. AF_LMAX exports are explicitly capped4096
until longer DMA/stream/ABI budgets are validated; cache identity includes it.

Linux bridge now matches CMake Release O3/NDEBUG and applies contraction-off
only to block_host.cpp. Global contraction-off was caught by a failed logit
comparison and corrected. The final Release bridge matches all16 CLI logit
steps exactly. Current survey: `survey-flash-1k-4k`, same128 outputs/3 paired
rounds, `--prefill-route auto` selecting layer-major. Runner inventories NPU
clients and rejects unknown clients or cross-unit platform changes. Avoid heavy
builds/tests or other NPU work until completion. The BFP16 failed experiment
was reverted; do not re-enable it as a model route.

Long-stream experiment after R-0001/R-0010/R-0011 review: shim repeat counts
must stay<=64. Added explicit AF_LONG_STREAM=1 (cache/probe/build env captured),
which splits K/V into consecutive <=64-chunk fills and limits >4096-row
streams to one pass in flight (Q+two KV fills=3 BDs/input channel). Existing
short defaults keep two passes. Bound experimental max8192; no larger export.
Standalone4352 and8192 fp64 gates pass (rel0.001062/0.001071, worst cosine
>=0.999996), with8 repeated dispatches each. Program13744/16384; placed core
buffer end59488/65536 incl reserved8192 stack region. Evidence:flash-long*.
Native parser extended to8192 with parse/refusal tests;4 CTests pass and15
focused Python tests pass. No arithmetic loop or KV/Q ABI is changed.

Running full fresh image export: `ATTN_KVSLICE=1 DENSE_GGUF_GEMM=1 DENSE_GGUF_HEAD_Q8=1 DENSE_FLASH_PREFILL=1 AF_LMAX=8192 AF_LONG_STREAM=1 python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/llama32-1b.json --out target/npu-dev/kernels/llama32-1b-gguf-flash8k --build-root target/npu-dev/kernel-builds/llama32-1b-gguf-flash8k --max-ctx 16384 --force -j 2` (log llama-gguf-flash8k-export.log).
Independent CPU8192 reference also running (quality-cpu-real8192.log).
Next: numerical/full-model4137 and8192 gates, explicit route audit, Release
bridge alignment, then controlled8k flash timing. Preserve inherited failures
and rejected BFP16 math; no parity/recall/serving claim.

Current full-model long-prefix sequential control is also running, using the
KV-sliced decode image without --gemm-block, all8192 fixture positions plus15
greedy decode steps: `quality-real8192-seq-kvslice.log`, output
`quality/real8192-seq-kvslice/logits`. CPU8192 reference finished, finite.
Do not run another NPU client until that command finishes. Full8k flash image
export is independently compiling on CPU; both must finish before candidate
NPU checks. Broad Python after long-stream changes:1814 passed,64 skipped,
8 deselected. Native4 CTests and long Release bridge handshake pass.

Full8k flash image export completed (all32 attention tiers and every other
stream rebuilt together). CPU8192 reference is finite. Sequential8192 NPU
control is near completion; candidate checks must wait for its DONE/client exit.
Long native parser tests and Release bridge handshake pass. Long-image raw
artifacts remain ignored under target; generated staging/locks archived there.

IMPORTANT precision audit: single-BF16-P flash is not Llama's required hi/lo
probability contract. Do not treat its passing300/8192 overall gates or earlier
749/744 token/s timings as a qualified Llama route. Its full8192 gate passed
(cos0.999950261,8190/8192), but isolated real boundary47 loss0.014254495 vs
seq0.014176436 fails0.5% (47/47 argmax, cosine0.999952236). Artificial zero
history still failscos0.999817537. First4096 outputs of4k/8k images are exactly
identical, and the8k Release bridge matches16 CLI logits steps. Raw retained.

Added explicit AF_PSPLIT=1 kernel branch: exp2v probability, BF16 hi plus low,
both in denominator and PV. Low plane is4096 B per core. One-key softmax loop
reduces stack/program pressure; existing single-P default stays unchanged.
Llama flash recipe now advertises psplit:true and forces AF_PSPLIT1 in every
attention stream. Runtime parses the flag and explicitly refuses Llama flash
without it, retaining CPU/window guards. No hardware BO argument ABI changes.

Standalone paired-P4352/8192 relative errors4.072e-5/3.997e-5 (old~1.07e-3),
8 repeated dispatches, finite; budgets on4352: program12160/16384, placed
buffer end63584/65536,8192 stack with maxhelperframe2816,3 queued inputBDs and
1 long pass. Stress16/64 and default40-head single-P regressions are running.
Full fresh export running: same long command but output/buildroot names
`llama32-1b-gguf-flashpair8k`, log `llama-gguf-flashpair8k-export.log`.
Candidate runtime rebuilt via reference-pair-build.log / reference-pair-tests.log;
Release bridge target is target/npu-dev/reference-pair-bridge. Preserve old
images as diagnostic evidence. Next: pair model300/8192 and boundary/artificial
checks; contract-refusal check; bridge alignment; only then diagnostic timing.

Paired-P full image finished with every stream rebuilt. Verified psplit:true
and AF_PSPLIT1 in all32 attention build records. Full-model gates:300 passes
(298/300,cos0.999948762,loss2.801876614);8192 passes (8191/8192,
cos0.999950368,loss0.035404766 vsseq0.035401045). Real boundary47 now passes
(47/47,cos0.999951085,loss0.014180733 vsseq0.014176436). Artificial zero
history remains failure (cos0.999850534;46/47;loss8.169178686). Keep opt-in.
Release pair bridge matches all16 long-run CLI logit arrays exactly.

Standalone paired-P stress qk-scale16/64 passes (rel1.976e-5/2.256e-5).
Both normal4352/8192 checks also pass a stricter1e-4 threshold. Fresh default
40-head single-P regression passes (rel0.001077). Full Python1814 pass/64skip/
8deselected;4 native CTests and pair bridge handshake pass. ArtLite K-hilo300
improvescos0.999908882 but still misses argmax gate (292/300), so no
matched-quality acceptance claim; rawquality retained.

CURRENT TIMING: `python utilities/gguf-prefill/paired_survey.py --bridge target/npu-dev/reference-pair-bridge/phlegm-reference-bench --artlite-python target/npu-dev/artlite-env/bin/python --model-dir target/npu-dev/models/llama32-1b --kernels target/npu-dev/kernels/llama32-1b-gguf-flashpair8k --ids-file specs/open-engine/tests/fixtures/issue212/llama-context-8192.ids --out target/npu-dev/survey-flashpair-1k-8k --sizes 1024,4096,8192 --outputs 128 --rounds 3 --prefill-route auto`.
Log survey-flashpair-1k-8k.log, execsession14893. No heavy CPU builds/tests,
artifact hashing or other NPU clients until completion. Next inspect controls,
record final table/provenance and retained failures. Update source ledger,
current status and evidence hashes. No parity or default promotion.

Corrected paired-P survey completed9/9 stable pairs. Prefill770.49/713.63/584.63
and decode34.98/21.70/14.38 tokens/s at1k/4k/8k. ArtLite9611.70/9305.79/7770.51
prefill and64.31/59.99/55.20 decode. Raw paired ratios and source identities in
the owning Linux GGUF report. Development evidence ledger SHA256:
5a26b44e16e83b02fe527b01faa28740b4c69bd64df33a1d1524a84571383ff7
at target/npu-dev/evidence-paired-flash-20261024.json. No accepted parity.
Next profile existing ATTN_NULL1 vs native DX on matched shapes before any
arithmetic/core-layout tuning. Potential reuse of the existing <=5-head G5
score/PV implementation for4-head Llama groups needs explicit opt-in, budgets,
independent precision and model gates; do not broaden measured defaults.

Decode compute ablation completed on synthetic zero-cache1-layer DX shapes,
200 reps, warmups and alternating paired trials bracketed by position0 native
controls. At0 native0.988/null0.987 ms;1024 native1.382/null1.044;4096
native2.596/null1.323 (2/3 valid; one5.6% control failure retained);8192
native4.381/null1.485 (3/3). This is wrong-answer ATTN_NULL timing, not model
rates or a pure KV-DMA ceiling. Raw decode-compute-ablation/{raw,pairs}.jsonl.

Added explicit ATTN_G4CORES1 for Llama32Q/8KV/hd64/RB4 with KV slicing, using
the existing <=5-head implementation at4 heads/core on8 cores, four shared
2-KV-head streams. Measured family defaults and ATTN_KVCORES behavior stay
unchanged. Cache/ownership regression56 tests pass. Initial inherited EXPA2
probe passed300 source quality but failed exact reproduction (299/300 argmax,
1/300 byte-identical). Corrected default probe to original ATTN_EXPA0; that
standalone reproduces all300 logits exactly. Its full8192 sequential run+15
decode completed; exact comparison is still running (g4-exactexp8192.json).
Budget with original exp: program12832/16384; helper stepw frame3968; prototype
placed buffer end57344/65536 with12KB attention stack. Need final-image check.

Complete fresh image export running: `ATTN_G4CORES=1 ATTN_KVSLICE=1 DENSE_GGUF_GEMM=1 DENSE_GGUF_HEAD_Q8=1 DENSE_FLASH_PREFILL=1 AF_LMAX=8192 AF_LONG_STREAM=1 python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/llama32-1b.json --out target/npu-dev/kernels/llama32-1b-gguf-g4-flashpair8k --build-root target/npu-dev/kernel-builds/llama32-1b-gguf-g4-flashpair8k --max-ctx 16384 --force -j 2` (session8266, llama-gguf-g4-flashpair8k-export.log).
Long exact comparison session66328, source g4exactexp8192 vsreal8192-seq-kvslice.
Next: inspect exact result; final image model/bridge checks and controlled
1k/4k/8k survey. No extra NPU work until the long runner exits. Preserve all
known failures and prototype artifacts; no default promotion.

Eight-core original-exp probe reproduces all8192 sequential baseline logits
byte for byte; all16 full-image post-prefill/decode arrays match the previous
paired-flash route too. Probe and final DX instruction bytes are identical;
xclbin metadata differs. Final-image Release bridge matches all16 CLI arrays.
Full Python1815 passed/64 skipped/8 deselected; Rust33 passed/2 ignored plus
release-provenance1 passed. `git diff --check` passes. Every stream rebuilt.

CURRENT TIMING: paired_survey.py with reference-pair-bridge, kernels
llama32-1b-gguf-g4-flashpair8k, out survey-g4-flashpair-1k-8k, sizes1024,4096,
8192, outputs128, rounds3, --prefill-route auto. Same other args as prior
paired-P command; log survey-g4-flashpair-1k-8k.log, session is current tool log.
No heavy CPU work/hash/builds or other NPU clients until complete. Next inspect
controls and update throughput/provenance ledger. Potential subsequent work:
compact BF16-scale GGUF pools (existing decode already narrows f32 scales) and
co-resident Q8 tail, after profiling/budget/precision checks; no implicit promotion.

G4 controlled survey completed9/9 stable pairs. Prefill770.58/717.46/585.79,
decode39.26/30.79/23.67 at1k/4k/8k vs ArtLite9591.56/9302.10/7778.89 and
64.29/60.04/55.10. Raw survey-g4-flashpair-1k-8k/{raw,metadata,summary,throughput}.
Full8k sequential G4 original-exp logits and16 full-image decode arrays match
baseline exactly; no numerical gate relaxed. Known artificial failure retained.

Next representation experiment DENSE_GGUF_COMPACT1: store the BF16 scale values
already used by decode/GEMM, with unchanged codes. New optional dense manifest
finalizer borrows parent5120-byte projection geometry/kernels into the GGUF
view, keeps std_perm_gguf reads, Q8 head, norms and paired flash. Requires
DENSE_GGUF_GEMM1 plusDENSE_GGUF_HEAD_Q8; defaults/rootq4nx untouched.
Prototype weights900726784 B vs1018167296, prefill_extra0; sequential300 and
block300 plusall16 decode arrays are byte-identical. Sequential8192 also
byte-identical (quality/compact-exact8192.json). Native packing regression
checks unchanged codes and nearest-even BF16 scales against adjacent representable
values;2 native CTests pass. Python1817 passed/64 skipped/8 deselected.

Complete fresh compact image export running (session5834): previous G4 export
command plus DENSE_GGUF_COMPACT=1, output/buildroot
llama32-1b-gguf-compact-g4-flashpair8k, log same-name-export.log. It is near the
last GEMMs. No other NPU client active after compactseq8192 comparison. Next:
final compact image metadata/kernel mapping audit;8k block/decode fidelity and
bridge alignment; controlled1k/4k/8k timing after all heavy jobs finish. Preserve
original head/precision contracts and artificial failures. No paid resource used
or authorized; never load credentials for this local work.

### 2026-10-24: ArtLite implementation audit saved

User requested tracing the competing implementation and saving reusable lessons
in both repositories. See `specs/open-engine/artlite-prefill-code-audit.md` and
xdna2-notes `raw/phlegm/2026-10-24-artlite-prefill-code-audit.md`. Read the
upstream planner, Llama runlist, chunk executor, MAC and paired/dedicated-dequant
designs plus its npu-prefill skill. Confirmed fast BFP matrix operands, device
resident intermediate panels/norms/SiLU, library programs in one context,
selective1024-token items, memtile residency and shared dequant. Static selection
for tile-majorQ4_1 T256 is GemmQ; dedicated-dequant is not universal. Individual
speed contributions remain unmeasured; uncompensated BFP remains rejected.

Final compact image: independent8192 gate passed; all8192 prefill arrays and
16 decode arrays byte-identical to validated predecessor. Command:
`python utilities/gguf-prefill/check_bridge_logits.py --bridge target/npu-dev/reference-pair-bridge/phlegm-reference-bench --model-dir target/npu-dev/models/llama32-1b --kernels target/npu-dev/kernels/llama32-1b-gguf-compact-g4-flashpair8k --ids-file specs/open-engine/tests/fixtures/issue212/llama-context-8192.ids --cli-prefix target/npu-dev/quality/compactfull8192/logits --out target/npu-dev/quality/bridge-compactfull8192.json`
returned layer-major, steps16, passedtrue. No compact throughput claim yet.
Next: capture real per-phase plans/timings, isolate matrix vs host/layout and
configuration costs, then prototype accurate fast-matrix arithmetic and device
resident stages. Artificial zero-history failure remains; no parity promotion.

### 2026-10-24: host-load audit requested by user

Inspected raw platform/load/client samples. Earlier survey-g4-flashpair-1k-8k
load1 was0.74–1.88, all9 pairs stable and clients known. Compact survey load1
was7.49–13.52,8/9 valid; failed1k pair had Phlegm decode control drift5.9978%.
Compact medians from valid pairs: prefill741.23/707.45/581.38, decode42.87/32.72/24.89.
These are load-confounded across runs; do not attribute their difference solely
to compact packing. The1k repeat also ran at load7.02–7.89. Current idle load
0.75: launched survey-compact-g4-flashpair-idle-1k with the same paired_survey
arguments, sizes1024,3rounds,128outputs,routeauto. Result pending. Stability
controls detect drift, not steady contention; no unknown NPU clients does not
imply an idle host. Earlier low-load evidence still shows the large gap.

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


## 2026-10-25 continuation: model gates and rotated geometry

Explicit experimental `bfp16_hilo` runtime input format, bounded AVX2 host
packing, parser rejection tests, and four rebuilt projection streams are now
integrated. Runtime patch remains against recorded external base478dbea3,
not a patch series. OPEN-MANIFEST, GGUF-PACK and BFP-HILO-PACK CTests pass.
Image `target/npu-dev/kernels/llama32-1b-gguf-hilo-g4-flashpair8k` uses the
6144-byte f32-scale pool; do not substitute it into the compact5120-byte image.
Independent gates: 300 median cosine .9999477636194,297/300 argmax,
loss2.803070816;8192 median cosine .9999503172159,8191/8192 argmax,
loss .0353892735892. Real47 slice4090..4136 passes (47/47 argmax).
Artifacts: target/npu-dev/quality/hilo{300,8192,-tail47} and gate JSONs.
Known artificial zero-history failure remains open.

Rotated geometry lives in open_kernels/designs/gemm_q4_prefill/compensated64/.
8 feature columns x4 token rows,64x64/core, .text5136B, frame64B,
stack4096B,L1 end60488B. Fresh canonical up build matches tested up stream's
three CDOs, insts.bin and insts.elf byte-for-byte. Full image
llama32-1b-gguf-hilo64-g4-flashpair8k is byte-exact to passing hilo32 at300
and8192 positions plus16 decode outputs. Resident bridge check
`target/npu-dev/quality/bridge-hilo64-8192.json`:16/16 exact,passed.
Paired kernel trials ~2.24–2.26ms versus2.83–2.89ms,~1.26x, but hostload
8.3–8.7 makes this provisional. No accepted end-to-end gain from busy trials.

ArtLite actual native prefill capture completed at
`target/npu-dev/quality/artlite-prefill300`:300 real positions from one padded
512-token dispatch, own Q6_K NPU head including output norm; final captured
row equals untouched forward. Own sequential control also saved. Independent
quality comparisons are the next analysis, not yet an accepted quality match.

Idle timing command is already queued through utilities/gguf-prefill/run_when_idle.py
(shared lock target/npu-dev/.npu-work.lock, wait original training PID1291266,
load<=2 for30s,12h waiting timeout). It launches paired_phlegm_routes.py with
reference native f32-scale image vs candidate hilo64, sizes1024,8192,3 rounds,
4 threads,capacity16384,max-host-load2; inputs from
survey-g4-flashpair-1k-8k/raw.jsonl. Log:
target/npu-dev/survey-native-vs-hilo64-idle-launcher.log. Do not launch duplicate,
stop the training job, or run competing NPU work without this same lock.

Next: finish ArtLite prefill quality comparisons, inspect idle paired engine
results once available, save xdna-engine source triage in owning spec and new
immutable xdna2-notes evidence. Keep kernel-only and engine claims separate.


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

Durable xdna-engine source triage: specs/open-engine/xdna-engine-code-audit.md.
New xdna2-notes raw evidence records model gates, preliminary engine results,
and actual ArtLite prefill quality controls; check.py passed before commit.


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

## 2026-10-25: explicit turbo mode implementation in progress

User requested opt-in higher loss for ArtLite-like reduced precision. Implementing
HH-only compensated64 MAC (default remains HH+HL+LH), retaining input ABI to
bound runtime changes. CLI run/serve --turbo selects fixed complete DIR/turbo
bundle and forces block; missing/mismatched manifests reject, no fallback.
Quality bounds preregistered before hardware measurements: cosine median .9990,
argmax drop <=5pp, NLL <=sequential*1.01. Strict profile unchanged; both report
CPU-reference forward KL. Agent builds all four fresh streams under Dibs build;
root owns assembly/model checks; agents run no NPU jobs. Focused selector and
publication tests pass. Full hardware/model/boundary and timing still pending.

Turbo initial hardware check: Dibs npu+shared flock, check_gemm.py --profile
turbo --activation-format bfp-hilo on fresh qkv3072x2048x256 stream passes
relative source .00920426736 and BF16-operand .00883547471 (limits .02/.02).
Artifact target/npu-dev/turbo64-check-qkv/result.json. No timing claim.
Actual CLI dev binary built; embedded adapter matches existing snapshot bridge
SHA256 ffb4dd5a39b5bc8d7ebdcd5ddb776ad309f25598b2b3980a988293ccded3f6a7.
Pending full300/8192/boundary checks, then live --turbo generation and paired
engine timing under Dibs --measure.

Turbo four-stream family completed:32 cores per stream,text5120B,MAC frame64B,
stack4096B,L1end60488B,identical CDO payloads; source/default and CPU layout
gates saved under target/npu-dev/turbo64-family/*.json. Assembled companion
under kernels/llama32-1b-gguf-hilo64-g4-flashpair8k/turbo; metadata binds both
manifest hashes. Auto retains layer-major; explicit block retained; selected
bridge must confirm batched route. This supersedes earlier 'force block' note.
Turbo300 model gate PASSES:median cosine .9998059036302681,argmax297/300,
NLL2.8056711277026714 vsseq2.803956639988847 (+.06115%),CPU forward KL
mean .0013846764360789265. Strict300 comparison fails cosine, retained in
quality/gate-turbo300-strict.json.35 focused Python tests pass (initial test
command referenced a nonexistent filename and ran zero; corrected full command
includes test_gguf_gemm_quality_gate.py and passes35). Rust turbo tests3 pass.
Long model check currently active; no turbo performance claim yet.

Turbo full8192 aggregate gate PASSES:cosine median .9998245962066122,
argmax8190/8192 (sequential8190),NLL .03545125443710632 versus .03540104505674053
(+.14183%),CPU forward KL mean .00005166273995015996. Real47-token boundary
slice FAILS NLL target: .014680250247053704 versus .014176436293688022
(+3.5539%,absolute+.0005038),47/47 argmax,cosine .9998176737960929. Retain this
known failure without changing preregistered bounds; full-context aggregate
pass is not an all-boundary quality acceptance. Long-K down independent check
passes .00936125 source/.00898937 BF16 vs .02/.02. Artifacts quality/gate-turbo
{300,8192,-tail47}.json and turbo64-check-down/result.json under target/npu-dev.
Actual target/debug/phlegm run --turbo generated8 tokens successfully with
matching bridge provenance (turbo-cli-smoke.log); auto retains batched layer-major.

Turbo bridge check:16/16 outputs byte-exact to CLI after8192 prompt positions
(quality/bridge-turbo8192.json). Paired1K standardcompensated64 vs turbo
started through Dibs --measure npu, currently waiting for CPU holder
codex-a53434a0b786/broad-more through~19:35. Exec session25407 will launch
automatically after the reservation clears; out survey-hilo64-vs-turbo-dibs,
sizes1024,rounds5,threads4,maxctx16384,maxhostload2.0,same saved inputs and
snapshot bridge. Do not start duplicate or interfere with its future window.
No turbo timing result yet. New xdna2-notes raw evidence saves aggregate versus
boundary quality lesson and hashes, retaining the failed bound explicitly.
- `target/npu-dev/quality/gate-turbo300.json`: `760f11e68d202130fafcf8da90e66f49105e728b81b0f3fc8d2110ed84d7baed`
- `target/npu-dev/quality/gate-turbo8192.json`: `40456cccceace252efe1ffbf30f9e0d6f9437014d44b6598558cfb75fa862dc4`
- `target/npu-dev/quality/gate-turbo-tail47.json`: `c838f1c64982ef3909032aff35b295a19a7b69d2532ec3da6f13699e66e5fe67`
- `target/npu-dev/quality/bridge-turbo8192.json`: `64a1b66b96cc9bb141947343a3bee61787f840cc8b76c4016d9677fefbf017ed`
- `target/npu-dev/turbo64-family/family.json`: `b7b308b56e752be2b41ae682ad8333c592c9043f2af3eda451045c8d5890c3fd`


### Turbo timing waiter cancelled; current state

The attempted --measure runner reserved machine/build while waiting for another
CPU session until19:35. To avoid holding the machine while waiting, released
ONLY our measurement reservation with dibs release --measure, then sent SIGINT
to our verified waiting launcher PID2471000 (exec session25407 exit130).
No benchmark child started and no other session's holds/jobs were changed.
The separate wait --measure exited FREE. There is NO pending turbo benchmark.
This supersedes the queued/automatic-launch notes immediately above.

Next command once resources are available (fresh out path, development engine
comparison; require CLEAN before/after and paired controls):

```bash
source target/npu-dev/env.sh
dibs run --measure npu --label 'Turbo versus compensated1K engine comparison' --eta 10m -- \
  flock target/npu-dev/.npu-work.lock python utilities/gguf-prefill/paired_phlegm_routes.py \
  --bridge target/npu-dev/reference-hilo-snapshot-bridge/phlegm-reference-bench \
  --model-dir target/npu-dev/models/llama32-1b \
  --reference-kernels target/npu-dev/kernels/llama32-1b-gguf-hilo64-g4-flashpair8k \
  --candidate-kernels target/npu-dev/kernels/llama32-1b-gguf-hilo64-g4-flashpair8k/turbo \
  --inputs-jsonl target/npu-dev/survey-g4-flashpair-1k-8k/raw.jsonl \
  --out target/npu-dev/survey-hilo64-vs-turbo-dibs \
  --sizes 1024 --rounds 5 --threads 4 --max-ctx 16384 --max-host-load 2.0
```

Current tests:36 focused Python checks (35 earlier plus final shape-rejection
case);3 Rust turbo tests; live CLI generation; bridge16 exact. Final git diff
--check passes. Turbo remains experimental with the explicit boundary NLL
failure; no speed/ArtLite parity claim. Lessons saved in xdna2-notes f288faf,
with a separate follow-up noting cancellation of the initially queued timing.

### 2026-10-25: clean turbo timing requeued at user's request

Exec session9838 runs the exact retry command above after
`dibs wait cpu gpu npu network runpod-ternary` returns FREE. Waiting does not
hold a measurement reservation. No benchmark has started yet; current holder
codex-671d23710056 is measuring stock Q4 versus ternary DFlash2 through~19:38;
the broad_c remote-training session also declares network/runpod activity.
The launcher will automatically request Dibs --measure and run five paired1K
trials when clear. Output survey-hilo64-vs-turbo-dibs; benchmark-phase log
target/npu-dev/survey-hilo64-vs-turbo-dibs-launcher.log (created after wait).
Do not duplicate this waiter. Require initial/final CLEAN and paired controls;
retain known boundary quality failure alongside any performance result.


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

Compact-A turbo producer freshly rebuilt all4streams; shared CDO checks pass.
Per-core text4640B,frame64B,stack4096B,L1end55880B (vs old60488B),
external high/low input ABI unchanged. Independent3072x2048x256 check passes
and output is BYTE-EXACT to original HH kernel (source .0092043,BF16 .0088355).
Fresh unpromoted image at kernels/llama32-1b-gguf-hilo64-high-only-probe/turbo;
full8192+16decode dump underway in exec session87191,Dibs npu,shared flock.
No timing of compactproducer yet. Source pipeline audit saved in owning
specs/open-engine/artlite-prefill-code-audit.md: actualT1024 still repeats
dequant every256token sweep; paired dequant halves conversion duplication
(2NKvs4NK),not externaltraffic. Device gate/up->SiLU/pack->down opportunity
is derived20.5MiBhostsyncpayload/block/layer,not measuredlatency.


Compact ABdepth1 fullmodel8192+16decode equality PASSES, aggregateSHA
89c0f4961c2114439981eaf91b43737028ef5000934fb208574f4843a2b48c20.
quality/turbo-high-ab-exact8192.json retains inherited boundary failure.
ABW2 all4freshstreams fit actual58952B L1,4752Btext,64Bframe,4096Bstack;
source hashes equalABdepth1,core5BD,membudgetmax24BD(16lowpool/8high),
6S2MM/3MM2S. IndependentdownK8192 output byteexact to priorHH; full8192
check active session72208. Claimed15GB quality-artifact disk growth viaDibs
for repeatedfullcaptures; release outstandingclaim after all checks complete.

Standalone ffn_silu_pack v1 ranNPU: relativeerror .00166343 againstdouble/FP32
passes .02 operatorbound, butexactCPUcandidate bytes FAIL(9lowplanebytes).
All mismatches occur within0–3FP32ULPs ofBF16 roundingmidpoints; nohighplane
or exponent mismatch. This suggests prepack arithmetic drift, NOT yetproven.
Originalfailedfixture retained. Newdiagnostic must expose actualdeviceF32
andprove independenthostencoder(deviceF32)==devicepacked, with separate
SiLUapproximationerror gates; no retroactive relaxation ofv1bytegate.
No runtimeintegration orperformanceclaim. CurrentCPU studies prevent clean
new timing; do not rununisolatedbenchmarks or replace existingturbo bundle.


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


### Current handoff after correctness gates, 2026-10-26 03:19 local

ABW2 partial300+16decode exact,full8192+16decode exact,andresidentbridge16/16
exact all PASS. All three new variants inherit prior turbo quality failures.
Standalone SiLU diagnostic actualdevicevalues packing has0byte mismatches;
diagnosticpacked equalsoriginalpacked,actualF32relative errors1.214261629e-6
(double) and1.213874056e-6(independentFP32). Originalstrict9bytefailure retained.
Evidence parallel-pipeline/silu-pack-fixture-v2/diagnostic-check.json; no fullmodelintegration.
36focused Pythonregressions anddiffcheckpass. Dibs diskclaim released; no
own CPU/build/NPU/measurement holds or pendingbenchmarks. OtherCPUstudies
consume~66% hostCPU; clean newtiming must wait. Notes committed xdna2-notes
7bbf10a (negativecleanresult) andbebe33d (compact/encoderlessons). MainPhlegm
newsource remains local/uncommitted alongside preserved unrelatedwork.

Next clean timing: use paired_phlegm_routes.py command above, freshout
survey-compensated-vs-turbo-high-ab-w2-dibs, --candidate-kernels
 target/npu-dev/kernels/llama32-1b-gguf-hilo64-high-ab-w2-probe/turbo,
--reference-kernels original llama32-1b-gguf-hilo64-g4-flashpair8k,
--reverse-init, sizes1024,rounds5,threads4,maxctx16384,maxhostload2.
Run under Dibs --measure npu +sharedflock only once other work clears; require
CLEAN before/after andcontrols. Then compare A-only/AB/W2 stages and swapped
initialization to attribute any gain. Do not replace installedturbo image or
claim speed from smaller buffers. DeviceFFN runtimechain is later work; first
vectorize/measure standalone pluscontextfloor, then addBOresidentinput/output
with explicitformat/mode and rerun independentwholemodel/boundarygates.

## 2026-10-26: separate native image model request

The user requested Qwen Image 2.1 and FLUX 3 Image Klein for native NPU image
inference via CLI and API. This is tracked in `.plans/native-image-models.md`
and does not change the ArtLite performance route. The temporary Diffusers
CLI wrapper was removed after the user specified NPU execution. Public Qwen
and FLUX prompt-upsampler downloads are running under Dibs, with a 110 GB disk
claim; gated FLUX model files await CLI authentication. No image-model NPU
correctness or speed claim exists. Preserve the existing performance evidence
and continue its next clean timing as above.


### Ready-work commit checkpoint, 2026-10-26

User requested committing ready changes. Reviewed and committed experimental
turbo kernels/quality gates, standalone SiLU diagnostics, residentgeneration
with explicit turbo selection, modelconfig guards, standaloneGPTOSS/Gemma
components, and duplicateheader fix in separate units. During review fixed
SSE splitUTF8 corruption with native stateful decoding and added BOS/EOS
chattemplate context; fullRust packages44passed,2prerequisite checksignored.
Also fixed gate/router checkers' fail-open NaN/count/probability validation;
38focused tests pass. Final combined Python source regressions106passed,
no skips. Coverage matrix reproduces byte-for-byte from pinned local registry.
Diff whitespace checks pass. No newNPU orperformance trials in this commit
pass; all existing acceptance limits and failures stay documented. Generated
artifacts/locks/stagingcopies and an active native-image planning file are
left outside the readywork commits. No reset, cleanup, push or paid action.
