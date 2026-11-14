# Dense GGUF carriage: P5 S1 offline implementation

2026-11-09. Development evidence on Strix Halo; **no NPU execution or performance
claim**. D1 uses int8 for Q6_K; D2 uses the PR 209 + Nanbeige lineage. The serving
runtime and bridges were not modified. This is part of OPEN-GGUF-DENSE.

## Contract and remaining stage boundary

`std_perm_gguf` preserves Q4_0, Q4_1 and Q4_K codes. Q4_K's scale and offset
are `float32(d * scale)` and `float32(-dmin * min)`, then one RNE bf16 rounding
for 5120-byte chunks. The 6144-byte twin stores those f32 products directly.
There is no intervening fp16 rounding or min/max fit. Higher-precision tensors
in a four-bit slot fail with a request to use `q8_perm`.

`q8_perm` now accepts GGUF matrices as well as its existing q4nx input. It
writes the existing **5120-byte padded 16-row half-tiles (10 bpw)**. Q8_0 codes
are preserved, including -128; the scale is rounded to bf16. Q6_K is decoded
by the runtime, then quantized per 32 values using bf16(amax/127), division by
that rounded scale, and ties away from zero, clipped to [-127,127]. Llama Q/K
rows are unpermuted; Qwen3 rows are left in their source order. Padding is zero.

The head keeps its prior `lmhead_q8` implementation, including the subnormal
fix. Its int8 rounding uses **f32 reciprocal multiply**, whereas the new
projection packer uses division. These differ at some half-step boundaries.
The independent oracle models both explicitly rather than changing the
existing Q4_1 model's head. The native and Python synthetic gates include
zero/subnormal scales, signed Q6_K scales, missing Q4 code extrema, Q/K row
order, and Q8_0 codes/scales. The native helper also dumps vectors through
`put` for independent norm checks.

Dense head/compact-view switches are no longer Llama-only. Existing default
Llama source generation is unchanged. The deriver recognizes K-quants and
refuses unknown projection types. `ModelSpec.from_gguf_metadata(...,
tensor_types=...)` derives a complete per-tensor selector, including mixed
layers; an incomplete map is refused. The old one-format-per-role API still
refuses mixed roles, rather than silently returning q4_1.

**S2 remains necessary:** `pack_plan` deliberately still refuses selector-bearing
specs. The compiled V1 probe does not implement V0/V1 per-layer instruction-stream
selection or q8 block prefill. No runnable Q4_K_M manifest is delivered or
implied. A V0 manifest cannot load this mixed file: Q6_K in its q4 slot is now
refused. The full V0 kernel set is usable for the existing Q4_1 regression path;
V1 is a compile artifact. G2-G5 on Q4_K_M cannot run until S2, even in a quiet
hardware window. No manifest ABI, DIT7, or probability-hi/lo arithmetic changed.

The new reference supports llama and qwen3 forwards, tied/untied heads, Qwen3
q/k norms, both rotary row orders, and GGUF `rope_freqs.weight`. Other forwards
are explicitly refused. Its `--carriage` mode changes weight/norm carriage;
activations remain fp64. It is not a replica of device activation arithmetic.
Q5_0/Q5_1 carriage is recognized by the selector but this runtime lineage does
not yet implement their source dequantizers; it refuses them at packing.

## Measured offline evidence

Source: unsloth/Llama-3.2-1B-Instruct-GGUF at
`b69aef112e9f895e6f98d7ae0949f72ff09aa401`, file SHA256
`3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`.
Evidence root: `/home/cyrus/code/phlegm/target/npu-dev/p5/s1/`.

- All **112 projections + the tied head + 33 norms** pass: C++ source
  dequantization equals gguf-py bit for bit; packed values equal the specified
  carriage bit for bit. Q4 codes and scale/offset planes are checked separately.
  Norms equal RNE bf16 (this file's norms are already exact).
- 96 Q4_K projections: relative L2 **0.00515451–0.00574500**. Sixteen Q6_K
  projections plus the head: **0.00535662–0.00614515**, head **0.00554405**.
  All are below 0.008; zero tensors are refitted to four bits.
- Packed projection/head payload: **976,388,096 B**; norms: **135,168 B**.
  This is the sum of inspected tensor payloads, **not resident RSS or a model
  load measurement**. Allocation rounding, context buffers and mapped source
  weights are not included. Warm-load time and resident size remain unmeasured.
- NPU-like fp64 control against the existing llama.cpp CPU top-20 reference:
  **491/491** argmax, median partial KL **0.000267979**. Its 55-token prefix:
  **55/55**, **0.000954457**. The prefix is not an independent short prompt.
  The source reference is the S0 `llama-cpu-top20.json`, not a new Vulkan control.
- Fresh independent 70-token short prompt against llama.cpp build 12015,
  a34de8618 (CPU): carriage **67/70** argmax, median partial KL **0.00212700**;
  its 55-position prefix is **52/55**, **0.00196015**. This is a material open
  quality result, not a G2 pass. The source-weight fp64 control is **69/70**,
  median partial KL **0.000992698** (55-prefix: **54/55**, **0.000881594**).
  Carriage loses two additional argmax positions; its KL is 2.14x that source
  control. The required Vulkan noise-floor control is absent.
- Five existing Llama generated C++ sources: byte-identical before/after;
  hashes in `source-identity.json`, directories `baseline-before/`, `baseline-after/`.
- Full V0 export: **47 sets**, **5,397,484 B** of xclbin + instruction payloads,
  capacity 16384 in its manifest. V1 decode probe: xclbin **173,330 B**,
  instructions **61,024 B**. All are outside the worktree under `p5/kernels/`.
- V1 main-core program **12,016 / 16,384 B**. Placed buffers **48,128 B** plus
  reserved stack **6,144 B** = **54,272 / 65,536 B**; conservative direct-call
  stack bound **1,280 B**. Shim peaks: 9 BDs/column, 4 outstanding starts/channel;
  11 fill and 14 drain channels. The inherited four 32-byte scalar `pb` buffers
  remain (the audit's blanket `BUFFER_ALIGNMENT` says false); DMA/main buffers
  are aligned. This reproduces the [fit probe](dense-tensor-fit.md), not hardware
  high-water usage. Toolchain: MLIR-AIE 1.4.2, Peano 21.0.0 / c9c5ecb7.
- Focused pytest: **168 passed, 2 skipped** (superseded catalogue-point probe,
  unavailable external installer source). Native synthetic subset: **23 passed**.
  `gguf_pack_test`: **GGUF-PACK OK**, including the new Q4_K/Q6_K cases.
  Rust core: **123 passed, 7 ignored**, doc tests 0, with serial test execution.

## Runtime handoff

Runtime copy: `/home/cyrus/code/phlegm/target/npu-dev/p5/runtime/`, made by
`cp -a --reflink=always` from `nanbeige-runtime/`. Bridge and diagnostic CLI:
`p5/bridges/dense/{phlegm-reference-bench,open_qwen36_cli}`. The bridge's
`.build.json` records compiler, source and binary hashes.

`open_kernels/upstream-patches/dense-gguf-carriage-runtime.patch` applies after
`linux-runtime-pr209.patch` and `nanbeige-runtime.patch`, not to the beccd46
lineage. `sources.json` records base/result hashes for both changed files and
the patch hash. A fresh two-file baseline plus the patch equals the runtime
copy byte for byte (`s1/patch-roundtrip.json`). No new crate/download was needed.

## Reproduction commands

From the worktree, with these abbreviations:

```sh
D=/home/cyrus/code/phlegm/target/npu-dev/p5
PY=/home/cyrus/code/phlegm/target/npu-dev/env/bin/python
GP=/home/cyrus/.cache/tmp-relocated/pair-cache-llama/gguf-py
MODEL="$D/src/llama32-1b-q4km/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
```

Native build/test and independent real-file checks:

```sh
dibs run build --label 'P5 S1: final extended packer including norms' --eta 5m -- bash "$D/s1/build-pack.sh"
dibs run cpu --label 'P5 S1: final all-tensor carriage gate' --eta 10m -- env OPENBLAS_NUM_THREADS=1 "$PY" -I open_kernels/model/dense_gguf_check.py --gguf "$MODEL" --packer "$D/s1/gguf_pack_test" --gguf-py "$GP" --out "$D/s1/pack-results.json" --head
# The recorded matrix/head and norm runs were separate; this checks only norms:
dibs run cpu --label 'P5 S1: all norm vectors' --eta 2m -- env OPENBLAS_NUM_THREADS=1 "$PY" -I open_kernels/model/dense_gguf_check.py --gguf "$MODEL" --packer "$D/s1/gguf_pack_test" --gguf-py "$GP" --out "$D/s1/norm-results.json" --norms-only
dibs run cpu --label 'P5 S1: final NPU-like CPU control' --eta 10m -- env OPENBLAS_NUM_THREADS=4 "$PY" -I open_kernels/model/dense_gguf_reference.py --gguf "$MODEL" --ids-file "$D/long.ids" --gguf-py "$GP" --out "$D/s1/reference" --carriage --logits-file "$D/s1/carriage-logits.bin"
dibs run cpu --label 'P5 S1: final control scores' --eta 2m -- env OPENBLAS_NUM_THREADS=1 "$PY" -I "$D/score.py" "$D/llama-cpu-top20.json" "$D/s1/carriage-logits.bin"
```

The exact native compile is in `s1/build-pack.sh` (C++17, `-O2`,
`-ffp-contract=off`, pools/manifest/GGUF/q4nx sources, no XRT). Bridge delivery:

```sh
dibs run build --label 'P5 S1: delivery bridge and CLI provenance' --eta 10m -- bash "$D/s1/build-delivery.sh"
# That script runs this requested builder, then builds cli.cpp against its objects:
"$PY" benchmarks/reference/build.py --reference-root "$D/runtime" --out-dir "$D/bridges/dense" --xrt-include /home/cyrus/code/phlegm/target/npu-dev/xrt/include --xrt-lib /home/cyrus/code/phlegm/target/npu-dev/xrt/lib/libxrt_coreutil.so.2 --cxx /usr/bin/g++
python "$D/s1/make-patch.py"
```

Kernel compilation (the preparer requires a fresh output directory when repeated):

```sh
"$PY" -I utilities/dense_fit_probe.py --gguf "$MODEL" --out "$D/kernels/v1"
source /home/cyrus/code/phlegm/target/npu-dev/env.sh
dibs run build --label 'P5 S1: V1 decode compile' --eta 10m -- env OPEN_KERNELS_SPEC="$D/kernels/v1/spec.json" DENSE_GGUF_LAYER=0 DX_ONECTX=0 OFLM_KEEP_FAILED=1 python "$D/kernels/v1/source/open_kernels/build_design.py" "$D/kernels/v1/source/open_kernels/designs/dense/dx.py" "$D/kernels/v1/dx"
dibs run build --label 'P5 S1: V0 complete compact kernel set' --eta 20m -- env DENSE_GGUF_GEMM=1 DENSE_GGUF_HEAD_Q8=1 DENSE_GGUF_COMPACT=1 python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/llama32-1b.json --out "$D/kernels/v0" --build-root "$D/kernels/v0-build" --max-ctx 16384 --force -j 2
"$PY" utilities/core_sizes.py "$D/kernels/v1/dx" --symbols
"$PY" -I /home/cyrus/code/phlegm/target/npu-dev/kernel-builds/p5-s0-dx/audit.py "$D/kernels/v1/dx"
for elf in "$D"/kernels/v1/dx/final.prj/elfs_main_core_*/*.elf; do "$PY" -I /home/cyrus/code/phlegm/target/npu-dev/kernel-builds/p5-s0-dx/stack_audit.py "$elf"; done
"$PY" -I "$D/s1/check-source-identity.py"
diff -ru "$D/s1/baseline-before" "$D/s1/baseline-after"
```

Focused regressions and required Rust gate:

```sh
PHLEGM_DENSE_PACK_TEST="$D/s1/gguf_pack_test" GGUF_PY="$GP" dibs run build --label 'P5 S1: derivation and native regressions' --eta 5m -- "$PY" -m pytest -q specs/open-engine/tests/test_dense_gguf.py specs/open-engine/tests/test_dense_tensor_quant.py specs/open-engine/tests/test_quant_mixed.py specs/open-engine/tests/test_quant_q8.py specs/open-engine/tests/test_qwen3_dense.py specs/open-engine/tests/test_spec_derive.py specs/open-engine/tests/test_gguf_twin.py specs/open-engine/tests/test_build_cache.py
dibs run build --label 'P5 S1: serial Rust core regression' --eta 5m -- env PHLEGM_NPU_LOCK=/tmp/phlegm-p5-s1-mock.lock cargo test -p phlegm-core --offline -- --test-threads=1
git diff --check
```

Independent short-prompt control (70 positions; the corpus preparer also writes
three greedy and real-prefix boundary/capacity inputs):

```sh
"$PY" -I "$D/s1/prepare-night.py"
dibs run cpu port-8402 --label 'P5 S1: independent short CPU oracle' --eta 5m -- bash "$D/s1/short-cpu.sh"
dibs run cpu --label 'P5 S1: independent short carriage control' --eta 5m -- env OPENBLAS_NUM_THREADS=4 "$PY" -I open_kernels/model/dense_gguf_reference.py --gguf "$MODEL" --ids-file "$D/s1/night-inputs/greedy0.ids" --gguf-py "$GP" --out "$D/s1/short-reference" --carriage --logits-file "$D/s1/short-carriage-logits.bin"
dibs run cpu --label 'P5 S1: independent short score' --eta 2m -- env OPENBLAS_NUM_THREADS=1 "$PY" -I "$D/score.py" "$D/s1/short-cpu-top20.json" "$D/s1/short-carriage-logits.bin"
dibs run cpu --label 'P5 S1: independent short source control' --eta 5m -- env OPENBLAS_NUM_THREADS=4 "$PY" -I open_kernels/model/dense_gguf_reference.py --gguf "$MODEL" --ids-file "$D/s1/night-inputs/greedy0.ids" --gguf-py "$GP" --out "$D/s1/short-source" --logits-file "$D/s1/short-source-logits.bin"
dibs run cpu --label 'P5 S1: short source score' --eta 2m -- env OPENBLAS_NUM_THREADS=1 "$PY" -I "$D/score.py" "$D/s1/short-cpu-top20.json" "$D/s1/short-source-logits.bin"
bash -n "$D/s1/night-q41.sh"  # syntax check only; never run here
# Later, in the authorized quiet hardware window:
bash "$D/s1/night-q41.sh"
```

## Failed attempts and corrections

- Initial native compile omitted the harness include:
  `fatal error: 'stream_patch.hpp' file not found`. Added the include and rebuilt.
- Initial bridge command used the default compiler resolved through ccache:
  `/usr/bin/ccache: invalid option -- 'O'`. Repeated the same build with explicit
  `--cxx /usr/bin/g++`; passed. `bridge-build.log` was replaced by the successful
  rerun; this paragraph retains the failure.
- First Python pack oracle treated negative floating zero as an int8 code sign:
  `AssertionError: blk.0.attn_v.weight: packed values differ from carriage oracle (0 values)`.
  Integer codes now canonicalize zero; the raw source still has a bitwise check.
- Head division oracle failed:
  `AssertionError: token_embd.weight: packed values differ from carriage oracle (713 values)`.
  Preserved runtime arithmetic; corrected the oracle to reciprocal multiply and
  lround semantics. Full head and synthetic tests pass.
- First focused suite: `1 failed, 144 passed, 2 skipped`; the old GGUF-view test
  allowed the q4_1 view only for Llama. It now checks the shared dense contract.
  Subsequent suites: 162/2, then 168/2 after native fixtures; final 168/2.
- Required initial `dibs run build --label 'P5 S1: Rust core regressions' --eta 10m -- cargo test -p phlegm-core --offline`:
  `120 passed; 1 failed; 5 ignored`. Failure in
  `bench::reference::generation_tests::resident_bridge_resets_and_stops_across_requests`:
  The panic text was:
  ``called `Result::unwrap()` on an `Err` value: "NPU lock ownership was lost; discard the run"``.
  A serial rerun with a task-local **mock** lock passed. No NPU was acquired.
- Dibs pressure blocked the diagnostic-CLI launch at 90% CPU. `dibs wait --pressure`
  returned `FREE: CPU and memory have been under the limits for 30 s`; the exact
  blocked command was retried unchanged. No override was used.

- Whitespace checking the previously untracked patch found three blank context
  lines (`dense-gguf-carriage-runtime.patch:222: trailing whitespace.`, likewise
  lines 224 and 266). Normalized those context lines, regenerated its hash, and
  repeated the apply/byte-identity round trip. Tracked and new-file whitespace
  checks now both pass.

## Untested hardware and follow-up

No G2-G6 acceptance, load-time limit, resident RSS, Qwen3 model run, or published
benchmark follows from these checks. The historical artificial zero-history
window failure remains open. Vulkan controls were not run. Night inputs and the
exact acquisition commands are in `p5/s1/night-inputs/` and `p5/s1/night-q41.sh`;
that script was syntax-checked only, never executed. It uses the existing Q4_1
model and existing compact flashpair kernels, the new isolated CLI, real-prefix
boundary cases, three independent greedy prompts, and both sequential/block
routes. Evaluate captured logits against same-file llama.cpp CPU and Vulkan
controls using the owning plan's G2-G5 limits before accepting them. The
16320-token input repeats the existing 8192-token fixture: it exercises capacity,
not retrieval. Long captures require an explicit Dibs disk claim.

For Q4_K_M, first implement S2's runtime variants and q8 block route, export a
complete manifest, then repeat those commands with that manifest and the
published source. There is deliberately no command that runs the V1 fit image
as though it were a complete mixed model.
