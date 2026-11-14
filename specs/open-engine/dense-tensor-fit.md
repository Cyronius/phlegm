# Dense GGUF per-tensor fit probe

2026-11-09, P5 S0. **The Llama 3.2 1B V1 compile fits physical memory and DMA
limits.** This is a compile-only development result. No NPU execution, model
quality, performance, compact-q8 layout, or runtime variant dispatch is accepted
by this result. The S0 census and CPU carriage results remain in their existing
handoff; its independent short-prompt / Vulkan control gap remains open.

## Selector contract

`ModelSpec.tensor_quant` optionally maps actual GGUF tensor names to carriage,
for example `{"blk.3.attn_v.weight": "q8", ...}`. The map includes the seven
projections of every layer being probed. `recipes/dense_precision.py` derives it
from the file's tensor types, keeping Q4_0/Q4_1/Q4_K at q4_1 and assigning Q6_K,
Q8_0 and the other explicitly supported higher precision types to q8. Unknown
projection types, missing projections, invalid names and out-of-range layers
are refused. Head and norm tensors are outside this selector.

`DENSE_GGUF_LAYER=<index>` explicitly selects the layer for the recipe, pool
layout, generated C++ and `dx.py` core/runtime sequence. This compile probe
supports q8 on `attn_v` and/or `ffn_down`, with every other projection at q4_1;
other q8 projections are refused pending their own gates. The q4 GEMV pair and
q8 output-band GEMV coexist; the q8 up/gate scratch entry is not generated.
The q4 and q8 projections share the existing weight FIFO, channels and context.
The combined head stream requires `DX_ONECTX=0` for this probe.

Without `tensor_quant`, old serialization/spec hashes and generator output
remain unchanged. With it, spec/kernel/quant hashes include the selector,
`DENSE_GGUF_LAYER` enters the probe build key, and the q8 header is a build input.
The emitted MLIR was also checked for exactly 16 q8 output calls, 24 q4
output calls and 16 q4 scratch calls across the eight main cores, with no q8
scratch call. The conservative hash currently includes layer placement; S2 still owns
geometry/variant-only caching and V0+V1 runtime stream selection. `pack_plan`
refuses selector-bearing specs and block prefill is disabled: the new field
cannot silently export a runtime manifest that interprets its pools incorrectly.

## Measured compile result

Source: `unsloth/Llama-3.2-1B-Instruct-GGUF`, revision
`b69aef112e9f895e6f98d7ae0949f72ff09aa401`, source SHA256
`3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`.
Source directory: `/home/cyrus/code/phlegm/target/npu-dev/p5/src/llama32-1b-q4km/`.
The header selected q8 for both `attn_v` and `ffn_down` at layers
`0, 1, 4, 7, 10, 13, 14, 15`; the compiled representative is layer 0.
Geometry: hidden 2048, FFN 8192, 32 heads, 8 KV heads, head dimension 64,
8 main cores, 4096-token capacity. The preparer checks source/spec geometry.

Evidence root (all generated files stay outside this worktree):
`/home/cyrus/code/phlegm/target/npu-dev/kernel-builds/p5-s0-dx/`.
`final-v1/` contains the frozen source snapshot, full tensor selector, selection
record, GGUF hash, build log, compiled image, instructions, ELF files, placed
MLIR, symbol sizes, memory/DMA audit and disassemblies. The earlier `v1/` build
has the same resource figures. Repository base: `7742985b53e464a1a8171321fe4e33281905ec8a`,
with the uncommitted worker changes. Toolchain: mlir-aie 1.4.2, Peano Clang
21.0.0 (`c9c5ecb725fc8c765e4b687356e6ec1e54da7a0e`), target npu2.
No hardware or power-mode comparison is made. Final artifact SHA256:
`final.xclbin` = `c9b7af59201b96d9772b7aeb13fb5ea4e77d15aafa261d1803d7622ad786219b`;
`insts.bin` = `2a46971c2a12ea5a2ef6a3ff2fff14da670b3bfb1c8ba470840f55bce394f401`.
The final snapshot's seven changed/build-entry sources match the worktree byte
for byte (`final-v1/source-sha256.json`).

| Resource | Measured V1 | Budget / margin |
| --- | ---: | --- |
| Main-core program, each of 8 | 12,016 B | 16,384 B; 4,368 B free |
| Attention program, maximum | 12,016 B | 16,384 B; 4,368 B free |
| Norm program | 5,552 B | 16,384 B; 10,832 B free |
| Main-core placed buffers | 48,128 B | table 18,432; weights 20,480; x 8,192; y 512; scratch 512 |
| Main-core reserved stack | 6,144 B | direct-call conservative bound 1,280 B; largest frame 576 B |
| Main-core buffers + stack / placed end | 54,272 B | 61,440 B planning ceiling; 65,536 B physical; 11,264 B physical free |
| Norm buffers + stack | 34,816 B | 65,536 B; stack bound 512 B |
| Attention buffers + stack, maximum | 32,800 B | 65,536 B; stack bound 3,328 B, largest frame 2,944 B |
| Weight FIFO element / depth | 10,240 B / 2 | unchanged from baseline |
| Core DMA channels, main | 2 S2MM + 1 MM2S | 2 input + 2 output maximum |
| Core DMA channels, attention maximum | 1 S2MM + 2 MM2S | 2 input + 2 output maximum |
| Core BDs, maximum | 13 (attention); 6 (main) | 16 per core |
| Memtile DMA / BDs | 0 / 0 | no memtile ring in this probe |
| Shim channels | 11 MM2S, 14 S2MM | at most 2 of each per column, 16 of each across 8 columns |
| Shim live BDs, maximum per column | 9 | 16 |
| Shim outstanding starts, conservative maximum | 4 per channel | 4 |
| Runtime buffer arguments / columns | 6 / 8 | 8 / 8 |

Placed buffer ends already include the stack at address zero: adding the stack
again to those ends would double count it. Stack bounds use ELF function symbols
and direct calls/tail jumps in `llvm-objdump -d`; they are static bounds, not
hardware high-water measurements. All main-core buffers and all DMA buffers have
64-byte aligned addresses and sizes. The four inherited attention `pb` metadata
scratch buffers are 32 B (scalar i32 fields), identical to the baseline; the
blanket 64-byte recommendation is not asserted for those scalar buffers.

Two design assumptions need correction before S2:

- The existing q8 GEMV consumes **5120-byte padded 16-row half-tiles**, 10 bpw,
  rather than compact 8704-byte 32-row chunks at 8.5 bpw. Its FIFO therefore
  remains the q4 FIFO's size; V1 adds no data-memory allocation. This build
  establishes the padded layout's fit. Compact q8 needs a separate stream-law
  design and fit check. The plan's 8.5-bpw cost/rate estimates do not describe
  these artifacts. The hi+lo fallback is not needed for this geometry's fit.
- `catalogue.LIMITS` / plan section 6 lists 11 shim drains. Both the existing
  baseline and V1 allocate 14 (8 main y, 4 attention og, attention output, norm
  output). The physical per-column limit is respected, and V1 adds no channel
  or BD. This pre-existing planning discrepancy is recorded without changing
  the global catalogue limit. A hi+lo fallback would retain these same drains.

## Byte identity and checks

The five Llama 1B generated C++ sources were generated **before editing** into
`baseline-before/`, then regenerated after the change into `baseline-after/`.
`diff -ru` exits 0 with no output. Per-file SHA256 checks are retained in
`source-identity.json`. This is source identity, not a claim that a new complete
binary build has an identical build stamp. The V0-selector regression also
compares its generated source dictionary against the original all-q4 recipe.

Focused Python checks: 145 passed, 2 skipped (a superseded catalogue-point
probe and the unavailable external OFLM installer source). The open_kernels tests with `python_files=test_*.py`: 14 passed,
5 skipped (gguf-py unavailable). The literal `pytest -q open_kernels` invocation
fails collection with 32 existing duplicate `make_test` module/import errors;
those scripts generate standalone kernel test data rather than pytest tests.
Required Rust core suite: 123 passed (121 unit + 2 integration), 7 ignored,
0 failed; doc tests 0. `git diff --check` passes.

No new crate or download was needed. No NPU or external runtime was used.

## Exact reproduction commands

From this worktree; `B` and `PY` below are task-specific abbreviations only:

```sh
B=/home/cyrus/code/phlegm/target/npu-dev/kernel-builds/p5-s0-dx
PY=/home/cyrus/code/phlegm/target/npu-dev/env/bin/python
# Use a new empty output directory; final-v1 already exists as evidence.
$PY -I utilities/dense_fit_probe.py \
  --gguf /home/cyrus/code/phlegm/target/npu-dev/p5/src/llama32-1b-q4km/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --out "$B/final-v1"
source /home/cyrus/code/phlegm/target/npu-dev/env.sh
dibs run build --label 'P5 S0 fit probe: final V1 dx' --eta 10m -- \
  env OPEN_KERNELS_SPEC="$B/final-v1/spec.json" DENSE_GGUF_LAYER=0 DX_ONECTX=0 OFLM_KEEP_FAILED=1 \
  python "$B/final-v1/source/open_kernels/build_design.py" \
  "$B/final-v1/source/open_kernels/designs/dense/dx.py" "$B/final-v1/dx"
$PY utilities/core_sizes.py "$B/final-v1/dx" --symbols
$PY "$B/audit.py" "$B/final-v1/dx"
for elf in "$B"/final-v1/dx/final.prj/elfs_main_core_*/*.elf; do
  $PY "$B/stack_audit.py" "$elf"
done
diff -ru "$B/baseline-before" "$B/baseline-after"
$PY -m pytest -q specs/open-engine/tests/test_dense_tensor_quant.py \
  specs/open-engine/tests/test_quant_mixed.py specs/open-engine/tests/test_quant_q8.py \
  specs/open-engine/tests/test_qwen3_dense.py specs/open-engine/tests/test_spec_derive.py \
  specs/open-engine/tests/test_gguf_twin.py specs/open-engine/tests/test_build_cache.py
$PY -m pytest -q open_kernels -o python_files='test_*.py'
dibs run build --label 'P5 S0 fit probe: Rust regressions' --eta 10m -- \
  cargo test -p phlegm-core --offline
git diff --check
```

For hardware follow-up, first implement the GGUF q8 pack ops and runtime V0/V1
selection in S1/S2 and export a complete set. The compile-only probe deliberately
cannot run G1-G6; running its image alone would not be a meaningful model test.
Then run the owning plan section 7 on that complete set, including the existing
long-prefix/window failures and the independent short-prompt/Vulkan control.

## Validation attempt ledger

The following abbreviations expand the exact pytest argument lists used during
implementation; all invocations used the Python interpreter above:

```sh
T4='specs/open-engine/tests/test_dense_tensor_quant.py specs/open-engine/tests/test_quant_mixed.py specs/open-engine/tests/test_quant_q8.py specs/open-engine/tests/test_qwen3_dense.py'
T5="$T4 specs/open-engine/tests/test_spec_derive.py"
T7="$T5 specs/open-engine/tests/test_gguf_twin.py specs/open-engine/tests/test_build_cache.py"
$PY -m pytest -q $T4  # 70 passed, 1 skipped
$PY -m pytest -q $T5  # 79 passed, 1 skipped
$PY -m pytest -q $T7  # 140 passed, 2 skipped; then 141/2 after the head-stream guard;
                     # finally 145/2 after the four independent-format cases
$PY -m pytest -q open_kernels  # exit 2: 32 errors during collection
```

The last invocation's two classes of collection error, verbatim (representative
paths; repeated for the other `make_test.py` files):

```text
import file mismatch:
imported module 'make_test' has this __file__ attribute:
  /home/cyrus/code/phlegm-wt/p5-s0b-v1-fit/open_kernels/designs/attn/make_test.py
which is not the same as the test file we want to collect:
  /home/cyrus/code/phlegm-wt/p5-s0b-v1-fit/open_kernels/designs/attn_block/make_test.py
E   ImportError: cannot import name 'bf' from 'make_test' (/home/cyrus/code/phlegm-wt/p5-s0b-v1-fit/open_kernels/designs/attn/make_test.py)
!!!!!!!!!!!!!!!!!!! Interrupted: 32 errors during collection !!!!!!!!!!!!!!!!!!!
32 errors in 0.66s
```

The corrected test-file selection (`-o python_files='test_*.py'`) passes without
changing the repository's pytest configuration. This is not a pass for the
literal broad command. No NPU tests were collected or run.

The initial preparer attempt failed with
`ValueError: too many values to unpack (expected 2)` because `read_gguf` returns
three items; fixed before compilation. An initial build launch failed because
the staging ignore rule also excluded `build_design.py`:

```text
python: can't open file '/home/cyrus/code/phlegm/target/npu-dev/kernel-builds/p5-s0-dx/v1/source/open_kernels/build_design.py': [Errno 2] No such file or directory
```

The ignore rule now excludes build directories only. The corrected first build
(`v1/`, same command as above with `final-v1` replaced by `v1`, no `DX_ONECTX`
override, label `P5 S0 fit probe: V1 dx compile`, ETA 15m) passed. Dibs pressure
blocked one launch at 95% CPU; `dibs wait --pressure` returned FREE, then the
identical launch was retried. No override was used. The final snapshot was then
prepared and compiled with the final source and explicit head-stream guard.

Two inspection setup errors were corrected before recording budgets: objdump
first used a flat ELF path (this aiecc places ELFs inside per-core directories),
and the audit first looked up `lni_shim_alloc` without stripping its suffix
(`KeyError: 'lni_shim_alloc'`). Final audit commands exit 0 on both baseline and
final V1. Raw inspection scripts and final output are retained at the evidence
root. Source identity and `git diff --check` were repeated after final edits;
all exit 0.

To regenerate the legacy sources into a separate directory, the actual source
regeneration body used before and after the implementation is:

```sh
PYTHONPATH=open_kernels $PY - <<'PY'
from pathlib import Path
import importlib.util
from recipes.load import load_spec
from recipes import dense
p = Path('/home/cyrus/code/phlegm/target/npu-dev/kernel-builds/p5-s0-dx/baseline-after')
p.mkdir(exist_ok=True)
s = importlib.util.spec_from_file_location('gen', 'open_kernels/designs/dense/gen_kernels.py')
g = importlib.util.module_from_spec(s)
s.loader.exec_module(g)
print(g.generate(dense.recipe(load_spec(Path('open_kernels/recipes/specs/llama32-1b.json'))), p))
PY
```

The pre-edit run used `baseline-before` instead of `baseline-after`; each prints
`5`. The snapshots are preserved, so the pre-edit result remains independently
inspectable after integration.
