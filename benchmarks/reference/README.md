# C++ reference benchmarks

This adapter runs Phlegm's in-tree `open_qwen36::Core`, derived from OpenFlowLM.
The engine stays C++; Rust owns the CLI, server, tokenizer, sampling and tray.
Results retain the label `openflowlm-cpp-reference`.

## Build on Linux

Requires Python 3.11+, a C++17 compiler with OpenMP/AVX2/FMA, and matching XRT
headers and `libxrt_coreutil.so.2`. Device access also needs the XDNA plugin.

```bash
cargo build -p phlegm-cli --features npu --offline
```

This builds `target/debug/phlegm` and the adjacent `phlegm-reference-bench`
with its `.build.json` sidecar. `--release` puts them in `target/release`.
No external engine checkout is needed. A build without `npu` needs neither XRT
nor a C++ compiler. Cargo invokes `build.py` so its flag list remains the single
definition: `-O3 -DNDEBUG -std=c++17 -fopenmp -mavx2 -mfma`, with
`-ffp-contract=off` on `block_host.cpp` alone. The imported CMake file is retained
for host tests; it does not build the shipped bridge.

The system compiler is the default; `CXX` overrides it and `PYTHON` selects Python.
XRT defaults to the system installation, then `/opt/xilinx/xrt`;
`XILINX_XRT`, `XRT_INCLUDE_DIR` and `XRT_LIB_DIR` override discovery.
Linking uses `-l:libxrt_coreutil.so.2`. XRT on the loader path needs no rpath;
otherwise only its selected library directory is recorded as `xrt_rpath` and
embedded. Conda GCC's injected environment rpath is suppressed by a generated
link-only specs override, recorded in the full source manifest. Compilation
flags stay unchanged. No conda rpath is added.

The sidecar records Phlegm HEAD, a dirty flag over engine/build inputs, an exact
release tag when present, source/dependency hashes, compiler version/commands,
XRT rpath and the binary hash. `engine_tree_sha256` hashes sorted relative
`bridge/`, `engine/`, `json/`, `harness/` inputs independently of machine paths,
XRT and compiler. Rust compares it with the constant emitted by `build.rs`.
An engine-tree mismatch is rejected; old sidecars need rebuilding. Keep the
binary and sidecar together. Provenance is not a signature for untrusted code.

The trunk import does not yet contain the later family patch chains or the
PR 209 manifest versions 5–7. Compilation and handshake do not establish
model compatibility or logit equivalence. `bridges/index.json` remains a stopgap;
the in-tree bridge takes precedence, including for developer test executables.

External A/B builds remain available until migration stage S8:

```bash
python3 benchmarks/reference/build.py \
  --reference-root /path/to/reconciled/openflowlm-next \
  --xrt-include /path/to/xrt/include \
  --xrt-lib /path/to/xrt/lib/libxrt_coreutil.so.2
```

`--out-dir` and `--cxx` override destination and compiler. External trees with
different engine hashes can be used directly by `ab.py`, but the Rust verifier
accepts only its compiled-in engine tree.

Every engine file must match OFLM trunk `38ae3d2` unless
`engine-changes.json` records the change (its origin and the file's new
sha256). Add an entry there with each engine change; check with:

```bash
python3 benchmarks/reference/check_import.py --upstream /path/to/trunk-checkout
```

## Windows migration

`build.ps1` and `--reference-root` are retained for migration. The Windows Cargo
path still builds the XRT shim only; the in-tree bridge and updated Windows
provenance are stage S10, untested here. The old PowerShell sidecar lacks the
required engine hash and must be updated before Rust will accept it.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/reference/build.ps1 `
  -ReferenceRoot C:/code/OpenFlowLM-Next2
```

For Linux hardware runs, check `/dev/accel/accel0` access and both soft and hard
memlock limits. The current local XRT stack needs a 64 MiB locked heap at device
open; an 8 MiB limit prevents even enumeration. Local setup and remaining
coverage are recorded in
[baseline readiness](../../specs/open-engine/artlite-baseline-readiness.md).

## Run and serve through the NPU bridge

The same resident bridge now handles normal generation. `phlegm run` and
`phlegm serve` prefill the prompt with the selected open-kernel route and decode
on the NPU. Rust samples the returned binary32 logits and applies the model's
chat template and stop tokens. Between requests the bridge keeps its context:
a request prefills only what its prompt adds (`rewind` to an earlier position,
`prefill_logits` with a `mark` that saves recurrent layers' state;
SERVER-PROMPT-REUSE), and resets when nothing is shared.
`PHLEGM_PROMPT_REUSE=0` resets for every request as before.

```powershell
cargo run -p phlegm-cli -- run --model-dir C:/models/my-model `
  --kernels C:/kernels/my-model `
  --bridge target/reference-bench/phlegm-reference-bench.exe `
  --prompt "Explain the NPU in one sentence"
cargo run -p phlegm-cli -- serve --model-dir C:/models/my-model `
  --kernels C:/kernels/my-model `
  --bridge target/reference-bench/phlegm-reference-bench.exe
```

`--max-ctx`, `--threads`, `--prefill-route auto|block|sequential`,
`--prefill-mode fast|lean`, `--timeout`, and `--dispatch-timeout-ms` select runtime
settings. `--threads` (the bridge's OpenMP threads) defaults to half the logical
CPUs, 4 to 16: the block routes' host stages scale with it (LFM3, 8192-token
prompt: 56 s at 4 threads, 33-36 s at 16, the same logits) and decode does not. The bridge sends logits as raw bytes right after their reply
line: the JSON names their count (`logits_f32_le_bytes`), then that many
little-endian float32 bytes follow (about 1 MB a token for a 248k vocabulary).
Before 2026-11-05 they went inside the JSON as base64, which cost an encode,
a 1.3 MB JSON parse and a decode per token. Through `phlegm serve` (2026-11-06,
greedy, 256 tokens, three alternating rounds, host load 6-10 from other jobs, identical
replies; `target/npu-dev/logits-ab/ab.json`): Clover (248k vocabulary) 91.5-111.9 ->
82.2-88.5 ms a token, faster in every pair; LFM3 (65k) 43.6-46.0 -> 40.2-46.0. `serve --mock` runs the
explicit test backend. No release tag is needed for run/serve; the strict
release and power-mode rules below still govern published benchmark reports.

The bridge never outlives phlegm. A normal exit kills it from `Drop`; on Linux it
is also started with `PR_SET_PDEATHSIG` = SIGKILL, so a SIGTERM or SIGKILL to
phlegm stops it at once instead of leaving it on the NPU until its current
request ends (2026-11-04: `phlegm serve` sent SIGTERM 8 s into a request; its
bridge was gone 2 s later, `target/npu-dev/effort/check.json`). The bridge must
be spawned from a thread that lives as long as the backend: the kernel sends the
signal when the spawning thread exits. `run` and `serve` spawn it on the main
thread. A left-over `/tmp/phlegm-npu.lock` naming a dead process is removed by
the next phlegm that wants the NPU.

## Run from a tagged Phlegm release

The Phlegm source must be clean, its Cargo version must match an exact `v<version>`
tag at the build commit, and the executable must be optimized. No development
override is provided. Build both the bridge and Phlegm from the same adapter
source. This document does not create a release tag for you.

```powershell
cargo build --release -p phlegm-cli
.\target\release\phlegm.exe bench `
  --reference-engine .\target\reference-bench\phlegm-reference-bench.exe `
  --model-dir C:/models/my-model `
  --kernels C:/kernels/my-model `
  --model-id organization/model `
  --model-revision <model-source-commit> `
  --quantization Q4_1 `
  --output benchmarks/results/my-model-2026-10-22-run1.json `
  --notes "AC power; selected TDP and cooling conditions"
```

Use the actual model source revision and quantization, not the example values.
The model directory needs `tokenizer.json` and `model.gguf` or `model.q4nx`; the
temporary reference engine chooses q4nx when both exist. The kernel directory
needs its matching `manifest.json` and all artifacts it names. Model, tokenizer,
kernel and executable SHA-256 fingerprints are collected outside timing; the
model/config/tokenizer/kernel hashes are checked again before saving.

Default suite: 512/1024/2048/4096 input tokens, 128 output tokens, one warmup and
three measured trials per input length. Greedy sampling never stops at EOS. The
first output belongs to prefill, leaving 127 measured decode steps. Context
capacity defaults to 8192. `--prefill-route auto` selects layer-major, block GEMM,
or sequential prefill according to the loaded kernel set; the actual selection is
recorded. Sequential prefill on large models can take a long time.

The process stays resident across all trials, resets state each time, and times
synchronous engine calls internally. IPC and metadata collection do not count.
Kernel/input incompatibility, nonfinite logits, incomplete runs and timeouts
discard the run. No partial report is saved and existing files are not overwritten.

The adapter reads NPU power mode with `xrt-smi` before and after each trial and
rejects a changed mode; it does not change the mode itself. Set the intended mode
before starting. Provide `--xrt-smi` if it is installed elsewhere. Windows driver
metadata is normally detected; `--driver-version` and `--machine-info` are explicit
overrides. OpenMP uses `PASSIVE` and the logical CPU count by default; `--threads`
changes the count. Inherited `OFLM_*` and `OMP_*` switches are cleared.

Close other NPU workloads before collecting published numbers. The Phlegm lock
coordinates cooperating Phlegm processes and renews its lease during long suites;
it cannot prevent unrelated NPU clients
from sharing the device. Record the machine's AC/battery, TDP and cooling conditions.

## Verify without publishing scores

Machine metadata and saved-report validation work in development builds:

```powershell
cargo run -p phlegm-cli -- bench machine
cargo run -p phlegm-cli -- bench validate benchmarks/results/example.json
```

After building the bridge, test its provenance and protocol without loading a model:

```powershell
$env:PHLEGM_REFERENCE_BRIDGE = "$PWD/target/reference-bench/phlegm-reference-bench.exe"
cargo test -p phlegm-core compiled_bridge_handshake_and_provenance -- --ignored
```

A short model correctness smoke check also works without publishing a benchmark:

```powershell
python benchmarks/reference/smoke.py `
  --bridge target/reference-bench/phlegm-reference-bench.exe `
  --model-dir C:/models/my-model --kernels C:/kernels/my-model
```

It verifies resident inference, complete decode steps, and reproducible state reset
using a tiny prompt. It writes no result file and prints no speed scores. Use token
IDs suitable for your tokenizer via `--ids 1,2,3,4` if needed. This smoke check does
not establish correctness or performance of the full 4096-token suite.

### Bonsai 2

Bonsai 2 requires a C++ checkout with ternary packing and Hadamard support,
plus matching kernels. The older `OpenFlowLM-Next2` checkout used for the Llama
smoke check lacks this support. Build a separate adapter against a compatible
checkout; do not reuse its binary just because the bridge protocol matches.

The following local setup generated a coherent reply ending at EOS on
2026-10-22: `Ternary-Bonsai-2-27B-NPU2-lmhead`, the `open_kernels_land2` set,
and C++ commit `a67d1c1ef646b5158cba0aeb79527542da04fe79` from `bonsai-land`.
All 64 layers loaded on NPU Strix; the adapter selected layer-major prefill.
This is a short correctness check, not a published performance result or
validation of every context length. The rotated LM head weights must match the
kernel recipe; the non-lmhead model directory is a different variant.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/reference/build.ps1 `
  -ReferenceRoot C:/code/bonsai-land -OutDir C:/code/phlegm/target/reference-bonsai
python benchmarks/reference/smoke.py `
  --bridge target/reference-bonsai/phlegm-reference-bench.exe `
  --model-dir C:/Users/josha/.flm/models/Ternary-Bonsai-2-27B-NPU2-lmhead `
  --kernels C:/code/bonsai-land/src/xclbins/Ternary-Bonsai-2-27B-NPU2/open_kernels_land2
```
