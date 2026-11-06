# C++ reference benchmarks

This adapter measures the external OpenFlowLM `open_qwen36::Core`, with the open
NPU kernels supplied by the caller. C++ remains the current inference engine;
the native Rust engine port is deferred until the final implementation phase.
Every result and console table is labeled
`openflowlm-cpp-reference`. The reference engine's sources are not copied here.

## Build on Windows

Requires MSVC Build Tools, the XRT headers/import library already used by
`phlegm-core`, and a reference checkout containing `src/open_qwen36/core.cpp`,
`gguf_file.cpp`, and the Core APIs this bridge calls. Older checkouts without
these APIs fail compilation. No automatic checkout, download, or source patching
is performed.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/reference/build.ps1 `
  -ReferenceRoot C:/code/OpenFlowLM-Next2
```

The default output is `target/reference-bench/phlegm-reference-bench.exe`, with
`phlegm-reference-bench.build.json` beside it. Optional build parameters are
`-OutDir`, `-XrtInclude`, `-XrtLib`, and `-Vcvars`. If the import library is missing,
run `phlegm-core/xrt-shim/build_import_lib.cmd` first.

The build records the reference commit and dirty-source flag; hashes the engine,
adapter, JSON/XRT headers, stream patch helper and import library; and fingerprints
the final executable. Inputs are checked again after compilation. Phlegm verifies
this sidecar and the bridge source embedded in its own build before executing it.
Keep the binary and sidecar together. This is provenance, not a signature proving
an untrusted executable safe: use your own reference build.

## Build on Linux

Requires Python 3.11+, a C++17 compiler with OpenMP/AVX2/FMA, matching XRT
headers and `libxrt_coreutil.so.2`, and the XDNA userspace plugin for device
access. The external reference checkout must implement the selected manifest
features; the retained companion patches have different bases and need manual
reconciliation. Building the bridge does not establish kernel compatibility.

```bash
python3 benchmarks/reference/build.py \
  --reference-root /path/to/reconciled/openflowlm-next \
  --xrt-include /path/to/xrt/include \
  --xrt-lib /path/to/xrt/lib/libxrt_coreutil.so.2
```

The output is `target/reference-bench/phlegm-reference-bench` with a
`.build.json` sidecar accepted by the same Rust provenance verifier as Windows.
`--out-dir` and `--cxx` override the destination and compiler; `CXX` supplies the
compiler default. Arguments are passed without a shell. The builder hashes the
engine, adapter, helper, headers, selected link library, compiler executable and
build script, and checks inputs and Git identity again after compilation. A
failed build invalidates the previous sidecar. The runtime library directory is
embedded as an rpath; keep the matching plugin and dependent libraries available.

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
