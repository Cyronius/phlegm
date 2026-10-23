# Phlegm Bench v1

Owner-run performance measurements committed as JSON. No database or public
submission endpoint. JSON structure is defined by `phlegm_core::bench::BenchReport`;
the benchmark protocol is `phlegm-bench-v1`, schema version 1.

## Availability

The harness, validator, and C++ reference adapter are implemented. C++ remains
the current inference engine; the native Rust engine port is the final phase.
`phlegm bench --reference-engine ...` drives a resident process built against an
external OpenFlowLM reference checkout. These reports are explicitly labeled
`openflowlm-cpp-reference`; they do not represent the future native Rust engine.
The Rust engine still has only a mock generate backend. Mock benchmark reports
remain rejected. No benchmark scores are checked into this repository.

Available now:

```sh
cargo run -p phlegm-cli -- bench --help
cargo run -p phlegm-cli -- bench machine
cargo run -p phlegm-cli -- bench validate results.json
```

For reference-engine builds and the complete tagged-release run command, see
[`benchmarks/reference/README.md`](../../benchmarks/reference/README.md).

Metadata collection and validation can run in development builds. Validation
checks the release provenance recorded in the report, not the validator's build.

## Protocol

| Setting | Value |
| --- | --- |
| Input lengths | Exactly 512, 1024, 2048, 4096 tokenizer tokens |
| Output length | 128 tokens, including the first token from prefill |
| Sampling | Greedy; ignore EOS to finish every trial |
| Batch / concurrent requests | 1 / 1 |
| Warmups | 1 per input length, excluded from results |
| Measurements | 3 per input length |
| State | Model resident; empty KV, recurrent and prefix caches before every trial |
| Prefix reuse / speculative decode | Disabled |

The harness tokenizes 256 repetitions of the versioned English corpus in
`bench::PROMPT`, without a chat template, outside the timers. The first 4096 IDs
are recorded in the report. Every smaller case uses an exact prefix. An adapter
must use the loaded model's tokenizer with automatic special-token insertion
disabled. Fewer than 4096 encoded IDs or insufficient context capacity fails
the entire run. The default suite requires 4223 context positions (4096 + 127).

The adapter must validate token IDs against the loaded model and return errors
for invalid IDs, unsupported context sizes, nonfinite logits, or device errors.
There is no partial-success report and no fallback to a smaller case.

## Timing boundaries

Use monotonic wall-clock timers around synchronous engine calls. Every call must
complete device work before returning; timing asynchronous enqueue operations
would produce invalid results. Loading, hashing, tokenization and state reset are
excluded. Warmups keep model weights and compiled kernels resident.

**Prefill** includes processing all input tokens, the output head, and the first
greedy token selection. It is also engine time to first token, not HTTP latency.
`prefill_tokens_per_second = input_tokens * 1000 / prefill_ms`.

**Decode** measures 127 subsequent forward passes and greedy token selections.
The first token is not counted again. Each step consumes the previous generated
token and returns the next one, even if it is an EOS token.
`decode_tokens_per_second = (output_tokens - 1) * 1000 / decode_ms`.

The C++ adapter overrides `measure_trial` and runs these exact timing boundaries
inside its child process. JSON serialization, IPC, power-mode checks, and reset
are outside the timers. The model stays loaded across the entire suite. Core's
reset clears recurrent state and logically invalidates every KV row by resetting
the valid position; it has no prompt/prefix reuse cache. Old KV bytes cannot skip
any prompt work.

Record raw prefill/aggregate decode milliseconds plus per-step decode latencies.
Summaries contain median, mean, min, max, and population standard deviation across
measured runs. Decode latency p50/p95 use linear interpolation over all measured
steps for that case. Aggregate decode time includes loop/timer overhead.

The API permits 1..10 warmups, 1..20 measured runs, and 2..4096 output tokens.
Nondefault configurations remain recorded and must be compared separately.

## Required metadata

- Run start in UTC Unix milliseconds, benchmark version, and schema version.
- Phlegm package version, exact release tag, full commit SHA, clean-tree flag,
  compiler, target triple, optimization level, and enabled engine features.
- CPU name, logical CPU count, RAM bytes, OS/version and architecture.
- Detected accelerators and driver versions; power profile when available.
- Actual backend/device, runtime driver version, kernel/artifact manifest SHA-256,
  and allocated context capacity.
- For reference runs: engine commit, dirty-source flag, source/dependency manifest
  hash, binary hash, compiler flags, actual prefill route/mode, NPU decode route,
  observed NPU power mode, OpenMP settings and selected weight filename.
- Model ID, source revision, quantization, weight artifact SHA-256, and tokenizer
  SHA-256. For multiple weight/kernel files, hash a documented canonical manifest
  containing relative names and individual SHA-256 hashes.
- Exact prompt IDs, warmup count, measured-run count, output-token count, raw
  samples, summary metrics and optional notes.

Hash artifacts before timing. Never substitute a download URL or model filename
for an artifact fingerprint. Linux/macOS automatic metadata collection leaves
accelerator inventory and power mode unfilled; the adapter/caller must supplement
those when known. Windows collects display/NPU device names and signed-driver
versions. Collection excludes hostnames, usernames, hardware serials and paths.

For useful comparisons, note AC/battery, TDP/performance setting, cooling, and
other workloads. Memory bandwidth/channel configuration and peak process/device
memory would be useful future measurements; v1 does not estimate them.

## Tagged releases only

Build provenance is embedded when compiling `phlegm-core`. A benchmark requires
the exact tag `v<Cargo package version>` pointing at the build commit, no tracked
modifications, no untracked engine/CLI source files, and optimization level 3
(`cargo build --release`). Missing Git metadata fails closed. No override flag
allows development or mock reports. Unrelated untracked research directories do
not make a build dirty. Tag and Git state changes invalidate Cargo's build cache.

This detects local build provenance; it does not cryptographically prove an
official GitHub Release exists or prevent someone editing a JSON report. Publish
only owner-run measurements from tags you have actually released. The validator
checks consistency, not authenticity of claimed hardware, release, or scores.

## Reference and native engine integration

Implement `BenchBackend` on the real loaded engine; do not adapt streaming
callbacks or infer tokens from text chunks. Supply actual model/runtime metadata,
clear all inference state in `reset`, and implement synchronous `prefill` and
`decode`. Configure greedy sampling and disable EOS stopping and prefix reuse.
Do not reload/recompile the model inside timed calls. Ensure identical reset
semantics for the first warmup and every later trial.

The CLI now loads the temporary reference adapter once, collects machine
information, calls `bench::run`, displays `report.table()`, and saves the report.
Saving refuses invalid reports and overwrites. A process timeout, malformed reply,
nonfinite logit, incomplete output, changed power mode, or changed model/kernel
artifact aborts publication. Inherited OFLM/OMP probe and tuning variables are
cleared; OpenMP passive waiting and the requested thread count are explicit.

The temporary adapter supports the reference engine's `model.gguf` and
`model.q4nx` directories, recording which artifact was actually loaded (q4nx wins
if both exist, matching Core). This exception is limited to reference runs; the
native Rust engine's GGUF-only plan is unchanged. No C++ engine source is vendored.

The Phlegm release guard still applies. The separately built C++ reference may
have local source edits; those are identified by its dirty flag and content hash.
Such measurements must be compared under the reference label and fingerprints.

## Publishing

Commit validated reports under `benchmarks/results/` after a complete run. Keep
one JSON file per model/machine/release/run, named with UTC date and a unique
suffix. The site can build its leaderboard from these files. No endpoint, login,
or write credentials are required in GitHub Pages. Never include fixture or
mock results as measured model performance.
