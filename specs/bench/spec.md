# Phlegm Bench v1

Owner-run performance measurements committed as JSON. No database or public
submission endpoint. JSON structure is defined by `phlegm_core::bench::BenchReport`;
the benchmark protocol is `phlegm-bench-v1`, schema version 1.

## Availability

The harness, validator and resident C++ adapter are implemented. Phlegm keeps
its engine in C++; Rust owns the application, tokenizer and sampling.
`phlegm bench --reference-engine ...` drives the bridge built from this repository.
Reports retain the label `openflowlm-cpp-reference`. Mock benchmark reports
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
- For reference runs: Phlegm source commit, release tag when present, dirty flag
  over build-input paths, machine-independent `engine_tree_sha256`, full
  source/dependency manifest hash, binary hash, compiler version/flags, recorded
  XRT rpath (null on the loader path), actual prefill route/mode, NPU decode route,
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
(`cargo build --release`). Missing Git metadata fails closed. Mock reports are
never allowed. Unrelated untracked research directories do
not make a build dirty. Tag and Git state changes invalidate Cargo's build cache.

**Development runs.** `phlegm bench --development ...` runs the same protocol on
any optimized build: no release tag is needed and a modified tree is allowed.
The report's `phlegm.development` is `true`, its commit and dirty flag are kept,
and `validate` and the table label it `DEVELOPMENT (not a release)` with the
commit. Optimization level 3 and full provenance are still required. A
development report is for comparing development builds and recording model
work; it is never a published release result. Reports without the field are
release reports, as before.

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

The adapter supports the in-tree C++ engine's `model.gguf` and `model.q4nx`
directories, recording which artifact was loaded (q4nx wins if both exist).
The engine derives from OpenFlowLM; its MIT notice ships in `open-kernels/LICENSE_OPEN_RUNTIME.md`.

The bridge sidecar records `source_commit` as Phlegm HEAD and `source_dirty`
over engine/build inputs only. `engine_tree_sha256` hashes sorted relative
`bridge/`, `engine/`, `json/` and `harness/` entries: each contributes its name,
a NUL, lowercase file SHA-256 and newline. It excludes machine paths, XRT and
the compiler. Rust compares it with the constant emitted by `build.rs` and
rejects mismatches, including bridges from a different family-patch tree.
`source_tree_sha256` additionally covers build tools, compiler and XRT inputs.
Older sidecars must be rebuilt. Keep the binary and sidecar together.
The [S3 integration record](../open-engine/in-tree-engine-s3.md) lists the
pristine import, host-test limitation and outstanding hardware gate.

The Phlegm release guard still applies. Development bridge edits are identified
by the dirty flag and content hashes; published reports need the clean release
build described above. Hardware equivalence remains a separate gate.

## Embedding benchmark

`phlegm bench embed` uses the same resident NPUE adapter as `phlegm serve`.
Supply the packed model, worker, projection design and optional attention
directory. The fixture file is a JSON array of cases. Each case declares
`tokens_per_input` and either `texts` or `token_ids`. Use an optional
`prompt_name` for text fixtures that require a task prefix.
Use `--embed-short-design` to exercise the serving router with a smaller
projection design. The report records the capacities and artifact hashes
for both routes.

For a GGUF input, pass `--embed-packer` and `--embed-gguf-sha256` with
`--embed-model`. The CLI caches the derived NPUE before timing and records
the GGUF and NPUE hashes. Serving uses the same cache. You can select its
directory with `--embed-pack-cache` and the packing script with
`--embed-pack-script`.

The model stays resident. The benchmark checks the processed token counts in
every warmup and timed call. It alternates the order of cases across measured
rounds and stores per-case samples, median latency and input token throughput.
Timing includes the worker protocol and encoder. HTTP latency is measured
through the serving endpoint separately.

Current NPUE workers also return an encoder timing record for each measured
sample. It includes the selected context, projection dispatch count and
encode wall time. The benchmark rejects a reported context that differs
from the route expected for the fixture's declared length. Stage counters
cover input preparation, pooling, rotary
work, projections, attention, normalization and activation. The host timing
starts inside the worker before encoding. Compare it with the outer worker
round trip when measuring IPC overhead. An older worker can return a null
stage timing record.

The `phlegm-embedding-bench-v1` JSON includes the machine, Phlegm build, model
revision, quant and hashes for the NPUE pack, worker, fixtures and design
payloads. Use `--development` for local bring-up runs. Establish hardware
parity and collect the NPU power mode and complete stack before publishing.

## Publishing

Commit validated reports under `benchmarks/results/` after a complete run. Keep
one JSON file per model/machine/release/run, named with UTC date and a unique
suffix. The site can build its leaderboard from these files. No endpoint, login,
or write credentials are required in GitHub Pages. Never include fixture or
mock results as measured model performance.
