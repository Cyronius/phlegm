# In-tree C++ engine: P1 S3

2026-11-10. Hardware-free integration; no model was loaded and no NPU was opened.
The permanent split is C++ inference, with Rust CLI, server, tokenizer, sampling
and tray. This stage imports the trunk; it does not merge the family patches.

## Import and build contract

The 77 files in `src/open_qwen36/` (including vision),
`src/include/nlohmann/` and `LICENSE_OPEN_RUNTIME.md` are byte-identical to
`38ae3d2643` in the saved OpenFlowLM lineage. No additional OFLM application
headers are needed by the bridge. The legacy OFLM application adapter
`engine.cpp` is preserved but is not a bridge translation unit.

```sh
python3 benchmarks/reference/check_import.py \
  --upstream /home/cyrus/code/phlegm/target/npu-dev/gemma4-moe-runtime
cargo build -p phlegm-core -p phlegm-cli --offline
cargo build -p phlegm-core -p phlegm-cli --features npu --offline
```

Wrap builds/tests in Dibs on the shared development machine. Linux `npu`
builds the shim and invokes `benchmarks/reference/build.py` for the bridge;
its existing flag list is the single definition. Only `block_host.cpp` gets
`-ffp-contract=off`. The bridge and sidecar are installed next to `phlegm`.
The tracked PR 209 `stream_patch.hpp` compiles unchanged against this trunk.
No upstream patch was applied.

The non-`npu` build does not invoke a C++ compiler or discover XRT. Tokenizers'
training-only `esaxx_fast` feature is disabled so it does not introduce an
unrelated C++ requirement. The existing `sha2` dependency is also used by the
build script; no new crate is introduced.

The resolver prefers explicit paths, the environment override, the adjacent
bridge, then the compiled developer bridge path, before the stopgap index.
Old bridge sidecars lack the required engine hash and need rebuilding;
external trees with different hashes remain usable directly through `ab.py`.

## Provenance

See [the benchmark contract](../bench/spec.md). The sidecar identifies Phlegm
HEAD, dirty build inputs, an exact release tag when present, compiler
version/commands, XRT rpath, source hashes and the executable hash. The
machine-independent engine hash includes bridge, C++ sources/headers, JSON
headers and the tracked stream helper, using sorted relative paths. Rust
emits the same hash without invoking Python, XRT or C++ in non-`npu` builds
and rejects a bridge with a different hash. Compiler/XRT inputs remain in
the separate full source manifest. Documentation edits outside the build
inputs do not mark the engine dirty.

## Known imported host-test failure

`manifest_test` does not compile at this pristine trunk. Lines 811, 814, 818
and 829 copy `Manifest`, whose `gguf` member is a `unique_ptr<Manifest>`:

```text
error: use of deleted function ‘open_qwen36::Manifest::Manifest(const open_qwen36::Manifest&)’
```

The import was not edited. Proposed follow-up: replace `Manifest k = f`
with `Manifest k = Manifest::load(argv[i])`; replace each `Manifest half/odd/wide = k`
with a fresh `Manifest::load(argv[i])`, then set that instance's `family = k.family`.
This preserves the fixture-based tests without adding copy semantics to the engine.
The proposed diff is supplied separately for review, not applied here.

`nemotronh_pack_check` and `xing_pack_check` are absent from this trunk.
The CLI target needs hardware for inference. Vision's configuration/window
tests need no model; its numerical fixture modes need separate model/fixture
data and are not covered by the configuration tests.

## Hardware equivalence remains open

All 18 registry entries currently map to patched leaves; none supplies a
recorded unpatched-trunk golden. In particular `gemma4` in the stopgap index
maps to `gemma4moe-blk` (26B MoE), not the trunk's 12B dense route. Nanbeige
needs the later PR 209/Llama work even though some manifest verbs overlap.

The available trunk candidate is the dense model at
`target/npu-dev/models/gemma4-12b`, kernels `target/npu-dev/kernels/gemma4-12b-shl`,
and token files in `target/npu-dev/gemma4-12b-ref/`. A dense golden must first
be captured with the previous dense-capable bridge and the same cases. Then
compare both compiler builds with that golden. A successful handshake proves
neither route equivalence nor logits nor model quality. LFM3 equivalence waits
for its S4 patch chain; Windows bridge integration remains S10.

## Recorded hardware-free results

Both bridges link system XRT `/usr/lib64/libxrt_coreutil.so.2` (2.27.0).
System compiler: `g++ (GCC) 16.2.1 20261029 (Red Hat 16.2.1-3)`.
Comparison compiler: `x86_64-conda-linux-gnu-c++ (conda-forge gcc 15.3.0-7) 15.3.0`.
Their executable SHA-256 values, respectively:

```text
735922254d525965bbe63fc1025359bed33cd7e88dcaedc5411a7914e681be5a
7178a08757fd105d061349e96effd8ac88c8cec2810906257c48110d98867768
```

Both have engine hash
`77b2cadf430a9d56447e07a4a3b2ee1eb850ee3dbaf60096c11ad536016c29eb`.
The build commit is `d226a4bf87391117d731c1b6e63b52adaefd6192`, dirty inputs
true, release tag null. This is development evidence, not a clean release.

`readelf -d` for each reports NEEDED `libxrt_coreutil.so.2`, `libstdc++.so.6`,
`libm.so.6`, `libgomp.so.1`, `libgcc_s.so.1`, `libc.so.6`; neither has
RPATH or RUNPATH. Conda's first build injected its environment's `lib` rpath.
The final builder overrides only GCC's `link_command` with its built-in
specs stanza (hashed in `source_files`); compilation specs/flags are unchanged.
`-rpath-link` locates XRT's system dependencies at link time without embedding
that search path. No `patchelf` or additional dependency is required.

Checks performed under Dibs:

- Pristine import: 77/77 SHA-256 matches.
- Default core/CLI build: pass, also with invalid `CXX` and XRT paths.
- Linux `npu` core/CLI build: pass with each compiler.
- Compiled bridge handshake/provenance: one test passes with each compiler's
  bridge; no model/device init command is sent.
- Default Rust suite: core 140, CLI 5, integration 2 pass; 7 ignored.
- Reference A/B Python test: 1 pass using the development environment's Python
  (`pytest` is absent from the default PATH).
- CTest: 5/5 pass with `OPEN-MANIFEST` excluded for the compile failure above.
- `block_host_test` against `replica_block.py`'s generated fixture: PASS.
- `pools_test` compiled separately (not a CMake target): PASS, 0 failures.

Local evidence: `/home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-*.log`, compiler artifacts and sidecars under
`/home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-artifacts/{system,conda}/`, host fixture
`/home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-block-fixture/`, and the unapplied test repair at
`/home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-manifest-test-proposed.patch`. The initial conda binary with the
unwanted rpath is retained separately under `conda-with-rpath` for diagnosis.
No shared development-tree file was written.

The exact check commands, results and resolved build failures are recorded in
`/home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-checks.txt`. `/home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-audit.py` independently verifies both
artifacts against current source inputs, their full and engine manifests,
binary hashes, Git metadata, ELF paths and per-file contraction flags.

Prepared follow-up cases: `/home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-gemma4-12b-cases.json` (30 cases: short,
long and forced-ID prefixes, lengths around 256 and 1024, auto/sequential,
8 greedy steps plus rewind). The dense golden below still needs capturing.
These are commands for the hardware owner; they were **not run**:

```sh
D=/home/cyrus/code/phlegm/target/npu-dev
W=/home/cyrus/code/phlegm-wt/p1-s3-vendor
# First record the previous dense-capable bridge's golden with these same cases.
dibs run npu --measure --label 'P1 S3: dense golden' --eta 1h -- "$D/npu-run.sh" \
  "$D/env/bin/python" -I "$W/benchmarks/reference/ab.py" record \
  --bridge "$D/bridges/gemma4moe-blk/phlegm-reference-bench" \
  --cases /home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-gemma4-12b-cases.json \
  --out "$D/one-runtime/golden/gemma4-12b.json"
for compiler in system conda; do
  dibs run npu --measure --label "P1 S3: dense $compiler equivalence" --eta 1h -- \
    "$D/npu-run.sh" "$D/env/bin/python" -I "$W/benchmarks/reference/ab.py" compare \
    --bridge "/home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-artifacts/$compiler/phlegm-reference-bench" \
    --golden "$D/one-runtime/golden/gemma4-12b.json" \
    --cases /home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/p1-s3-gemma4-12b-cases.json --rounds 2 \
    --out "$D/one-runtime/s3-gemma4-12b-$compiler.json"
done
```

The previous MoE leaf's dense compatibility is an inferred candidate for
this capture, not an already verified golden. If it cannot serve the dense
cases, build the saved dense/trunk baseline first; do not substitute a 26B
MoE golden or claim equivalence from overlapping manifest verbs.

## Review changes (2026-11-10, orchestrator)

- Bridge selection: a kernel family listed in `bridges/index.json` keeps its per-family bridge and never falls back
  to the in-tree bridge, even when Cargo has placed one beside `phlegm`; only families outside the index use the
  in-tree bridge (adjacent first, then the Cargo-built path). The trunk carries none of the registry families'
  patches until S4-S7, so the worker's order (in-tree bridge before the index) would have run every registry model
  on an engine that lacks its patches.
- Provenance: `engine_tree_sha256`, `release_tag` and `xrt_rpath` are optional in the sidecar. A bridge built from
  this repository must match the compiled-in engine hash; legacy per-family bridges (no field) stay accepted until
  their families are folded in. Test: `bridge_from_a_different_engine_tree_is_rejected` also checks the legacy case.
- `benchmarks/reference/bridge.cpp` is unchanged: its sha256 (`97c5cef3...`) is the `adapter_sha256` every legacy
  bridge carries, so editing even its header comment would reject all 13 current bridges. Its header comment is
  updated in S8, when the per-family bridges are retired.
- Evidence moved from `/tmp/p1-s3*` to `/home/cyrus/code/phlegm/target/npu-dev/one-runtime/s3/`.
- Checks after review: `cargo test -p phlegm-core -p phlegm-cli --offline` core 140 + CLI 5 + 1 + 1 pass, 7 ignored;
  `cargo build -p phlegm-core -p phlegm-cli --features npu --offline` (system XRT) pass; `git diff --check` clean.
