# P5 S6 offline tokenizer acceptance

The probe loads a directory containing one publisher-named GGUF header, without
`tokenizer.json`, tokenizer configuration, or an external chat template. It uses
the production `Tokenizer::load` path. No inference or NPU access is involved.

`fetch_headers.py` pins each public repository's current commit using its API,
requires HTTP 206 and a matching Content-Range, and reads bounded 4 MiB ranges
until the metadata and tensor descriptors end. Only that header is saved;
the last range can include a discarded suffix beyond the header. `source.json`
records every range and the exact saved-header SHA256. `vocab.gguf` changes only
the tensor count to zero, drops the tensor descriptors, and pads to 32 bytes.
Every metadata byte is retained for the independent reference. No tensor data
is needed or saved. Existing destination directories are refused.

`fetch_publishers.py` separately fetches pinned public tokenizer JSONs into new
directories. Meta's three original repositories returned HTTP 401 on 2026-11-09;
`--mirrors` additionally fetches the public Unsloth tokenizer mirrors. Mirror
results are labelled with their own source repository and commit.

The installed `~/.local/bin/llama` is build 12015/a34de8618. It reports
`error: unknown command 'tokenize'` and `error: invalid argument: --vocab-only`.
The fallback reference is `reference.cpp`, linked to the existing local
`pair-cache-llama/build/bin/libllama.so.0.7.0`, using the public `vocab_only`
API. This is an independent llama.cpp implementation, **not a build-12015
comparison**. That library's build metadata says commit `unknown`; hashes of
the actual library, helper and inspected sources are retained with the evidence.
No claim that the inspected source is exactly the library's source revision is
made. Publisher JSON comparisons provide another independent control.

Commands used from this worktree (all cargo commands use the populated cache):

```bash
dibs run network --label 'P5 S6: pinned GGUF tokenizer headers' --eta 20m -- python -I utilities/gguf-tokenizer/fetch_headers.py /home/cyrus/code/phlegm/target/npu-dev/p5/s6-headers
dibs run network --label 'P5 S6: public publisher tokenizer JSONs' --eta 10m -- python -I utilities/gguf-tokenizer/fetch_publishers.py /home/cyrus/code/phlegm/target/npu-dev/p5/s6-headers
dibs run network --label 'P5 S6: public Llama tokenizer mirrors' --eta 5m -- python -I utilities/gguf-tokenizer/fetch_publishers.py /home/cyrus/code/phlegm/target/npu-dev/p5/s6-headers --mirrors
dibs run build --label 'P5 S6: reference EOG metadata probe' --eta 2m -- c++ -std=c++17 -O2 utilities/gguf-tokenizer/reference.cpp -I/home/cyrus/.cache/tmp-relocated/pair-cache-llama/include -I/home/cyrus/.cache/tmp-relocated/pair-cache-llama/ggml/include -I/home/cyrus/.cache/tmp-relocated/pair-cache-llama/vendor -L/home/cyrus/.cache/tmp-relocated/pair-cache-llama/build/bin -Wl,-rpath,/home/cyrus/.cache/tmp-relocated/pair-cache-llama/build/bin -lllama -o /home/cyrus/code/phlegm/target/npu-dev/p5/s6-reference
dibs run build --label 'P5 S6: native EOG tokenizer probe' --eta 3m -- cargo build -p phlegm-core --example gguf_tokenize --offline
dibs run cpu --label 'P5 S6: final ids, EOG and publisher gate' --eta 5m -- env LD_LIBRARY_PATH=/home/cyrus/.cache/tmp-relocated/pair-cache-llama/build/bin /home/cyrus/code/phlegm/target/npu-dev/env/bin/python -I utilities/gguf-tokenizer/check.py --headers /home/cyrus/code/phlegm/target/npu-dev/p5/s6-headers --probe /home/cyrus/code/phlegm-wt/p5-s6-gguf-tokenizer/target/debug/examples/gguf_tokenize --reference /home/cyrus/code/phlegm/target/npu-dev/p5/s6-reference --output /home/cyrus/code/phlegm/target/npu-dev/p5/s6-check-v5
dibs run cpu --label 'P5 S6: final independent Jinja2 rendering' --eta 3m -- /home/cyrus/code/phlegm/target/npu-dev/artlite-env/bin/python -I utilities/gguf-tokenizer/check_rendering.py --headers /home/cyrus/code/phlegm/target/npu-dev/p5/s6-headers --evidence /home/cyrus/code/phlegm/target/npu-dev/p5/s6-check-v5
dibs run build --label 'P5 S6: final core offline regression' --eta 5m -- cargo test -p phlegm-core --offline
dibs run build --label 'P5 S6: final pinned-header fixture gate' --eta 5m -- env PHLEGM_S6_HEADERS=/home/cyrus/code/phlegm/target/npu-dev/p5/s6-headers cargo test -p phlegm-core --offline --test gguf_tokenizer_headers -- --ignored --nocapture
git diff --check
```

Choose **new output/download directories** for another network/gate run. Existing
headers can be reused without downloading. The checked-in expected IDs come
from the independent reference, with exact header hashes; the opt-in Rust
acceptance test refuses missing or changed headers. Normal `cargo test` reports
this dependency as ignored. The synthetic tokenizer and registry checks always
run without external data.

The Python gate deliberately exits **1** if any publisher mismatch remains.
Raw inputs, rendered conversations, actual/expected IDs, publisher provenance,
and NFC controls are retained; a llama.cpp-only pass cannot hide this failure.
See [the owning evidence report](../../specs/open-engine/gguf-tokenizer.md).

Earlier attempts are retained in `s6-check-v1` through `s6-check-v4` beside the
final evidence, rather than overwritten:

- v1: reference infrastructure failure on every target:
  `error while loading shared libraries: libggml.so.0: cannot open shared object file: No such file or directory`.
  An attempted RPATH change did not resolve transitive loading; explicit
  `LD_LIBRARY_PATH` is required in the commands above.
- v2/v3: 10 cases per target, 20 native encodings per target, no native ID
  mismatches; each Qwen publisher had two NFC-related case mismatches. v3 adds
  the three reachable Llama mirrors and independent native NFC controls.
- v4: added the `eog-markers` case and stop-ID comparison. This exposed an
  over-broad `</s>` inference for Gemma4: two native encoding-mode mismatches,
  one publisher-case mismatch, and a stop-set mismatch for each Gemma target.
  The fix follows llama.cpp's explicit Gemma4 exception. Qwen3 has three
  publisher mismatches with this corpus (two NFC, one `</s>`); Qwen3.5 has two.

The first opt-in Rust test also asserted that an unflushed native streaming
decoder must immediately emit a terminal literal replacement character. It
failed with `gemma4-12b/unicode: streaming decoder`: the streamed string ended
at NUL, while complete decoding included the final `�`. The test now appends
an ASCII sentinel before comparing stream and whole-sequence decoding, as the
native API deliberately holds a terminal replacement character. The separate
production server finish/UTF-8 flush tests remain in the core suite. Retrying
the ten-case header test passed (1 test, all nine targets, 35.03 s).

`rustfmt` is absent from the selected stable toolchain; formatting used the
already installed
`/home/cyrus/.rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin/rustfmt`
on the three new Rust files, without downloading or changing a toolchain.

Final checks: core 91 unit tests + 1 release-provenance test pass, 2 bridge/NPU
prerequisite tests ignored; the opt-in header test separately passes (1 test,
all nine targets); independent Jinja2 rendering passes 27/27. v5 has 198/198
native encoding matches and 9/9 stop-set matches, while retaining 11 publisher
case mismatches and exiting 1. No header target was unavailable. No NPU was used.

Additional focused check commands executed during implementation:

```bash
dibs run build --label 'P5 S6: focused Rust tokenizer checks' --eta 10m -- cargo test -p phlegm-core --offline tokenizer --no-fail-fast
# PASS: 14 passed, 72 filtered (before new GGUF fixture tests).
dibs run build --label 'P5 S6: tokenizer and registry regressions' --eta 3m -- cargo test -p phlegm-core --offline tokenizer --no-fail-fast
# PASS: 17 passed, 73 filtered (the filter excludes the registry-named test).
dibs run build --label 'P5 S6: full core offline regression' --eta 5m -- cargo test -p phlegm-core --offline
# PASS at that checkpoint: 89 unit + 1 integration; 2 native prerequisites ignored.
dibs run build --label 'P5 S6: pinned header acceptance regression' --eta 5m -- env PHLEGM_S6_HEADERS=/home/cyrus/code/phlegm/target/npu-dev/p5/s6-headers cargo test -p phlegm-core --offline --test gguf_tokenizer_headers -- --ignored --nocapture
# FAIL: unflushed terminal replacement character, described above.
dibs run build --label 'P5 S6: pinned header acceptance regression retry' --eta 5m -- env PHLEGM_S6_HEADERS=/home/cyrus/code/phlegm/target/npu-dev/p5/s6-headers cargo test -p phlegm-core --offline --test gguf_tokenizer_headers -- --ignored --nocapture
# PASS: 1 test, then-ten-case corpus, all nine targets.
```

The failed streaming assertion reported verbatim:

```text
assertion `left == right` failed: gemma4-12b/unicode: streaming decoder
  left: "café cafe\u{301} Ελληνικά 中文 日本語 한국어 العربية हिन\u{94d}दी 🙂👨\u{200d}👩\u{200d}👧\u{200d}👦 🦙\0"
 right: "café cafe\u{301} Ελληνικά 中文 日本語 한국어 العربية हिन\u{94d}दी 🙂👨\u{200d}👩\u{200d}👧\u{200d}👦 🦙\0�"
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.82s
```

The final header test also verifies that the fixture covers every corpus entry,
matches the native stop set, and preserves all fixed input bytes. Its success
does not override the separately retained publisher failures.
