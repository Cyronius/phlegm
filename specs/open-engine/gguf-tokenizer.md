# GGUF tokenizer and chat metadata (P5 S6)

2026-11-09. Offline development evidence; no NPU execution or performance claim.

`Tokenizer::load` keeps the existing `tokenizer.json` path when the file exists,
including its error behavior. Otherwise the directory must contain exactly one
`.gguf`, with any publisher filename. The metadata reader stops before tensor
descriptors or payloads and caps its header budget at 128 MiB; individual
strings and arrays have additional bounds. The tensor packer's existing reader
continues skipping arrays.

Supported metadata pipelines are `gpt2/llama-bpe` (including `llama3`/`llama-v3`
aliases), `gpt2/qwen2`, `gpt2/qwen35`, and `gemma4/gemma4`. The fetched Gemma 4
targets actually declare the distinct `gemma4` model, not the older `llama`
SentencePiece tokenizer. Gemma uses raw UTF-8 merges, space-to-`▁` replacement,
and byte fallback. Token types identify control/user-defined tokens; merge
order controls BPE. Scores are validated when present. Unknown model/pre names
and unsupported text-changing flags fail with errors. Explicit NFC metadata is
supported; absence preserves the GGUF/reference behavior.

BOS/EOS flags and IDs construct the postprocessor. A rendered chat already
starting with BOS does not get a duplicate. EOS/EOT/EOM and FIM stop IDs include
llama.cpp's EOG spelling inference. This includes Qwen3's normally typed `</s>`:
the reference promotes it to a control/EOG token. External tokenizer directories
retain their current generation-config and special-token behavior.

The registry's optional `published_gguf` object is `{repo, file, revision}`;
the revision must be a 40-digit commit, and repository/file paths are checked.
`qwen3:0.6b-q4km` is a pinned example with `details.npu_ready=false`. Source
declaration does not publish a kernel set or establish hardware support. Full
CLI/bridge onboarding, including downstream weight-filename handling, belongs
to the later P1/P5/U2 stages and has not been verified here.

Evidence lives outside Git at
`/home/cyrus/code/phlegm/target/npu-dev/p5/s6-headers/<target>/` and
`/home/cyrus/code/phlegm/target/npu-dev/p5/s6-check-v5/`. Every header has a
pinned repository/file/revision, requested ranges, and saved-header SHA256.
The small corpus and reference IDs are checked in beside the Rust acceptance
test. Exact reproduction commands and reference limitations are in
[the utility README](../../utilities/gguf-tokenizer/README.md).

Measured acceptance and remaining disagreements are recorded below. The full
S6 publisher-equality gate remains open wherever the two independent sources
disagree; changing Phlegm to match one would break equality to the other.

Two publisher/reference conflicts were isolated, rather than treated as
implementation passes:

- All four Qwen publisher JSONs apply NFC. These particular GGUF headers omit
  `tokenizer.ggml.normalizer.nfc`. On decomposed `cafe\u0301` and the combining
  corpus, the raw GGUF/llama.cpp IDs differ from the JSON's. As a control,
  NFC-normalizing the input before native llama.cpp tokenization exactly
  reproduces the publisher IDs for both cases on all four targets.
- Qwen3 has `</s>` at ordinary-vocabulary ID 128247. Native llama.cpp promotes
  it to control/EOG, while the publisher JSON tokenizes the literal spelling
  as ordinary text. The added `eog-markers` case retains this difference.
  Gemma4 also has an ordinary `</s>` (ID 212), but llama.cpp explicitly restores
  it to ordinary status; Phlegm follows that exception.

No downloaded headers were changed to resolve either conflict. Explicit NFC
metadata is implemented and independently regression-tested for future files.
The owning plan needs a decision about source-conflict acceptance, or newly
published files whose metadata and publisher tokenizer agree. Normalizing all
Qwen inputs unconditionally would break the current GGUF/llama.cpp gate.

Final v5 results: all nine headers fetched; **198/198 native encodings** equal
the independent llama.cpp reference (11 cases x 2 special-token modes x 9
targets); **9/9 stop sets** equal. Python Jinja2 independently renders the
same three conversations for each target. Publisher results below compare
without automatic BOS/EOS, as rendered templates own their special tokens.

| Target | GGUF repository @ revision prefix | Native IDs | Publisher JSON |
| --- | --- | --- | --- |
| Llama 3.2 1B | unsloth/Llama-3.2-1B-Instruct-GGUF @ b69aef112e9f | 22/22 | 11/11 public mirror; original 401 |
| Llama 3.2 3B | unsloth/Llama-3.2-3B-Instruct-GGUF @ e7d0997e49c9 | 22/22 | 11/11 public mirror; original 401 |
| Llama 3.1 8B | lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF @ 8601e6db7126 | 22/22 | 11/11 public mirror; original 401 |
| Qwen3 0.6B | unsloth/Qwen3-0.6B-GGUF @ 50968a4468ef | 22/22 | 8/11; NFC and `</s>` conflicts |
| Qwen3 4B | unsloth/Qwen3-4B-GGUF @ 22c9fc8a8c77 | 22/22 | 8/11; NFC and `</s>` conflicts |
| Qwen3 8B | unsloth/Qwen3-8B-GGUF @ a6adef130ffb | 22/22 | 8/11; NFC and `</s>` conflicts |
| Gemma 4 12B | lmstudio-community/gemma-4-12B-it-GGUF @ 40b870babe39 | 22/22 | 11/11 |
| Qwen3.5 9B | lmstudio-community/Qwen3.5-9B-GGUF @ 1379f25c6b50 | 22/22 | 9/11; NFC conflict |
| Gemma 4 E4B | lmstudio-community/gemma-4-E4B-it-GGUF @ 99210a71094c | 22/22 | 11/11 |

Full revisions, filenames and header hashes are in the 56,709-byte fixture and
the external per-target `source.json` files. These publisher choices are
concrete candidates for P5 D5, not a claim that model/kernel gates exist.

Native library SHA256:
`f07f8a092074c6ba27e545974c8f3055fdcdcd92a894bda4e51cbb0259f197c6`.
Reference helper SHA256:
`dead92ddce9f78c36df11bd438bf5ac203738917b36eb0d6a784dac8024a6fc0`.
Final summary SHA256:
`342affba5f81ba27ea1e2154091acc769071e8645e39c2f686963997f861704d`.
Expected fixture SHA256:
`485a111a0be44daf41ab2c5e03ea930e92a8d8f9f5e911d8bbc25392af6dd9a8`.
`provenance.json` also hashes the installed build-12015 binary, native library
dependencies, inspected source/header, helper source, Rust probe and corpus.
The publisher check uses Python tokenizers 0.23.3; production Rust uses the
existing lockfile's tokenizers 0.23.2. No dependency was added.

The combined Python gate exits 1, intentionally retaining the 11 publisher
case mismatches. It must not be reported as a full S6 pass. The final core
suite passes 91 unit tests plus one release-provenance test; two bridge/NPU
prerequisite tests remain ignored. The external-header acceptance test is
opt-in and separately run with `PHLEGM_S6_HEADERS`.

The final explicit header test passes (1 test, all nine targets, 30.55 s),
including all 198 reference encodings, all stop sets, streaming decode with a
flush sentinel, exact corpus inputs and deterministic rendered conversations.
The final independent Jinja2 check passes 27/27. These are correctness check
durations on a shared host, not benchmark measurements.
