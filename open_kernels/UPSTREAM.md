# Sync ledger: phlegm <-> openflowlm-next

phlegm is the canonical home of `open_kernels/` and `specs/open-engine/`. openflowlm-next
(https://github.com/Atomic-Germ/openflowlm-next) receives the work as pull requests. Both trees use the same paths, so a
sync is a plain diff of these two directories.

## Rules

- Keep the manifest contract compatible. A change to the manifest schema, its `manifest_version`, or a recipe function's
  signature goes upstream together with the matching change to openflowlm-next's `src/open_qwen36` reader.
- Open one upstream PR per finished feature. Do not hold work for a large batch.
- Bring other contributors' commits in with `git cherry-pick -x`, so that their authorship stays.
- When review changes one of our PRs upstream, pull the merged version back in.

## Pulled in from openflowlm-next

| date | what | openflowlm-next ref |
|---|---|---|
| 2026-10-19 | `main`, full history of both dirs | `c05b00fd66fe16f2cbb03f2894c49cf5aea61d4a` |
| 2026-10-19 | PR #209 perf/granite-speedup (open, not merged upstream) | `814640b` |
| 2026-10-19 | PR #184 feat/k2-horizon-7b (open, not merged upstream) | `03a562e` |
| 2026-10-19 | PR #203 feat/bonsai2-dflash2-draft (open, not merged upstream) | `0cea11f` |
| 2026-10-23 | `main` through #214, including #216's Qwen3.5 0.8B/2B/4B specs and #218's spec correction | `478dbea3f00f868ac58205803cc537b402c6cad5` |
| 2026-10-23 | PR #193 dense prefill GEMM8, combined with #209's existing routes | `ca2cd600793d52a348a849a4512772c6f6d55faf` |
| 2026-10-23 | PR #194 K2 batched attention; #209 already carries its newer hi/lo implementation | `394f663d89b16e07e438cda1181296a44b220c5a` |
| 2026-10-23 | PR #207 Granite attention tiling knob | `ed5e3c6e1cadbe8974493aa04bb3873ae0753644` |
| 2026-10-23 | PR #213 Qwen3.8-27B precision and reference validation (upstream draft) | `5eda57ee98556df439c91e44c334325a82ab3362` |
| 2026-10-23 | PR #223 GGUF-direct, dense split shapes and Qwen3 size specs | `0c22864a5258019b7a885fb4be962ec62e3cc276` |
| 2026-10-28 | PR #225 Gemma 4 12B text decoder on two dense geometries (open, not merged upstream): its five `open_kernels/`/`specs/open-engine/` commits, cherry-picked with `-x` | `beccd4699d812642b63741255322d19a0c1dd738` |

## Not pulled in yet

| PR | why |
|---|---|
| None from the kernel PRs audited above | Runtime-only #185 and #219 are outside the kernel import. |

## Changes made here that upstream does not have

| date | change |
|---|---|
| 2026-10-19 | #203's `OPEN-DECODE-ONE-CONTEXT-DENSE` (qwen35 `dux.py`) collided with #209's requirement of the same ID (dense `dx.py`). In phlegm, #203's requirement is `OPEN-DECODE-ONE-CONTEXT-QWEN35`. Upstream, one of the two PRs must rename it the same way. |
| 2026-10-19 | The union of #209 and #203: both `CLEAR` knob lists in `export_qwen36_kernels.py`, and both the `rows` and `verify` manifest keys. |
| 2026-10-19 | `designs/gemm_q4/NOTES.md`: the gemm_q4 working notes from phlegm's old README. |
| 2026-10-23 | `bfp16_dit` uses manifest version **7** locally: #209 already reserves 4/5/6. This mapping must accompany any future upstream/runtime port; the original #193 reader patch alone reads only 1 through 4. |
| 2026-10-23 | Preserve Bonsai's t2 packing, merged images, verify passes and tail while adding #223's f32-scale twins. MoE GGUF twins remain disabled upstream until expert pack ops exist. |
| 2026-10-23 | Preserve the recovered attention cache key's head-width component and Qwen3-0.6B's 128-wide heads; #223's checked-in 0.6B spec instead used 64. |
| 2026-10-23 | Keep Bonsai/Hadamard on its existing attention and linear ABI; #213's precision selection is limited to the non-Hadamard geometry. Keep #193's default DIT route on measured q/k-normalized attention rather than promoting the recovered Llama rope-only route. |
| 2026-10-23 | Normalize config paths and compiler dependency parsing for Windows, and compare generated source text independently of line endings. Native/runtime tests declare missing prerequisites explicitly. |
| 2026-10-28 | #225's four Gemma 4 attention flags live in `recipes/dense.attn_flags`, phlegm's one flag source for `dx.py` and the attention gate; `recipes/gemma4.py` re-exports it. `attn.h` refuses `ATTN_QLOCAL` with the `ATTN_TREE8` / `ATTN_G5` scorers, which postdate #225's base and index q by global head. |
| 2026-10-28 | `dx.py` streams o's og elements only when they outnumber the norms' x depth (`OG_STREAM`), not at og > 2 as #225 does: phlegm's x fifo already holds og elements, so #225's condition moved Qwen3-14B (og 3, from #223) onto the new path. At depth 4 Gemma 4's global layer would also overflow its main cores. Upstream needs the same condition once #223 lands. |
| 2026-10-28 | `OPEN-FAMILY-GEMMA4` is defined in `specs/open-engine/spec.md`; #225's test traces the ID, but the PR does not add it. |

## Pushed out to openflowlm-next

| date | phlegm ref | upstream PR | state |
|---|---|---|---|

## Linux baseline build correction (2026-10-24)

The first native harness build exposed duplicate `AttnRowPatch`,
`attn_rows_table` and `attn_rows_apply` definitions in `harness/stream_patch.hpp`.
Removed the older second block and retained the KV-slice-aware implementation.
The canonical harness compiles against local XRT 2.27, and the compiler-backed
row-offset/slice regression passes. Setup, source identities and untested
hardware/model routes are in
[baseline readiness](../specs/open-engine/artlite-baseline-readiness.md).
No arithmetic optimization or ArtLite parity claim follows from this correction.

## Kernel sync integration (2026-10-23)

The import was refreshed from the live upstream refs above. Original contributor
history, filtered to `open_kernels/` and `specs/open-engine/`, is retained on the
local `import/open-kernels-20261023` branch. Its merge resolutions retain #209,
#203 and the newer precision/GGUF routes together. The main worktree also keeps
the local recovery below; it was merged against the pre-import HEAD rather than
overwritten by the prepared snapshot.

Embedding/build dependencies under `npu_offload/` are now included as source.
Generated binaries and model data were excluded. `npu_offload/PROVENANCE.md`
records their source and licensing. The companion reference, replay and build
tools are listed with source commits and hashes in
`utilities/kernel-sync-sources.json`.

Phlegm's Rust engine port is still pending. The corresponding OFLM C++ changes
for #193/#194/#207/#213/#223 are preserved as `upstream-patches/kernel-sync-*.patch`,
with exact bases in `upstream-patches/sources.json`. These are separate upstream
branch diffs, **not** a patch series or a ready-made combined C++ runtime. A
consumer of this combined kernel tree must implement all selected manifest
features, including the local version-7 DIT mapping. The offline checks here do
not claim a fresh NPU build or hardware acceptance of this combined tree.

Validation: the full offline suite passed with **1793 passed, 58 skipped and
8 deselected** (`python -m pytest -q --tb=short`, with BLAS/OpenMP threads set
to 1). The skips include explicitly optional OFLM runtime/native prerequisites.
All five new runtime companion patches parse successfully, and the worktree
passes `git diff --check` with CRLF line endings allowed.


## Local worktree recovery (2026-10-23)

The current Phlegm branch reconciles the uncommitted work from openflowlm-bonsai
(f92ac640) and openflowlm-issue212 (478dbea3). The source worktrees were not
modified or deleted.

- Bonsai: the Hadamard preps, sign tables, ternary packing and tests already exist
  here in a newer implementation: unpadded 2176-byte chunks, a rotated ternary
  head, native t2 GEMM and the one-context Qwen35 route. Keep those versions.
  The old worktree's generated .cc files and xh_signs.h are build products,
  not missing source. Its older 2560-byte layout and q4-copy prefill are not imported.
- Issue 212: merge native-Q8 embedding reads into the Python reference reader;
  add Llama 3.2 1B/3B and Qwen3 0.6B export specs; extend full-rotary Llama block
  attention without removing Granite's flash-attention route. Bring the associated
  regression tests, logits/benchmark tools and historical validation fixtures.
- The OFLM q4nx-build converter is intentionally excluded. openflowlm-k2conv
  changes only that converter, so none of its files are imported. Phlegm's own
  converter is unchanged.
- Phlegm does not vendor OFLM's C++ engine or model registry. Their companion
  changes are retained as upstream-patches/bonsai-runtime.patch and
  upstream-patches/issue212-runtime.patch; exact source heads and file lists
  are in upstream-patches/sources.json. The patches are separate alternatives
  against those source heads, not a series to apply blindly one after the other.
  Apply with 'git apply --3way <patch>' in the matching OFLM checkout and resolve
  against that checkout's current code before building.
- **Llama compatibility:** export the Llama batched-attention sets together with
  issue212's C++ high/low softmax-probability fix. An older single-BF16-probability
  reader is not the validated runtime for this route. The Q8 embedding reader and
  shared-prefill-buffer changes also need the companion C++ patch upstream.
- specs/open-engine/issue-212-validation.md records the original worktree's
  hardware results. They are not measurements of the recovered Phlegm branch.

- Recovery also fixes attention cache collisions: dense score/pv build directories
  include both M and head width. Llama 1B and Qwen3 4B share M=1024 but need
  different 64/128-wide heads. The AG16 experiment keeps 64-wide heads on the
  regular route because its 8 columns of 16 cannot tile N=64.


## Linux development validation (2026-10-24)

The Strix Halo Linux continuation is recorded in
[GGUF prefill validation](../specs/open-engine/gguf-prefill-linux.md).
Canonical changes add opt-in f32-scale GGUF GEMMs, an explicit Q8-repacked
head, and a separate `DENSE_FLASH_PREFILL=1` experiment for eligible Llama GGUF
sets. No measured default family table is promoted. Kernel images are rebuilt
with all streams, with independent arithmetic and model gates. The artificial
zero-history/window failure remains recorded; parity has not been reached.

Two additional **alternative** companion patches are based directly at
478dbea3: `linux-baseline-runtime.patch` preserves the measured 1–3/7 baseline;
`linux-runtime-pr209.patch` additionally reconciles PR209's runtime routes,
retaining local DIT7 and the Llama hi/lo sub-block merge. The latter's source
integration head/base are in `upstream-patches/sources.json`; it is not a patch
to apply after the baseline. It also refuses unsupported CPU fallback when an
experimental flash window is unavailable. Bonsai/Hadamard is still not part of
this combined runtime. DIT and additional family hardware acceptance stay open.

Fresh KV-sliced Llama 1B decode matches all 300 and 4137 prefill logits, plus
15 subsequent decode steps, byte for byte against the corresponding baseline.
Real-prefix model gates pass at 4137 positions and independently on the 47-token
4096 boundary slice. These development results do not establish recall quality,
HTTP performance, or clean-release benchmark parity.


## PR #225 Gemma 4 import (2026-10-28)

Imported from openflowlm-next PR #225 at head `beccd46` (base `24bd049`). The
five commits that touch `open_kernels/` or `specs/open-engine/` (`a1edd18`,
`523a96e`, `f7b928b`, `a5f310f` and `ec560e4`) were cherry-picked with `-x`
and keep their author. Their `src/` changes, and the five runtime-only commits
(`81c719c`, `ad128bb`, `5f86381`, `24dd54c` and `beccd46`), are in
`upstream-patches/pr225-gemma4-runtime.patch`: one standalone diff against
`24bd049`, recorded in `sources.json`. The patch passes `git apply --check
--3way` at that base. It was not built here.

Validation (Fedora, Strix Halo development checkout; offline, no NPU run):

- Full offline suite: `python -m pytest -q` gives 1953 passed, 64 skipped and
  8 deselected. The unmodified HEAD gave 1943 passed, with the same skips. The
  difference is `test_gemma4.py`'s 10 tests.
- All 24 shipped specs that derive have unchanged `to_dict`, build
  environments, attention flags and manifests except `build_key`, which hashes
  the changed sources. `qwen3-14b` refuses at `lm_head_q4` K=5120 before and after.
- `emit_mlir.py designs/dense/dx.py` emits byte-identical MLIR before and after
  for the 17 dense specs plus LFM2 1.2B, in both q4nx and GGUF-direct
  variants. Qwen3-14B was compared with `OPEN_KERNELS_UNVALIDATED=1`; #225's
  unmodified og condition changes it.
- `G4_KIND=global GEMV_SCALES_F32=1` (`dxg_f32`) compiles in 88 s. The fullest
  core has 12704 of 16384 B of program memory, matching upstream's report. The
  main GEMV cores place buffers up to 65280 of 65536 B; the attention cores up
  to 41744 B. Both local and global, q4nx and GGUF, MLIR emit.

Not established here: hardware arithmetic, model output, catalogue validation
and the q4nx route. See `OPEN-FAMILY-GEMMA4` and
`specs/open-engine/gemma4-text-kernels.md`.

### Gemma 4 block prefill and bring-up patches (2026-10-29)

Two more standalone diffs against #225's head `beccd46`, recorded in `sources.json`:

- `upstream-patches/gemma4-block-prefill-runtime.patch` adds the dense `gemm_block` fields
  `cd_preffn` / `cd_postffn`. With them, the block route's host chain reads the FFN-side
  sandwich norms from the consts rather than the file. Gemma 4 needs this because only the
  consts copy of `post_feedforward_layernorm` carries the folded `layer_output_scale`. Absent,
  the fields keep Gemma 3's file read.
- `upstream-patches/gemma4-bringup-dumps-runtime.patch` adds two bring-up options to
  `open_qwen36_cli`: `--dump-act all:<prefix>` and `--dump-kv <prefix>`.
- `upstream-patches/gemma4-compact-gguf-runtime.patch` makes `std_perm_gguf` follow the
  pool's chunk law: 6144-byte f32 scales, or 5120-byte BF16 ones rounded to nearest even.
  That is `linux-runtime-pr209.patch`'s branch without its Llama rotary permute, plus a
  `gguf_pack_test` case. Gemma 4's GGUF view needs it by default (`gemma4.compact_on`); a
  runtime without it packs 6144-byte chunks into a 5120-byte pool, so it must not run those
  kernel sets.
- `upstream-patches/gemma4-attn-block-runtime.patch` adds dense `attn_block` prep `qkvnorm_rope`
  for Gemma 4's block attention products. It reads the q / k norms from the consts and adds a
  weightless V norm, the score scale, a rotation per layer type (pair distance and frequencies),
  the sliding window and Q as a BF16 hi / lo pair. The window is honoured in the first row read,
  the softmax mask and the KV read-back. It adds `attn_window_test`, which checks the host half
  against an fp64 loop. Unlike the other three, this diff applies over `beccd46` plus those three
  (`sources.json` `applies_after`). A runtime without it refuses the prep at manifest load.
- `upstream-patches/gemma4-prefill-host-runtime.patch` speeds up the dense block route's host
  stages without changing a bit of output. Each sandwich join is one pass over a token row.
  The FFN activation's fp64 half runs while the up projection does, and `h = act * up` goes
  straight into the down projection's tiled x. `attention_npu`'s row loops and the K / V tilers
  are spread over the host threads. A new dense layer-major prompt runs every block's qkv3, then
  their attention, then the rest of each layer. It is on for `qkvnorm_rope` sets;
  `OFLM_OPEN_DENSE_LAYER_MAJOR=1` forces it for the other dense families, which have not been
  gated on it, and `=0` turns it off. It applies over `beccd46` plus the four patches above.
  Kernel sets need nothing new: an older runtime runs them block-major.
- `upstream-patches/gemma4-bfp16-prefill-runtime.patch` (2026-10-30) runs the dense block route's
  five projections on `designs/gemm_bfp16` (OPEN-PREFILL-BFP16). It reads manifest_version 4:
  the `bfp16_gemm` pack op and the `bfp16` step format. A bfp16 step must read a `bfp16_gemm`
  weight, and only dense routes may have one. The pack op also reads a GGUF tensor's own blocks.
  The x global is packed bfp16, and the c-tiled y is read back as `[N, T]` or as token rows.
  `act * up` is encoded straight into the down projection's operand. The prefill mode is now
  chosen in the GGUF view as well, which is where a GGUF-direct set keeps its routes. Gemma 4's
  recipe emits the bfp16 route as `gemm_block` (`fast`, the default) and the q4 route as `lean`.
  A runtime without this patch refuses those sets at version 4. `G4_BFP16=0` at export emits the
  q4 route alone, which such a runtime still loads. `bfp16.hpp` is the openflowlm-next Granite
  branch's (`07e635b`) plus a c-tile offset helper. The diff applies over `beccd46` plus the
  five patches above.
- `upstream-patches/gemma4-prefill-pipeline-runtime.patch` (2026-10-30) runs the dense
  layer-major prompt step-major: each GEMM step over every block, and the attention products
  over every block and kv head. One run is outstanding at a time. Each x, y and attention global
  has a `#1` twin, so the host fills or reads one while the NPU runs the other. Every block
  sees the same operations on the same data, so the output bits do not change.
  `OFLM_OPEN_STAGE_MAJOR=0` restores the block-at-a-time stages. On the bfp16 route the
  GELU-tanh act runs in fp32 on AVX2, fused with `act * up` and the encode (`gelu_bfp16_a`).
  That changes a few mantissas a layer; `OFLM_OPEN_GELU32=0` keeps the fp64 act. An optional
  `attn_block.kernels_shl` stacks Q's hi and lo halves into one scores product of twice the
  rows (`OFLM_OPEN_ATTN_SHL=0` keeps two). The `bfp16_gemm` packer no longer divides per
  block, and a pack-built weight zeroes only the bytes no op writes. `--prefill-mode accurate`
  is another name for `lean` in the engine and CLI; `oflm`'s own flag parser is outside this
  clone and still takes `fast|lean|bfp16`. Kernel sets need nothing new: without
  `kernels_shl` the scores take two dispatches. The diff applies over `beccd46` plus the six
  patches above.

`designs/dense/dx_attn.py` now takes its recipe module from `for_spec` and its flags from the
recipe's `attn_flags`, and gained `dx.py`'s ATTN_QLOCAL q loop. It emits byte-identical MLIR
for the 15 shipped dense specs with a block route that emit at all; `qwen3-14b` refuses at
`lm_head_q4` K=5120 before and after. The full offline suite gives 1954 passed, 53 skipped and
8 deselected. The 12 GPT-OSS and Qwen3.5-glue failures also fail at clean HEAD in this
environment.
