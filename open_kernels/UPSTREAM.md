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
| 2026-11-01 | `designs/trit_embed/`: the trit-embed-3l sentence-embedder encoder, moved here from an unversioned experiment. It is standalone, with its own build and host library, and uses no recipe, manifest or runtime ABI. |

## Pushed out to openflowlm-next

| date | phlegm ref | upstream PR | state |
|---|---|---|---|
| 2026-11-07 | `nanbeige-runtime.patch` `lmhead_q8` fp16 fix (OPEN-FAMILY-NANBEIGE) | #232 `feat/gguf-direct`, commit `e87740cc` with a `gguf_pack_test` case | pushed to the open PR's branch; [comment](https://github.com/Atomic-Germ/OpenFlowLM-Next/pull/232#issuecomment-6709022463) |

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
  block, and a pack-built weight zeroes only the bytes no op writes. Later the same day the
  host stopped four copies without changing a logit: rope angles once a position, up's own
  output pair, gate's h straight into down's input, qkv3's rows straight into q / k / v.
  The stacked softmax also adds hi and lo as it reads them. `OFLM_OPEN_PIPE_LOG=1` prints
  per-step host and wait times. `--prefill-mode accurate`
  is another name for `lean` in the engine and CLI; `oflm`'s own flag parser is outside this
  clone and still takes `fast|lean|bfp16`. Kernel sets need nothing new: without
  `kernels_shl` the scores take two dispatches. The diff applies over `beccd46` plus the six
  patches above.
- `upstream-patches/gemma4-q40-gguf-runtime.patch` (2026-10-30) gives `std_perm_gguf` a
  third chunk law. A 4608-byte pool chunk holds the bf16 `d` plane and the codes, with no
  min plane: the cores form `m = -8 d` themselves, which is exact in bf16 and so the value
  the 5120-byte law stores. Only Q4_0 tensors pack into it; anything else is refused at
  load. This is the pool law of the Q4_0-native GGUF view (`recipes/gemma4.py`, `G4_Q40=1`;
  `gemv_q4.h` `GEMV_Q4_0_NATIVE`, `gemm_q4_dequant.h` `GQD_Q4_0`). `gguf_pack_test` checks
  it against the 5120-byte law's `d` plane and codes. The diff applies over `beccd46` plus
  the seven patches above.
- `upstream-patches/gemma4-kvslice-runtime.patch` (2026-10-30) brings OPEN-ATTN-KVSLICE's
  per-core KV windows to this runtime. A kernel's manifest `kvslice` (2 or 4 rows an element,
  `attnpos` only, `rb` 1) makes every KV transfer below one row a window fill. Each fill's
  length is patched to the window, padded to whole elements, times its own words a row; its
  offset to its compiled in-row base plus the window's first row, so a sliding window works
  too. The KV buffer keeps `kvslice - 1` rows of slack. The `stream_patch.hpp` hunks are this
  repository's own header's, without its `rb_win`, slot and `attnposv` additions.
  `kvslice_patch_test` checks Gemma 4's local stream (four fills, a 1024-row window) and global
  stream (two fills) at positions 0 to 4095. The diff applies over `beccd46` plus the eight
  patches above.

- `upstream-patches/gptoss-runtime.patch` (2026-10-31) runs GPT-OSS from its GGUF
  (OPEN-FAMILY-GPTOSS). It adds the MXFP4 GGUF type, zero-extends a GGUF embedding row and an
  F32 norm into a padded width, and adds four pack ops: `gx_q8` (Q8_0 to q8 half-tiles with
  the bias in the pad), `gx_ug` and `gx_down` (MXFP4 experts to half-tiles, up and gate per
  band) and `gx_lmhead` (Q8_0 to `lm_head_q8`'s f32-scale chunks). They write the bytes
  `recipes/gptoss_pack.py` writes. The `gptoss_route` program step runs the router on the
  host between `gx_att` and `gx_moe` and patches `gx_moe`'s fills through `moeroute2`'s
  table. The diff applies over `beccd46` plus the nine patches above; a round trip over the
  Gemma 4 runtime reproduces the seven files exactly.

- `upstream-patches/gptoss-prefill-runtime.patch` (2026-10-31) gives GPT-OSS its native MXFP4
  experts and its block prefill (OPEN-FAMILY-GPTOSS, OPEN-PREFILL-GPTOSS). Pack ops `gx_ug_mx4`
  and `gx_down_mx4` write the experts as mx4 half-tiles (`designs/gemv_q4/gemv_mx4.h`: the
  GGUF's own nibbles, re-signed for the core's decode); `gp_bfp16` writes a GGUF tensor's rows
  into a `gemm_bfp16` B operand at any K that is a multiple of 64, optionally as the hi or lo
  half of a stacked operand; `gp_mx4` writes the experts as `gemm_bfp16` GB_B=mx4 tiles. The
  manifest's `gptoss_prefill` section names the four GEMMs (and a 128-row expert variant), their
  weights and the small tensors; the `gxexpert` patch re-points an expert GEMM's buffer-0 fills
  per dispatch, alternating two copies of the stream so the next run's expert is patched while
  the last one runs. `Core::step_gptoss_prompt` is the layer-major route `step_gemm_prompt` takes
  when the set carries it; `block_prefill_min` lets a set state its own crossover, which the
  engine uses. The diff applies over `beccd46`, the nine Gemma 4 patches and
  `gptoss-runtime.patch` (sha256 `8a5f808e`, unchanged); `target/npu-dev/gptoss-prefill-patch.sh`
  regenerates it and checks the chain reproduces the clone.

- `upstream-patches/flint-runtime.patch` (2026-10-31) runs Recursive Flint from its Q4_1 GGUF
  (OPEN-FAMILY-FLINT, `recipes/flint.py`). Pack op `fl_q4`: a rows x in_dim block of a Q4_0 / Q4_1
  tensor (trailing dims flattened into rows), per expert at a row / column step, into q4_1-law chunks
  (`std_perm_gguf`'s bytes: codes kept, d / m RNE to bf16) with a per-expert and a per-band destination
  stride, so a slot's gate and up bands interleave and the shared expert / dense MLP split into
  512-row slots. Program step `flint_route`: the sigmoid router over xm (no bias, top 6 renormalised)
  plus fixed shared-expert slots at weight 1, the weights beside xm and moeroute2 on the MoE dispatch.
  `gguf_pack_test` pins `fl_q4` to `std_perm_gguf`. The diff applies over `beccd46`, the nine Gemma 4
  patches, `gptoss-runtime.patch` (`8a5f808e`) and `gptoss-prefill-runtime.patch` (`f28a389e`), both
  unchanged; GPT-OSS's code paths are untouched. `target/npu-dev/flint-patch.sh` regenerates it and
  checks the chain reproduces the clone (`target/npu-dev/flint-runtime`).

- `upstream-patches/hyv3-runtime.patch` (2026-10-31) runs Tencent Hy-Memory-30B-A3B (hy_v3) from our
  pure Q4_0 GGUF (OPEN-FAMILY-HYV3, `recipes/hyv3.py`). Program step `hy_route`: the sigmoid router
  with the per-expert selection bias (`exp_probs_b` picks, the unbiased scores weigh, renormalised and
  scaled by `router_scaling_factor`), the shared expert's slot at weight 1, up to 16 slots in an
  f32[16] record beside xm, moeroute2's stock apply on `hy_moe`; a dense mode points the slots at the
  dense MLP's slices. The moe geometry's topk limit rises from 8 to 16; the device router's 8-slot
  record is checked in `Core::route` instead. No new pack op (`fl_q4`, `gx_lmhead`, `put` are the
  chain's). The diff applies over `beccd46`, the nine Gemma 4 patches, `gptoss-runtime.patch`,
  `gptoss-prefill-runtime.patch` and `flint-runtime.patch` (`323e4dc8`), all unchanged;
  `target/npu-dev/hyv3-patch.sh` regenerates it and checks the chain reproduces the clone
  (`target/npu-dev/hyv3-runtime`).

- `upstream-patches/lfm2moe-block-runtime.patch` (2026-11-03) is LFM2.5-24B-A2B's block prefill
  (OPEN-FAMILY-LFM2MOE; the recipe's default, `LFM2MOE_BLOCK=0` drops it). A new `gemm_block` kind `lfm` (mixer `conv` | `att`):
  `block_layer_lfm` runs the entry norm, the fused input GEMM, the conv's elementwise middle over the
  block with the carried two-row f32 state (or `attention_block` with a null gate), the out GEMM, the
  residual and `ffn_norm`; then a leading dense layer's up|gate and down GEMMs (`ffn_program`), or
  hy_route's router arithmetic over the block (`lfm_route_rows`) and `moe_block` with mbl's
  `[rbg][half][row]` y order (`moe_batch.y_layout`). `op_region` reads a one-tensor `fl_q4` op as the band
  law. No new pack op. The diff applies over `beccd46`, the nine Gemma 4 patches, `gptoss-runtime.patch`,
  `gptoss-prefill-runtime.patch`, `flint-runtime.patch` and `hyv3-runtime.patch`; it edits `moe_block`'s y
  read like `gemma4-moe-block-runtime.patch` does, so the two are alternatives on one clone, not a stack.
  `target/npu-dev/lfm2moe-block-patch.sh` regenerates it and checks the chain reproduces the clone
  (`target/npu-dev/lfm2moe-runtime`).

- `upstream-patches/hyv3-block-runtime.patch` (2026-11-04) is Hy-Memory-30B-A3B's block prefill (OPEN-FAMILY-HYV3;
  the recipe's default, `HYV3_BLOCK=0` drops it), over `lfm2moe-block-runtime.patch`: kind `lfm` with mixer `att`
  as it is, plus two optional route fields that LFM2-MoE's manifests do not carry -- `shared` (pool experts every
  token also takes at weight 1 after the router's top k: the ungated shared expert, pool expert 128) and `dense`
  (n slots = pool experts 0..n-1 at weight 1, no router: the dense layer's nine 768-wide slices). The experts run
  on LFM2.5's `mbl` stream unchanged (Hy's pool already has its byte law). `block_host.cpp` is not touched. No new
  pack op. The diff applies over `beccd46`, the nine Gemma 4 patches, `gptoss-runtime.patch`,
  `gptoss-prefill-runtime.patch`, `flint-runtime.patch`, `hyv3-runtime.patch` and `lfm2moe-block-runtime.patch`;
  `target/npu-dev/hyv3-block-patch.sh` regenerates it and checks the chain reproduces the clone
  (`target/npu-dev/hyv3-runtime`, which now carries both block patches; `hyv3-patch.sh` diffs the 10-31 tree,
  kept as `lfm2moe-runtime-base`).

- `upstream-patches/flint-block-runtime.patch` (2026-11-04) is Recursive Flint's block prefill (OPEN-FAMILY-FLINT;
  the recipe's default, `FLINT_BLOCK=0` drops it), over `lfm2moe-block-runtime.patch`: a gemm_block kind `flint` that
  runs through `block_layer_lfm` as kind `lfm` mixer `att`, plus `cd_sinks` (the learned per-head sinks, bf16 in the
  consts), `window` (513 on the sliding layers; only the KV rows the window reaches are read back), `nope` (the full
  layers rotate nothing) and `route.shared` (the shared expert's halves, pool experts 256 and 257, after every token's
  sigmoid top 6 at weight 1; no selection bias). Its own kind name, so an engine without these fields refuses a Flint
  set by name. `host::attention_block` gains `const float* sinks = nullptr` and honours `AttnGeom::window` -- small
  edits to the REFERENCE body; `block-host-attention-runtime.patch`'s vectorised body already has the window and needs
  the sink (max includes it, denominator gains exp(sink - max)) folded in before the two can stack. The experts run on
  LFM2.5's `mbl` stream unchanged. No new pack op. The diff applies over `beccd46`, the nine Gemma 4 patches,
  `gptoss-runtime.patch`, `gptoss-prefill-runtime.patch`, `flint-runtime.patch`, `hyv3-runtime.patch` and
  `lfm2moe-block-runtime.patch`; it is an alternative to the Hy and LFM3 block patches on one clone.
  `target/npu-dev/flint-block-patch.sh` regenerates it and checks the chain reproduces the clone
  (`target/npu-dev/flint-blk-runtime`; base snapshot `flint-blk-runtime-base`; the sequential `flint-runtime` clone is untouched).

- `upstream-patches/lfm3-block-runtime.patch` (2026-11-03) is LFM3-1.7B's block prefill (OPEN-FAMILY-LFM3,
  on by default, `LFM3_BLOCK=0` drops it), over `lfm2moe-block-runtime.patch`. A GEMM weight may be a byte `range` of the decode pool
  as it is (the gate | up slots' alternating 64-row bands and the mixer projections), so the pack plan and
  every decode kernel stay as they were. The `lfm` kind gains sandwich norms (`cd_postop`, `cd_postffn`), the
  elementwise sigmoid gate (`gate`: q | gate | k | v), a sliding `window`, the layer type's own
  `rope_inv_freq`, `ffn_slots` (gate | up and down GEMM pairs whose outputs the host sums) and any conv tap
  count with a (taps - 1)-row f32 state. `target/npu-dev/lfm3-block-patch.sh` regenerates it and checks the
  chain reproduces the clone (`target/npu-dev/lfm3-runtime`).

- `upstream-patches/block-host-attention-runtime.patch` (2026-11-04, sha256 `7a633bf0...`) replaces
  `host::attention_block` (the block routes' host attention: the `lfm` kind, LFM3 and LFM2.5) with a
  vectorised, tiled version of the same arithmetic in the same order: bit-identical output and cache rows,
  10-20x faster per block past a few thousand cached rows. Over `lfm3-q40-runtime.patch`; the
  `lfm3-runtime` clone now carries it, so regenerate `lfm3-block-runtime.patch` against a clone without it
  (`target/npu-dev/attn-bench/block_host.lfm3-runtime.orig.cpp` is the file before). The
  `lfm2moe-runtime` clone carries the same body; `block-host-attention-lfm2moe-runtime.patch` is that
  variant (the lfm2moe-block point of the chain; it also reproduces the `hyv3-runtime` clone over
  `hyv3-block-runtime.patch`, so `hyv3-block-patch.sh`'s round trip needs it appended). Flint's
  `flint-block-runtime.patch` was regenerated with the same body plus its `sinks` argument (bit-identical to
  Flint's scalar sink version in the CPU harness: windows 513, NoPE, partial blocks). Evidence:
  `.plans/long-prefill.md`.

- `upstream-patches/lfm3-q40-runtime.patch` (2026-11-03) is LFM3-1.7B's decode bytes (OPEN-FAMILY-LFM3,
  on by default; `LFM3_Q40=0`, `LFM3_HEAD=f32` select the q4_1 pools and f32-scale head), over `lfm3-block-runtime.patch`. `fl_q4` gains the Q4_0-native 4608-B chunk
  law (bf16 d plane, m = -8 d on the core; byte-identical to `std_perm_gguf`'s Q4_0 pool, checked in
  `gguf_pack_test`) and a new pack op `gx_lmhead_bf16` packs the tied Q8_0 head with RNE bf16 per-32 scales
  (8704-B chunks). `target/npu-dev/lfm3-q40-patch.sh` regenerates it and checks the chain.

- `upstream-patches/gemma4-moe-runtime.patch` (2026-10-31) runs Gemma 4 26B-A4B from its QAT
  Q4_0 GGUF (OPEN-FAMILY-GEMMA4-MOE). Two pack ops: `g4_q40` (a Q4_0 tensor, 2-D or the
  experts' 3-D, into Q4_0-native chunks in `std_perm`'s band law, with rows and columns padded
  by zeros, an expert stride and a band stride for the interleaved gate | up bands) and
  `g4_lmhead_q8` (the Q6_K tied embedding requantised to `lm_head_q8`'s bf16-scale chunks).
  Two program steps around the `gm_moe` dispatch: `gemma4_route` (the router on the host over
  the attention residual, top 8 renormalised times `per_expert_scale`, `pre_ffw_norm_2` into
  the expert input, `moeroute2`'s table patched) and `gemma4_combine` (`post_ffw_norm` of the
  dense MLP's and the experts' normed outputs, with the folded layer scalar, into `xres`).
  `host_step` dispatches them beside `moeroute2`; `arm_route_records` now arms `moeroute2`
  steps only. `gguf_pack_test` checks `g4_q40` against `std_perm_gguf` and with padding,
  expert and band strides, and `g4_lmhead_q8` within half a step. The diff applies over
  `beccd46` plus the nine Gemma 4 patches above (not over `gptoss-runtime.patch`, which is
  a separate clone); applying the ten in order reproduces the runtime tree exactly.

- `upstream-patches/gemma4-moe-decode-runtime.patch` (2026-11-01, sha256 `e1cd6b79...`) is Gemma 4
  26B-A4B's decode work over `gemma4-moe-runtime.patch` (OPEN-FAMILY-GEMMA4-MOE). `gemma4_route`'s
  router runs over the host threads with eight partial sums per logit (it was one serial double
  chain per expert: 7-9 ms of a 115 ms step; now about 1 ms; the logits of a full gate run are
  byte-identical). `gemma4_combine` takes `moe_parts` (default 1): m is the sum of that many
  f32 [hidden] partials from `a_moe`, for the 8-core `gm8` probe. `OFLM_G4_SPLIT=1` prints the
  route / combine host split per step. No pack-op or kernel ABI change; the merged layer image
  (`designs/gemma4_moe/dxm.py`) needs nothing new from the runtime (two kernels on one context,
  as `lx` / `ax` already are). Applies over `beccd46`, the nine Gemma 4 patches and
  `gemma4-moe-runtime.patch` (`target/npu-dev/gemma4-moe-runtime`).

- `upstream-patches/gemma4-moe-block-runtime.patch` (2026-11-03, sha256 `e65b29f3...`) is Gemma 4
  26B-A4B's block prefill over `gemma4-moe-decode-runtime.patch` (OPEN-FAMILY-GEMMA4-MOE). A dense
  `gemm_block` may carry `g4_moe` (the route / combine steps and a `moe_batch` of
  `designs/gemma4_moe_batch` streams) and an empty `attn_kernel` when it has an `attn_block`
  (the 26B has no per-token dxB build); `step_gemm_block_layer` then closes the layer with
  `g4_moe_tail` (t1, the host router over the f32 residual, `moe_block` reading mb4's y order, the
  combine). The CLI gains `--force-ids-file` (teacher forcing in the decode loop). `op_region` accepts a contiguous `g4_q40` op, and the staged schedule is off for the
  kind. Needs the token-batched expert kernels (built by default; `G4_MOE_BATCH=0` drops them); without them the manifest has no block route and prefill
  is sequential, as before. Applies over `beccd46`, the nine Gemma 4 patches,
  `gemma4-moe-runtime.patch` and `gemma4-moe-decode-runtime.patch`
  (`target/npu-dev/gemma4-moe-runtime`).

- `upstream-patches/qwen35moe-gguf-runtime.patch` (2026-10-31) runs Qwen3.5 / 3.6 MoE text
  models (Ornith 1.5 35B-A3B, Nex N2.5 mini, KAT Coder V2.5 Dev) from a llama.cpp `qwen35moe`
  GGUF (OPEN-FAMILY-QWEN36MOE-GGUF). Seven pack ops write the q4nx ops' bytes in the q4_1 law:
  `q35_mat`, `q35_stripes`, `q35_down`, `q35_lmhead`, `q35_t`, `q35_vec` and `q35_conv`. Q4_0,
  Q4_1 and Q4_K blocks keep their codes, with the scale pair rounded to bf16 (RNE). Q8_0, Q5_0
  and Q6_K blocks are requantized. The ops undo the converter's per-head `[q | gate]`
  interleave and its tiled linear-attention value heads. On a GGUF, the MoE block route's
  host stages read their small weights from the packed consts. `q35_pack_dump` writes packed
  bytes for `open_kernels/model/qwen35moe_gguf_check.py`. The diff applies over `beccd46`
  plus the nine Gemma 4 patches and the recorded `gptoss-runtime.patch` (sha256 `8a5f808e`);
  the clone is `target/npu-dev/qwen35moe-runtime`, and a round trip reproduces its tree.
  Kernel sets that carry the new `gguf` section need this runtime to parse their manifest.

- `upstream-patches/saluki-runtime.patch` (2026-11-06) runs Underdog Saluki 27B, a Qwen3.8-27B
  published only as an IQ2-mix llama.cpp `qwen35` GGUF, on the all-q8 Qwen3.5 dense view
  (OPEN-FAMILY-SALUKI). `gguf_iq.hpp` decodes IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ3_S, IQ4_XS and Q2_K a 32-value block at a time as `ggml-quants.c` does; its grid tables
  (`gguf_iq_tables.hpp`) are generated from llama.cpp `a34de8618b25`'s `ggml-common.h` (MIT).
  A block with one scale and integer codes keeps those codes in the new `q35_q8` op (q8_perm
  half-tiles), only its scale rounded to bf16; other blocks are requantized to int8. `q35_mat`
  gains `split` hi / lo, the block route's exact q4_1 split of the same q8 values. The patch also
  gives the dense block route the 27B's decode state ABI (`attn_kv_dtype` `bfloat16_pair`,
  `linear_conv_dtype` `float32`): without it the route's prefill logits are right and the decode
  that follows is not, which applies to any Qwen3.8-27B set on this lineage. Applies over
  `beccd46`, the nine Gemma 4 patches, `gptoss-runtime.patch` and
  `qwen35moe-gguf-runtime.patch`; the clone is `target/npu-dev/saluki-runtime`, and a round trip
  reproduces its tree.

- `upstream-patches/nanbeige-runtime.patch` (2026-11-07) serves Nanbeige4.3-33B, a dense Llama run
  from our own pure Q4_0 GGUF on the dense recipe's q4_1-law GGUF view (OPEN-FAMILY-NANBEIGE). It is a
  diff over `reference-runtime-next` (`478dbea` plus `linux-runtime-pr209.patch`, the lineage that
  already un-permutes llama.cpp's Q / K rows in `std_perm_gguf`), not over `beccd46`. It fixes the
  raw-GGUF Q8_0 head path of `lmhead_q8`, whose own fp16 widening wrapped a subnormal scale to inf /
  NaN (0.6% of this head's blocks), with a `gguf_pack_test` case that failed 4096 / 4096 before; and it
  adds `gguf_pack_dump`, the no-XRT dump `open_kernels/model/nanbeige_gguf_check.py` reads. The clone is
  `target/npu-dev/nanbeige-runtime`, and a round trip reproduces its tree.

- `upstream-patches/xing-runtime.patch` (2026-11-08) serves Xing4.0-29B-A4B, a DeepSeek-V3 decoder (MLA,
  routed SiLU experts) with DeepSeek's mHC residual, from our own pure Q4_0 GGUF (OPEN-FAMILY-XING). It is a
  diff over the Nemotron-H clone (`beccd46` plus the Gemma 4 / gemma4-moe chain plus
  `nemotronh-runtime.patch`), whose host-step program, one-image expert stream and `nh_q40` pack op it
  reuses. It adds the `xg_*` host steps: mHC pre / post over four f32 residual streams in the `xres` global,
  the MLA glue (q / kv norms, interleaved YaRN RoPE, an f32 KV row of latent plus rope key per position, the
  absorbed attention through `attn_k_b` / `attn_v_b`), the sigmoid router with the shared expert as one more
  slot (`nhroute` with a pool base), and the stream mean before the head. It also adds `band_stride` to
  `nh_q40`, for the gate | up band interleave, and `xing_pack_check`, a CPU readback whose samples
  `open_kernels/model/xing_gguf_check.py` compares with gguf-py. The clone is `target/npu-dev/xing-runtime`,
  and a round trip reproduces its tree.

`designs/dense/dx_attn.py` now takes its recipe module from `for_spec` and its flags from the
recipe's `attn_flags`, and gained `dx.py`'s ATTN_QLOCAL q loop. It emits byte-identical MLIR
for the 15 shipped dense specs with a block route that emit at all; `qwen3-14b` refuses at
`lm_head_q4` K=5120 before and after. The full offline suite gives 1954 passed, 53 skipped and
8 deselected. The 12 GPT-OSS and Qwen3.5-glue failures also fail at clean HEAD in this
environment.
