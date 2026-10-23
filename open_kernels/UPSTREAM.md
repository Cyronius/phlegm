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

## Pushed out to openflowlm-next

| date | phlegm ref | upstream PR | state |
|---|---|---|---|

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
