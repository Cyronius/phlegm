# Migrate the open engine from FLM 1.0.2 xclbins to FLM 1.0.3

**Status:** ABANDONED (2026-09-01) — bad trade, see Result below. Kept for the
record; do not resume without new information (e.g. a faster 1.0.3 engine
release) that changes the Result finding.
**Follows:** [[qwen36-1.0.3-format-support.md]] — that plan solved the 1.0.3
*file*-format read (Q4_K dequant, reorder-undo, Python + Rust), but deliberately
stayed read-side-only: a 1.0.3 file is dequantized in full on the CPU and fed
through the **unchanged 1.0.2 pool builders and kernels**. This plan is the
sequel it punted: actually run the 1.0.3 kernels, which means packing pools the
way the *new* kernels expect, not full CPU dequant.

## Why now
Currently pinned to v1.0.2 (`tools/get-kernels.ps1:17`, hard comment: "do not
bump without re-capturing the ELFs"). FLM v1.0.3 ("Higher-Accuracy Qwen3.5 &
Qwen3.6-MoE Weights", released 2026-08-27) re-quantizes Qwen3.6-MoE from Q4_1 to
Q4_K. Any model FLM publishes going forward — including Josh's own 27B, the
original motivating goal ([[pruned-qwen36-flm-project]]) — ships in this format.
Staying on 1.0.2 kernels means the open engine is validated against a
weight format FLM itself is moving away from.

## Result: killed after step 1 — FLM's own 1.0.3 engine is ~3x slower
Got a real 1.0.3 setup running (MSI portable-extracted to `C:\flm-test-103`;
genuine Q4_K `model.q4nx`, 21,044,871,704 B, downloaded from HF
`FastFlowLM/Qwen3.6-35B-A3B-NPU2` branch `flm_q4k_high_precision` into
`C:\Users\josha\.flm-103\models\Qwen3.6-35B-A3B-NPU2`; old 1.0.2 file backed up
as `model_q4_1_1.0.2_backup.q4nx` in the live model dir). Header-verified
genuine Q4_K (4736 B chunks) on the **actual A3B MoE architecture** — the file
format work in [[qwen36-1.0.3-format-support.md]] had only validated Q4_K on a
small dense 0.8B model before this. `config.json` now also carries an explicit
`flm_version` field, a cleaner format-detection signal than chunk-size sniffing
if this is ever revisited.

Then measured FLM 1.0.3 itself (its own `/v1/chat/completions` `decoding_speed_tps`/
`prefill_speed_tps`, 3 requests, same machine, same prompts used for the 1.0.2
baseline) end-to-end on the base 40L model:

| | FLM 1.0.2 (Q4_1) | FLM 1.0.3 (Q4_K) |
|---|---|---|
| decode | ~6.8 tok/s | ~2.28 tok/s |
| prefill | ~3.1 tok/s | ~1.0 tok/s |

**~3x slower, on FLM's own closed engine, same hardware.** The kernel-size
growth noted below (`mm`/`dequant_mm`/`layer`/`lm_head`) isn't just a relabeled
format — it's real extra per-matmul work to unpack Q4_K on-device. Also hit one
`ERT_CMD_STATE_TIMEOUT` on the very first decode step (retry succeeded,
same flaky-first-attempt pattern already logged for 1.0.2's interval-3 work) —
noted but not chased, moot given the speed finding.

**Decision (Josh, 2026-09-01): bad trade.** Porting phlegm to 1.0.3 kernels
would inherit whatever cost is baked into that on-device Q4_K dequant path —
phlegm would likely still beat FLM proportionally (it beat 1.0.2 by ~34%), but
would plausibly land around ~3 tok/s decode instead of today's ~9. Not worth
the RE cost (steps 2-10 below, never started) for an accuracy gain at that
speed cost. Staying on 1.0.2 kernels/pools.

**Cleanup owed:** `C:\flm-test-103\` (MSI extraction), `C:\flm-103-src\`
(downloaded MSI + logs), and `C:\Users\josha\.flm-103\` (21 GB model + 962 MB
vision weight) are still on disk — ~22 GB, harmless but unused; delete when
convenient. The 1.0.2 backup (`model_q4_1_1.0.2_backup.q4nx`, next to the live
`model.q4nx`) is worth keeping regardless, as a second safety copy alongside
the pre-existing `.orig`.

## Recon done this session — scope is smaller than it looks
Compared `.xclbin` sizes, v1.0.2 tag vs v1.0.3 tag, same model dir
(`ROCm/FastFlowLM` GitHub, `src/xclbins/Qwen3.6-35B-A3B-NPU2/`):

| kernel | 1.0.2 bytes | 1.0.3 bytes | changed? |
|---|---|---|---|
| `attn.xclbin` | 315,900 | 315,900 | **identical** |
| `GateDeltaNet_prefill.xclbin` | 228,636 | 228,636 | **identical** |
| `conv.xclbin` | 386,380 | 386,380 | **identical** |
| `mm.xclbin` | 186,684 | 194,364 | +7,680 B |
| `dequant_mm.xclbin` | 190,652 | 198,844 | +8,192 B |
| `layer.xclbin` (fused decode) | 356,716 | 406,124 | +49,408 B |
| `lm_head.xclbin` | 269,580 | 274,123 | +4,543 B |

The three kernels that operate purely on **activations** (attention, conv,
DeltaNet recurrence — never touch quantized weights) are byte-identical. Only
the kernels that **dequantize weights on-device** (`mm`, `dequant_mm`, the fused
`layer` kernel, `lm_head`) grew — consistent with them gaining a Q4_K unpack
path alongside (or instead of) Q4_1. This means:
- `attn`, `GateDeltaNet_prefill`, `conv` almost certainly need **no
  re-verification** — same bytes as the already-verified 1.0.2 kernels. Worth
  one quick confirming check (byte-diff a captured control-ELF submission
  against each, same "wrong kernel times out" oracle already used in M1) but
  not a re-derivation.
- `mm`, `dequant_mm`, `layer`, `lm_head` are the real work: their pool inputs
  must now be packed in **Q4_K tiling**, not Q4_1, and that tiling has to be
  reverse-engineered the same way the 1.0.2 pool tiling was (M3:
  `build_pools.py`, ctrlcode-address decoding, one-hot probes).

## What's already usable vs what's new
**Reusable as-is:** the whole capture/replay/verify methodology
([[flm-capture-oracle]]) — proxy DLL (`tools/seq-capture`), `prefill_capture.ps1`,
`analyze_op.py`, kernel-ID-by-timeout mapping, byte-exact-output verification.
None of that is 1.0.2-specific.

**Already solved:** the *file*-format side (Q4_K dequant math, reorder-undo) —
`tools/kernel-interp/q4nx_v103.py`, `npu-engine/src/q4nx.rs` (`fmt=="q4k"`).
Validated against the reference converter and a real Qwen3.5-0.8B Q4_K file.

**New work, in dependency order:**
1. **Get real 1.0.3 artifacts.** Portable-extract the 1.0.3 MSI the way the
   1.0.2 exe was extracted (flm.exe already self-reports "latest v1.0.3
   available" — download link is in its own startup log). Back up the current
   local model (`model.q4nx.orig` already exists as a safety copy) before
   re-running FLM's downloader — the catalog (`model_list.json` on the
   currently-installed 1.0.2 exe) already advertises `quantization_level:
   Q4_K_S` for `qwen3.6-moe:35b-a3b` on the mutable `flm_quant` HF branch, so a
   re-download under the same path now likely pulls Q4_K weights even without
   upgrading the exe — confirm this before assuming a version bump is required
   for the file, separately from the kernel/engine bump.
2. **Capture a 1.0.3 run.** Same proxy-DLL harness, pointed at the 1.0.3
   flm.exe + a genuine 1.0.3 model. Sanity-check op-stream structure
   (`seq_struct.py`) against a 1.0.2 capture — same op types/counts expected
   (Tier-0 already showed op scheduling is config-driven, not
   version-driven).
3. **Confirm the 3 unchanged kernels need no rework** — replay a captured
   1.0.3 `attn`/`conv`/`GateDeltaNet_prefill` control ELF against the existing
   verified 1.0.2 xclbin bytes (they're identical files) and against the
   1.0.3-labeled copy; should both complete/match trivially.
4. **Reverse-engineer the Q4_K pool tiling** for `mm`/`dequant_mm`/`layer`/
   `lm_head`. Same approach as the original: decode ctrlcode addresses,
   diff a captured resident pool buffer against the known file-order Q4_K
   layout (already read-verified), find the permutation. Layer kernel pack
   format (`[ln|postln|sgate|router]`) and side/state buffers are pure
   activation-side — expect unchanged; the 512 MB weight **pool** layout is
   the unknown.
5. **Update pool builders** (`tools/kernel-interp/build_pools.py`,
   `l30_build.py`, `pools_only_l40.py`, and `npu-engine/src/pools.rs`) to pack
   Q4_K-tiled pools, replacing the Q4_1 path — no dual-format support to
   maintain (decided: no 1.0.2 compat requirement). 1.0.2 captures stay on
   disk only as a cross-check oracle during the port, not as a shipped path.
6. **Byte-verify** each of the 4 changed kernels against fresh 1.0.3 captures,
   same standard as M1 (byte-identical NPU output vs FLM's own capture, real
   data, not just "doesn't crash").
7. **Re-verify the full chain** — prefill + decode byte-exact vs FLM 1.0.3's
   own captured boundary states, the same way [[flm-capture-oracle]] validated
   1.0.2 (`prefill_capture.ps1 -Planes both`).
8. **Confirm the interval-3 fix still holds under 1.0.3.** The whole reason
   this open engine exists ([[pruned-qwen36-flm-project]]) is FLM 1.0.2's
   engine NaN-collapsing on `full_attention_interval=3`. Q4_K is a weight
   requant, not an architecture change, so the bug is expected to reproduce
   identically on 1.0.3 — but must be checked, not assumed, since it's the
   project's core claim.
9. **Update `tools/get-kernels.ps1`** — make the FLM version a real supported
   parameter once 1.0.3 is verified (not just relax the pin), default to
   whichever is fully verified at the time.
10. **Re-benchmark** (this session's `l40-run` vs FLM numbers) on 1.0.3 for an
    apples-to-apples "higher-accuracy weights" comparison.

## Decisions (confirmed with Josh, 2026-09-01)
- **No 1.0.2 compat to preserve.** Cut over — step 5 (pool builders) *replaces*
  the Q4_1 packing path, it doesn't grow a second one. 1.0.2 captures stay on
  disk as the correctness oracle for cross-checking during the port
  ([[flm-capture-oracle]]) but the shipped pool-builder / kernel config target
  only 1.0.3 once verified. `get-kernels.ps1`'s `-Ref` simplifies to a straight
  bump, not a dual-supported parameter.
- **Order: base 40L model first, interval-3 pruned (30L) second.** Get the 40L
  base model byte-verified end-to-end on 1.0.3 kernels/pools before touching
  the pruned config at all. The pruned model is 30 layers cut from the same 40
  ([[pruned-qwen36-flm-project]]); once the 40L pool-tiling + kernel
  verification is solid, re-running it against a 30-layer slice is a config
  change (`num_hidden_layers`), not new RE — same relationship 1.0.2's M2→M3
  had (get the base model right, then the interval-3 slice was schedule
  changes on already-proven kernels).

## Risks
- The Q4_K pool tiling could differ nontrivially from Q4_1's (different
  metadata layout — `s8`/`m8`/`S`/`M` vs plain `q4_1` scale/min — may force a
  different tile shape, not just a relabeled one), i.e. step 4 could be as
  involved as the original M3, not a small delta.
- No 1.0.3 sample of the **32.7GB-class model** family has been captured yet;
  the file-format work validated Q4_K on a small dense 0.8B, not on this A3B
  MoE (same caveat already logged in [[pruned-qwen36-flm-project]] for the file
  side — now applies to the kernel side too).
- Re-downloading the model risks losing the current known-good 1.0.2
  `model.q4nx` if the backup step is skipped — must copy `model.q4nx` to a
  clearly-named 1.0.2 backup (distinct from the already-existing `.orig`,
  which may itself get overwritten by tooling that assumes `.orig` = pristine)
  before touching the download.
