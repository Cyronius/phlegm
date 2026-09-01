# Support FLM 1.0.3 (Q4_K) q4nx model files in the open engine

**Status:** Python reader BUILT + VALIDATED (2026-08-30). Rust port in progress.
**Date:** 2026-08-30.

## Results (Python)
- `tools/kernel-interp/q4nx_v103.py` — Q4_K dequant, q8 column-major lm_head,
  and the full reorder-undo map. `q4nx.py::Q4NX` now auto-detects format (4736 vs
  5120) and dispatches; `full_forward.py` routes through it (q4_1 byte-identical).
- **Unit validation** (`tools/q4nx-convert/validate_v103.py`): drives the
  reference converter's REAL `_pack_q4k`/`_pack_q8nx`/`_refit_one_side` with
  hand-built (t,u,q) triples, reads back — Q4_K layout rel 3.8e-3 (refit floor),
  q8 3.6e-4, every reorder-undo at the floor. A wrong layout/reorder would be
  rel~1, so the 250-1000x margin is the proof.
- **Integration validation**: `transcode_102_to_103.py` turns the verified
  `model_3LiF.q4nx` into a genuine 1.0.3 file via the reference packer + forward
  reorders. `full_forward` on both gives the SAME argmax (1798) and logit
  corr 0.972 (the 0.028 is the transcoder quantizing twice, not a read bug).
  Per-tensor read-back (`compare_readback.py`) confirms this: EVERY tensor
  matches in logical order at rel <= 6.7e-2 / maxabs <= 0.02 (uniform requant
  noise); the pure-permutation tensors (`ssm_a` f32, `embed`/norms bf16) match at
  EXACTLY 0.0 — isolating reorder correctness from quant noise.

## CORRECTION to the validation plan below
The "synthetic-GGUF -> reference converter" path does NOT work: **gguf 0.19
cannot ENCODE Q4_K** (`unpack` treats Q4_K as read-only; the reference converter
only emits 1.0.3 when its SOURCE GGUF is already Q4_K). Validation instead drives
the reference `_pack_q4k` directly with (t,u,q) triples — no ggml Q4_K encoder,
no 16.5 GB download. The reference converter is vendored (gitignored) at
`tools/q4nx-convert/reference/`.
**Goal:** make the open NPU engine (`npu-engine/`) and the Python tooling
(`tools/kernel-interp/`) *read* q4nx model files produced by the **newer FLM
1.0.3 converter** (`FastFlowLM/FLM_Q4NX_Converter`, `default_tensor_type: Q4_K`),
in addition to the 1.0.2 (q4_1) files they read today.

This is a **read-side** change only. The installed engine is still FLM v1.0.2 and
the open engine reuses the 1.0.2 closed kernels; a 1.0.3 file is dequantized to
plain `[out,in]` fp weights and fed through the *existing, unchanged* pool
builders and kernels. Nothing downstream of dequant changes.

---

## Why this is needed

The public reference converter has moved to the 1.0.3 format. Any model FLM
publishes going forward ships in 1.0.3. Cyrus's own 27B will be distributed this
way. The open engine must consume those files directly, not only the 1.0.2 files
that happen to be on disk today.

## Confidence and the one hard constraint

The byte layout below was read **directly from the reference converter source**
(cloned locally during planning) and **independently re-derived by a second
agent** — the two reads agree byte-for-byte. Confidence is high on everything
marked *[src]*.

**The hard constraint: there is no 1.0.3 sample file on disk, and FLM 1.0.2
cannot produce or read one.** So validation cannot be "diff against a known-good
file." See Validation below for the path that resolves this without a 16.5 GB
download. Two facts are *inferred*, not read, and must be confirmed on the first
real 1.0.3 file:
- `ssm.state_size = 128` (inferred from `4096 = q2·g16·p128`; read the GGUF
  field to confirm).
- whether stored `ssm_a` is `−exp(A_log)` (pre-baked, as 1.0.2 stores it) or raw
  `A_log`. The 1.0.3 converter stores `reorder(GGUF ssm_a)` verbatim with no
  `−exp`, so this depends on what the GGUF carries.

---

## The 1.0.3 FILE format, byte-exact

Same safetensors container as 1.0.2 (`<u64 header-len><json header><data>`), same
I8 tensor shape `[out//32, in//256, CHUNK]`, same **raster** tile order (tile
`(p,u)` → rows `32p..`, cols `256u..`), same 32×256 tile. **No `__metadata__`
version field** — `save_file` is called with no metadata *[src]*. Three things
differ from 1.0.2.

### 1. Quant scheme: q4_1 (5120 B) → Q4_K-refit (4736 B) *[src]*

One chunk = one 32-row × 256-col tile, concatenation order literally
`[s8 | m8 | q | S | M]`:

| bytes | field | count / dtype | index formula |
|---|---|---|---|
| `[0:256]`     | `s8` per-group scale | 256 × uint8 | `g*32 + r`  (g=col//32 ∈0..7, r∈0..31) |
| `[256:512]`   | `m8` per-group min   | 256 × uint8 | `g*32 + r` |
| `[512:4608]`  | `q` nibbles          | 4096 B      | byte `= C*16 + (R//2)`, C=col 0..255, R=row 0..31; **low nibble = even row, high = odd** |
| `[4608:4672]` | `S` super scale      | 32 × bf16   | row `r` (1 super-block/chunk) |
| `[4672:4736]` | `M` super min        | 32 × bf16, **stored negated** | row `r` |

- group = 32 in-cols (8/tile); super-block = 256 in-cols (1 per row per chunk).
- **Dequant:** `value(r,c) = S[r]·s8[g,r]·q(r,c) + M[r]·m8[g,r]`, `g=c//32`,
  `q`∈0..15, `s8/m8`∈0..255, `S`,`M` bf16, `M` already negative so its term
  subtracts. (ggml `w = t_j·q − u_j` with `t_j=S·s8`, `u_j=(−M)·m8`.)
- The 4736 vs 4608 "mystery": FLM re-fits ggml's 20 B/super-block metadata
  (12 B packed 6-bit + 2 B d + 2 B dmin) into 8×uint8 s + 8×uint8 m + bf16 S +
  bf16 M = 20 B/super-block → 256 s8 + 256 m8 + 64 S + 64 M + 4096 q = 4736.
- The refit is lossy-on-purpose at pack time; **reading is exact to the stored
  values** — we do NOT reproduce the refit search, just read `S·s8`, `M·m8`.

### 2. Intra-chunk order: 16-lane interleave → plain column-major *[src]*

1.0.2 nibbles used `(r//16)*4096 + bc*512 + i*16 + (r%16)`. 1.0.3 is plain
column-major `C*16 + R//2`. Different, and it also applies to the lm_head bytes.

### 3. lm_head: still Q8_0 / 8704 B, but column-major *[src]*

`[scales | data]`, no min: scales = 256 × bf16 (512 B, order `g*32+r`), data =
8192 × int8 (**column-major** `c*32+r`). `value = scale[g,r]·q`, `g=c//32`. Same
8704 B as 1.0.2 → **cannot detect format from lm_head**; detect from the Q4_K
chunk size (4736 vs 5120).

### 4. Linear-attention reorders 1.0.3 bakes in — reader must UNDO *[src]*

`reorder_linear_required=True` for this arch. All are one head-pairing
permutation on the two 16-head halves: `(q g …)→(g q …)` interleaves
`[0..15],[16..31] → [0,16,1,17,…]`. Inverse to apply on read: `(g q …)→(q g …)`.
Dims: `g=16`, `q=2`, `p=state_size=128` (inferred), `value_length=256`.

**Same in 1.0.2 AND 1.0.3 — passes through, do NOT undo:**
- full-attn `q_proj`: rows `(g p h)→(p g h)` (g=16,p=2,h=256). Our pipeline
  already expects this planar `[q|gate]` order.

**NEW in 1.0.3 (absent from the verified 1.0.2 files) — undo these:**
| tensor (linear layers only) | 1.0.3 reorder | undo on read |
|---|---|---|
| `self_attn.gate_proj` (z) | rows `(q g p)→(g q p)` q2 g16 p128 | `(g q p)→(q g p)` |
| `linear_attn.qkv_proj` | **2nd half only** (rows 4096:8192) `(q g p)→(g q p)` | undo v-half only; q+k half (0:4096) untouched |
| `linear_attn.ssm_out_proj` | **columns** `r (q g p)→r (g q p)`; metadata at group granularity p=4 | undo on in-dim |
| `linear_attn.ssm_conv1d` | 2nd half `(q g p)→(g q p)` then transpose → `[4,8192]` bf16 | undo v-half after un-transpose |
| `ssm_alpha_proj`/`ssm_beta_proj` | rows `(q g)→(g q)` then transpose → `[2048,32]` bf16 | undo after un-transpose |
| `ssm_a` (f32) | `(q g)→(g q)` | `(g q)→(q g)` |
| `ssm_dt.bias` (f32) | `(q g)→(g q)` | `(g q)→(q g)` |

**Unchanged (no reorder, identical to 1.0.2):** MoE experts (contiguous per
expert, gate/up separate), `moe_router` (`.T`→`[2048,256]`), `shared_expert_gate`,
all norms (bf16 passthrough; +1 baked by GGUF, not the converter), `token_embd`.

---

## Implementation

Keep the change isolated to the front-end reader in each language; detect format
once per file; dispatch dequant + reorder-undo by tensor name using the
already-derived per-layer schedule (linear vs full by presence of `qkv_proj` vs
`q_proj`). No pool-builder, kernel, CPU-forward, or orchestration change.

### Python — `tools/kernel-interp/q4nx.py`
1. On open, detect format: scan I8 tensors, if a non-lm_head chunk is 4736 →
   `format="q4k"` (1.0.3); 5120 → `format="q4_1"` (1.0.2). Store on the reader.
2. Add `dequant_q4k_file(packed, out, in_)` — the §1 layout → `[out,in]` fp.
3. Add `dequant_q8_q4k_file(...)` — the §3 column-major lm_head.
4. Add a `_undo_reorders(name, W, layer_type)` layer applying the §4 inverse
   permutations, keyed by tensor name + whether the layer is linear/full.
5. Route `bf16()`/`f32()`/quant reads through detect → dequant → undo, so every
   existing caller (`build_pools.py`, `full_forward.py`, `decode_step.py`,
   `run_5li3_npu.py`) gets identical `[out,in]` weights regardless of format.
   **These callers do not change.**

### Rust — `npu-engine/src/q4nx.rs`
Port the same four pieces (detect, `dequant_q4k`, `dequant_q8_q4k`,
`undo_reorders`). `forward.rs`, `decode.rs`, pool building unchanged.

### Format detection helper
A tiny `q4nx_format.py` CLI (or a flag on the existing one) that prints the
detected format + per-tensor chunk sizes, for quickly classifying any file.

---

## Validation (resolves the "no sample file" constraint)

**Primary — reference converter as ground truth, no 16.5 GB download:**
1. `tools/kernel-interp/hf_ref/` real HF tensors → `make_synthetic_gguf.py` →
   small `qwen35moe` GGUF (already exists in `tools/q4nx-convert/`).
2. Run the **cloned reference converter** on it → a genuine 1.0.3 `model.q4nx`.
3. Read it with the new 1.0.3 reader → `[out,in]` fp weights.
4. Assert vs (a) the HF originals and (b) our verified 1.0.2 reader on the same
   weights. Correct layout + correct reorder-undo ⇒ **quant bound (~0.01–0.05)**;
   any wrong reorder ⇒ **~0.3** (the same sharp diagnostic the 1.0.2 converter
   work used).
5. Full-slice forward: real activation through 1.0.3-read projections must be
   finite at **corr > 0.999** vs the 1.0.2-read weights.

**Secondary — self-consistency:** 1.0.2 file → dequant → re-pack with the
reference `_pack_q4k` logic → read back with the new reader → round-trips to the
same weights. Proves the reader inverts the packer exactly.

**Final lock (when a real 1.0.3 file exists):** run the new reader on a real FLM
1.0.3 model (Cyrus's 27B, or the base-35B), confirm every tensor decodes at the
quant bound, and resolve the two *inferred* facts (`state_size`, `ssm_a`
`−exp` vs raw). Then run it end-to-end on the open engine — which already runs
interval-3 to finite logits where FLM 1.0.2 NaN-collapses.

---

## Real-file validation (2026-08-30) — both inferred constants RESOLVED
Converted `FastFlowLM/Qwen3.5-0.8B-q4_k.gguf` (a real published GGUF, arch
`qwen35`, the same DeltaNet family as Cyrus's `qwen35moe`) through the reference
converter -> a genuine 1.0.3 file (150 Q4_K + 37 q8 tensors). Two results:
- **Q4_K byte layout confirmed vs INDEPENDENT ground truth**: our
  `dequant_q4k_file` vs `gguf.dequantize` on 126 non-reordered Q4_K matmuls, ALL
  at rel 3.2e-3–4.5e-3 (the refit floor), 100% pass (`validate_real_08b.py`).
  This is beyond the packer-based test — ggml's own decoder, not our packer.
- **state_size = 128 CONFIRMED** from the GGUF field `qwen35.ssm.state_size`.
- **ssm_a = −exp(A_log) CONFIRMED**: stored value is byte-identical to the GGUF
  tensor, all-negative, and ≠ −exp(gguf) — i.e. the GGUF already carries the
  pre-baked −exp(A_log) and the converter passes it raw. Matches the reader's
  assumption exactly (NOT raw A_log).
Caveat: the 0.8B is small (`ffn=3584 ≤ 6144` → `reorder_linear_required=False`),
so it does NOT exercise the A3B linear reorders (only the passthrough `q_proj`).
Those remain validated against the reference packer only; a real A3B (qwen35moe)
1.0.3 file — Cyrus's 27B — is the one remaining item to exercise them end-to-end.

## Risks / open items
- **A3B linear reorders** validated against the reference packer + a transcode,
  not yet against a real qwen35moe 1.0.3 file (none published; needs Cyrus's 27B).
  state_size=128 and ssm_a=−exp are now confirmed for the qwen35 family and
  near-certainly identical for qwen35moe (same DeltaNet), but not independently
  reconfirmed on an A3B file.
- **ssm_out_proj column reorder at two granularities** (element for nibbles,
  group=4 for the uint8 metadata) — the fiddliest piece; the synthetic-GGUF test
  catches it.
- Format detection is by chunk size only (no version field) — robust for the two
  known formats; if FLM later adds a third, revisit.
- The reference converter is cloned under the scratchpad for planning. Decide a
  permanent home (e.g. `tools/q4nx-convert/reference/` as a gitignored submodule,
  or vendor just the needed spec) since validation runs it.

## Scope summary
- Touches: `tools/kernel-interp/q4nx.py`, `npu-engine/src/q4nx.rs`, a small
  detector CLI, a validation script under `tools/q4nx-convert/`.
- Does NOT touch: pool builders, kernels, CPU forward, decode driver,
  orchestration, tokenizer, server.
