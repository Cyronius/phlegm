# Open kernels phase 2, item 1: fusing the linear-attention layer

Where the 27B step stands after the MoE work (2026-09-02): 302 dispatches,
350–400 ms under shared load. Per linear-attention layer (20 of 30):

| dispatch | ms | context |
|---|---|---|
| ln (+residual) | 0.35 | L |
| gqkv (10.5 MB) | 1.75 | Q |
| gz (5.2 MB) | 1.33 | Z |
| dump xn / load side (host round trip) | ~0.2 | — |
| glue | 0.74 | G |
| dn (S 2 MB read twice, written once) | 1.26 | N |
| post | 0.37 | P |
| gout (10.5 MB) | 1.31 | O |
| ln (+residual, post-attn norm) | 0.35 | L |
| **chain** | **~7.7** | 8 contexts, 9 submits |

floor.cfg established that a dispatch following one in another context costs
~0.5 ms more than a same-context one, and the submit floor is 0.13 ms. So of
the 7.7 ms, roughly 4.5 ms is context switching + submits and ~3 ms is work
(26 MB of weights at ~12 GB/s ≈ 2.1 ms, S traffic ~0.3 ms, the rest small).
Fusing the chain into one context is worth ~90 ms/step; the same treatment
of the 10 full-attention layers (8 dispatches each) another ~35 ms.

## The constraint that shapes the design

Every core has 2 input + 2 output DMA channels, every column shim 2 fills + 2
drains (16 + 16 for the array), a memtile 6 + 6. Today's kernels, as
separate designs, use per layer:

| design | cores | shim fills | shim drains |
|---|---|---|---|
| gemv_q4 (×3: qkv, z, out) | 8 | 8 w + 1 x | 8 y |
| deltanet | 8 (one per column) | 8 S + 8 vec | 8 S_out + 8 o |
| dn_glue | 1 | 2 (side, act) | 1 |
| dn_post | 1 | 1 | 1 |
| ln (×2) | 1 | 1 | 1 |

The DeltaNet step alone uses the whole shim budget (16 + 16). Nothing else
can share its xclbin without restructuring it. The GEMVs' 8 cores could
carry the DN phases too (one core per column doing GEMV bands, then its 4
heads of S), but the weight stream's element (4 chunks = 20480 B) and the S
slice (16 rows = 8192 B) share no useful common multiple with a 128-row S
(10 rows per 5120 B; 128 is not a multiple of 10), so a single-stream core
means either padding the resident S to 130 rows or reworking dn_pass1/2 for
a different slice height. That is the single-context design's real cost.

## Proposal: three contexts per layer first (design A), then decide on one

**A. Regroup by shim budget, DDR-bounce between stages inside one runtime
sequence.** No kernel body changes; the xclbin holds the stage's workers
side by side and the sequence orders fills after the drains they depend on
(`drain(wait=True)` + `finish()` before the dependent fill — the command
processor executes in order). The `dump xn / load side` host round trip
disappears: the glue's xn slot is filled from the ln's xn drain.

| context | workers | fills | drains | today |
|---|---|---|---|---|
| A1 `lin_a`: ln → qkv GEMV → z GEMV → glue | 1 + 8 + 1 | 1 + 8 + 1 + 2 = 12 | 1 + 8 + 1 = 10 | L Q Z (dump/load) G |
| A2 `dn` (unchanged) | 8 | 16 | 16 | N |
| A3 `lin_c`: post → out GEMV → ln | 1 + 8 + 1 | 1 + 8 + 1 + 1 = 11 | 1 + 8 + 1 = 10 | P O L |

The two GEMVs in A1 are one worker set streaming qkv's bands then z's (x
broadcast once; y drains at two offsets, as `GEMV_YOFF` does today).
Estimated chain: 3 switches + 3 submits ≈ 2 ms overhead + ~3 ms work ≈
**5 ms/layer, −2.7 ms × 20 = −55 ms/step**, plus the host round trip.
One to two days: IRON plumbing (multi-worker programs are what moe_experts
already is), a `layer_chain`-style harness against the same layer-0 vectors,
then `make_27b.py`.

**A′. The same for the full-attention layer**: `attn_a`: ln → q/gate/k/v
GEMVs (8 cores, 4 weight regions in one stream) → attn (1 core) → o GEMV →
ln. Fills 1 + 8 + 2 (meta, kv) + 1 = 12, drains 1 + 8 + 2 (kvnew, og) + 1 =
12 — fits in **one** context: 8 dispatches → 1, ≈ −3.5 ms × 10 = −35 ms.
Attention's cached-row count is still a compile-time parameter (plan item 3).

**B. Single context for the linear layer** (later, if A's 5 ms/layer is not
enough): rework DeltaNet to 8 fills + 8 drains (vec as the first element of
each core's S stream, o as a padded last element of the S_out stream), then
A1 + dn + A3 = 12 + 8 + 11 fills — still over 16, so the GEMV and DN cores
must be the same cores with a unified stream (S padded to 130 rows, or new
slice height). Saves the remaining ~1.5 ms/layer (30 ms/step). Not before
A has been measured.

## What comes after

With A + A′ the step is ≈ 350 − 55 − 35 ≈ **260 ms (3.8 tok/s)**, of which
the fused MoE is 95, the linear chain ~100, attention ~25, lm_head 18–23.
Then item 5 (GEMV bandwidth: 12 → 25 GB/s with memtile staging, the
lm_head already there) is worth ~40 ms, and the MoE's 3.2 ms/layer (7 GB/s;
4 idle cores during up/gate, the h join round trip) another ~40. The
20–30 tok/s target needs all of it plus B — the per-token weight traffic
alone is 65 ms at 25 GB/s.

## Decision

- Go with A (+ A′), or straight to B?
- Order: A1/A3 first (linear layers, 20 of them) then A′, unless you want
  the full-attention path first because it is the one FLM gets wrong.

## Progress (2026-09-02)

- **A1 + A3 done** (`designs/lin_layer/lin_a.py`, `lin_c.py`; README
  "designs/lin_layer"). Both built first try; the shim-column pinning is
  `of.prod(tile=Tile(c, 0))` / `of.cons(tile=Tile(c, 0))` on the runtime
  handles (IRON 1.4.2), workers pinned with `Worker(tile=Tile(c, r))`.
  Stage hand-over through a DDR scratch BO (`act`) with `dma_wait` before
  each dependent fill works as assumed; weight streams and constants are
  issued ahead of the waits. The glue's xn slot is filled from `act` (host
  round trip gone), qkv/z stream from the resident pool at their offsets,
  the conv state is updated in place, and lin_c writes the MoE header.
- Unit test (`make_test.py` / `compare.py` on layer_chain's vectors): PASS
  with identical numbers to the unfused chain; **la 2.4–2.9 ms, lc 1.8 ms,
  dn 1.5–1.7 ms ≈ 6.2 ms/layer** (estimate was 5; the three context switches
  are ~1.5 of it).
- **27B step: 302 → 202 dispatches, logits corr 0.999998, same argmax /
  top-5.** 378 ms on a box ~10–15 % more loaded than the 348 ms run (shared
  kernels `me`/`lm` show the drift), i.e. ≈ −45–65 ms like for like.
- Next: **A′** (full-attention layer: ln → q/gate/k/v GEMVs → attn → o GEMV
  → ln in one context, 8 → 1 dispatches, ~−35 ms), then measure and decide
  on B.
- **A′ done** (`designs/attn_layer/attn_l.py`, README "designs/attn_layer"):
  the full-attention layer as one dispatch, built first try for pos 11 and
  pos 0. Unit test on attn_chain's vectors: PASS with attn_chain's numbers
  (residual 0.9999999), **2.5 ms warm** vs ~8 ms as 8 dispatches. Both entry
  point sets share the GEMV core; one x fifo (bf16[4096]) carries xn then og.
  `ironutil.Pipeline.finish(*eps)` added for per-channel stage waits.
- **27B step with A + A′: 132 dispatches, 8 contexts, 311–315 ms (3.2 tok/s)
  under the same load as the 378 ms run; logits corr 0.999998, same argmax /
  top-5.** Estimate for A + A′ was ≈ 260 ms on a quieter box; like for like
  (me 3.3 → 3.6 ms drift ≈ 10 %) we are at ~285. Remaining: MoE 108 ms
  (item 5), linear layer's 3 switches ~30 ms (B), router 22 ms (fold into
  the fused layer or `me`), lm_head 22 ms, dynamic KV (item 3) before the
  resident driver (item 4).
- **Item 5 (GEMV bandwidth) done first, 2026-09-02** — see
  `open-kernels-phase2-item5-bandwidth.md`: the q4 GEMV moved to the integer
  matrix unit (int16 block-quantised activations), every fused design
  rebuilt; **27B step 313 → 220 ms (4.5 tok/s), same logits.** Remaining:
  ~60 ms of context switches (router fold, design B, whole-layer context),
  MoE core balance ~10–15 ms, lm_head mmul ~5 ms.
