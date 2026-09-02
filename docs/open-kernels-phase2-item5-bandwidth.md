# Open kernels phase 2, item 5: GEMV bandwidth

Where the 27B step stood before this (2026-09-02, `run_27b_attn2.log`, box at
~25 % CPU from two idle llama-servers): 132 dispatches, 313 ms, 3.2 tok/s. The
fused MoE (`me`) was 108 ms of it at ~7 GB/s; the linear layer's two GEMV
contexts (`la`, `lc`) 96 ms; the attention layer (`al`) 30 ms.

## What was actually slow (measured, not modelled)

1. **The q4 GEMV kernel was compute-bound at 1.4 GB/s per core**, not
   DMA-bound. A `GEMV_NULL` build (same dataflow, no arithmetic) streams the
   qkv projection's 10.5 MB in 0.45 ms (~32 GB/s over 8 shim streams); the
   real kernel took 0.95 ms.
2. Two plausible fixes did nothing: 32-lane MACs with independent partial
   accumulators (0.90 ms), and vectorising the per-block scalar float work
   (0.85–0.90 ms). The disassembly showed why: vegah's nibble → bf16
   conversion (`to_float<bfloat16>(uint8 vector, shift)`) goes through the
   accumulator path (ups + magic add/sub + convert = 4 accumulator ops per 32
   lanes), four conversions are live per 64 B step, and the allocator spilled
   ~30 accumulator quarters per iteration. The MAC slot was never the limit;
   the ALU/accumulator slots and the spills were.
3. The bf16 route cannot get much better: the conversion count is inherent.

## The fix: the integer matrix unit, activations block-quantised to int16

`aie::mmul<4, 8, 8, int16, uint8>` on AIE2P takes B = one 64 B nibble block
straight from the pool bytes ([8 k][8 row pairs], N fastest — the pool's own
order, no unpack), masked to the low nibbles (even rows, value nib) or the
high nibbles (odd rows, value 16·nib; the 16 folds into d). A = the
activation octet as int16, replicated over the 4 unused rows. Per 64 B that
is 2 ANDs, 2 mmuls and one replicate, instead of ~20 accumulator/ALU ops.

The activation is block-quantised once per x per core (`gemv_q4_prep`,
entry points `gemv_q4_prep_k{512,2048,4096}`): per 32-wide K block,
`s = 14 − floor(log2(max|x|))`, `xi = round(x·2^s)`. That is **exact** for
every bf16 element within 2^7 of its block max (8 significant bits fit in
int16's 15); smaller elements round at 2^-15 of the block max. The int32
partial sums are exact; the per-block epilogue is unchanged
(`y += d·2^-s·part` as bf16 hi/lo, `+ m·xs` with the block sum precomputed
as bf16 hi/lo in the table). Table = `int16 xi[K] | int32 s[K/32] | bf16
xs_hi[K/32] | bf16 xs_lo[K/32]` (2.25 K bytes, a per-core `Buffer`).

Lane order: the mmul yields 8 even rows / 8 odd rows per product, so the
accumulator holds `[evens 0..30 | odds 1..31]`; d/m are unzipped to match
and y is zipped back to row order on the band's last k-tile only.

Entry-point signature change: `(chunks, x, y)` → `(chunks, tab, y)` plus a
prep call after each x acquire; every design that uses the GEMV was updated
(gemv_q4, moe_experts, lin_a, lin_c, attn_l). K is a runtime argument of the
shared tile/prep bodies (one COMDAT copy each): a design mixing K = 512 and
2048 overflowed the 16 KB program memory with per-K template copies.

Traps met: `aie_api`'s int32 → float `to_float` on a 16-lane vector produced
NaN on the device (compiled fine) — the block sums use the vector-only fp32
tree instead; `bit_and` on int16 vectors scalarises (32 extract/and/push) —
mask through the uint8 view; `get_cycles()` is not linkable under Peano, so
the probes are `GEMV_NULL` (DMA only) and `GEMV_TABDUMP` (table windows into
y) instead of cycle counts; the interleave_zip/unzip helpers return vectors
of the input size.

## Results (2026-09-02)

Unit tests, all PASS with the same tolerances as before:

| test | before | after |
|---|---|---|
| gemv_q4 qkv 10.5 MB (maxrel 1.2e-5 → 1.6e-5) | 0.95 ms | **0.50–0.55 ms** (null floor 0.45) |
| share_up / share_down / exp_up / exp_down / exp_gate | 0.24 | 0.16–0.27 |
| moe_experts (maxrel 2.5e-5) | 2.24–2.5 ms | **1.36–1.57 ms** |
| lin_layer la / dn / lc | 2.4–2.9 / 1.5 / 1.8 | **1.9–2.0 / 1.4 / 1.4–1.5** |
| attn_layer al | 2.5 ms | **1.33–1.5 ms** |

**27B decode step: 313 → 220 ms (4.5 tok/s), same box load, logits corr
0.999998, same argmax (846) / top-5, all 30 residuals ≥ 0.999998**
(`run_27b_item5.log`, `compare_27b.py`). Per kernel:

| kernel | n | before | after |
|---|---|---|---|
| me | 30 | 3.60 → 107.9 | **2.01 → 60.3** |
| la | 20 | 2.92 → 58.5 | **1.96 → 39.1** |
| lc | 20 | 1.86 → 37.2 | **1.49 → 29.8** |
| dn | 20 | 1.60 → 32.0 | 1.46 → 29.2 (no GEMV) |
| al | 10 | 3.01 → 30.1 | **1.97 → 19.7** |
| lm | 1 | 22.3 | 22.3 (q8, untouched) |
| rt | 30 | 0.74 → 22.2 | 0.62 → 18.6 |

## What is left in item 5, and what is not item 5

- **MoE core balance** (`me` 2.0 ms/layer): cores 4–7 idle during up/gate,
  which is 2/3 of the bytes, so those stream through 4 shim channels; plus
  the per-expert join bubble. Spreading up/gate over 8 cores (64 rows each,
  the RS=2 band law on a strided DMA tap) needs the h join to take 8 parts —
  a memtile has 6 DMA inputs, so pair columns through the east-west shared
  L1 (legal on AIE2: a core reads its west neighbour's memory) and join 4.
  Worth ~0.3–0.5 ms/layer ≈ 10–15 ms/step.
- **lm_head q8 → mmul** (`mmul<4,4,8,int16,int8>` after a 16-row split
  shuffle): the q8 kernel is at ~3 GB/s/core, the same to_float path; DMA
  floor for 540 MB is ~17 ms → −5 ms.
- **Not bandwidth: context switches.** 132 dispatches × ~0.45 ms per
  switch ≈ 60 ms of the 220. The router fold (−15 ms), design B for the
  linear layer (−18 ms) and a whole-layer single context (−54 ms) are the
  next plan decisions; they are worth more than the rest of item 5 now.
- The DMA side itself is near the ceiling: 8 streams reach ~32–34 GB/s of
  the ~40 GB/s per-agent figure; 16 streams (2 per column) are untested.

## Progress, second half (2026-09-02)

- **MoE core balance done** (`moe_experts.py`): all 8 cores do up/gate (64
  rows each) via a strided half-stripe tap `[8, 4, 2560]` (three real dims:
  the shim BD's highest dim is a repeat count, its length covers only the
  lowest three, innermost wrap < 4096 B); odd → even neighbour hand-over of
  the 64 h rows through shared L1 (`moe_cat.cc`), 4-producer memtile join
  unchanged. Driver `moeroute` keeps the intra-stripe offset (216 fills).
  Unit test PASS, **0.85 ms** (was 1.36–1.57): 17.7 MB at ~25 GB/s.
- **lm_head on the mmul path** (`lm_head_q8.h`, `gemv_tab.h` shared):
  **21.4 → 15.6 ms, 34.7 GB/s**, PASS maxrel 4.6e-6. Level with FLM's
  closed lm_head (15.4 ms).
- **27B decode step: 313 → 208 ms (4.8 tok/s)** at the same box load,
  logits corr 0.999998, same argmax / top-5 (`run_27b_item5c.log`). In the
  step `me` is 1.85 ms/layer, not 0.85: the context switch (~0.5) and a
  different 512 MB pool BO every layer (cold IOMMU/DRAM state; the unit
  kernel against a pool BO with `moeroute` runs 0.86–1.29 ms) — a
  same-context / resident-driver matter, not bandwidth.
- Left: ~60 ms of context switches (router fold −15, design B −18,
  whole-layer context −54), then dynamic KV (item 3) before the resident
  driver (item 4). DMA is at 25–35 GB/s per design against a ~40 GB/s
  per-agent ceiling; 16 streams (2 per column) untested.
