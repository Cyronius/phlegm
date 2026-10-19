# gemm_q4 working notes (2026-09-14 to 2026-09-15)

These notes were in phlegm's `tools/open-kernels/README.md` and never reached openflowlm-next. The code they describe is the copy in this directory (openflowlm-next #90).

- `designs/gemm_q4/` — **batch prefill**: `gemv_q4` with an M dimension, the
  kernel the hybrid plan is gated on
  (`npu-prefill-engine/.claude/plans/hybrid-npu-prefill-igpu-decode.md`). Same
  pool-order chunks and the same tile body; four tokens ride the mmul's four A
  rows (`gemm_q4_tile4`) instead of one replicated activation.
  `GEMM_CORES=32` fans one shim stream per column into that column's mem tile
  and splits it to four cores; `GEMM_BLOCKED=1` streams the weights
  k-tile-major and keeps every band's accumulator resident, which shrinks the
  per-token table from 4608 B to 576 B and raises the L1 ceiling on M from 8 to
  24. `make_test.py --m M --cores 32 --blocked` (it does the host repacks a
  backend would do at load) + `check.py M [fast]`: every token must equal
  `gemv_q4`'s `ref_qkv.bin`.

  qkv, 8192x2048, 10.5 MB of weights per pass, 32 cores, min of 75 submits:
  DMA floor 0.279 ms; **M = 8 at 0.287 (1.03x), M = 16 at 0.350 (1.25x),
  M = 24 at 0.478 (1.71x)**; the naive dataflow at M = 8 is 0.301.
  `GEMM_PER_CALL` 1/2/4 all measure the same floor, so the ~4.7 GB/s per shim
  stream is the stream's own rate, not descriptor overhead. Absolutes move ~36%
  with host load - compare ratios against a floor from the same pass.

  Variants, all verified against `ref_qkv.bin`, measured as tokens carried per
  weight-stream-time (T) on the 10.485 MB qkv pass, 32 cores:

  | build | M | x floor | T |
  |---|---|---|---|
  | naive | 8 | 1.08x | 7.4 |
  | `GEMM_BLOCKED=1` | 24 | 1.70x | 14.1 |
  | **+ `GEMM_TAB_IL=1`** | **24** | **1.31x** | **18.4** |
  | + `GEMM_INT8=1` | 32 | 1.69x | 18.9 |
  | + `GEMM_BPT=1` | 48 | 3.38x | 14.2 |

  `GEMM_TAB_IL=1` is the one to use. It interleaves the tokens' k-octets so the
  mmul's A operand is one 32-wide load instead of four 8-wide loads and a
  concat - bit-identical output, and a third of the per-token op count gone.

  `GEMM_INT8=1` uses the dense `mmul<8,8,8,int8,uint8>` (512 MACs per issue
  against 256) and a 320 B table instead of 576 B, so it reaches M = 32. It
  gains only 3% though: eight tokens x 32 rows of live float accumulator is
  twice what the register file holds, and the spilling eats the wider mmul.
  Needs `stack_size` 0x2000 - at 0x1000 the core hangs silently (ERT state 8).

  `GEMM_BPT=<bands per output tile>` keeps one band's accumulators resident
  instead of all four, which reaches M = 48, but the activation prep is redone
  per tile and that costs more than the extra M is worth. Fix that by streaming
  an already-quantised table instead of bf16 activations.

  `GEMM_G128=1` is a **timing probe, not a kernel** - it reuses one 32-block's
  d/m for four, so the arithmetic is wrong on purpose. It measures what a
  group-128 weight format (AWQ/GPTQ's default) would cost: the epilogue runs
  twice per chunk instead of eight times, int32 accumulating across 128 K.
  Result: **1.02x the floor at M = 28** against group 32's 1.44x at M = 24, i.e.
  the kernel stops being compute-bound. Every GGUF 4-bit type is group 32
  (`QK4_0` = `QK4_1` = `QK4_NL` = 32; Q4_K and IQ4_XS are eight 32-blocks in a
  256 superblock), so none of them get this.

  `GEMM_Q4K=1` is the real kernel that gets most of the way there **without a
  format change**. A chunk is 32 rows x 256 K, i.e. exactly one Q4_K superblock
  per row, so it eats Q4_K as it lies: bf16 `d`/`mcoef` per row, 6-bit `sc`/`mn`
  per 32-block, the same nibbles - **4736 B against q4_1's 5120**, so the weight
  stream shrinks 7.5% too. `sc` is an integer, so it is applied to the mmul's
  int32 output rather than in float: the partial is narrowed to int16 by a 9-bit
  SRS (13 for the odd rows, which is where the nibble's 16x goes) and
  accumulated across the whole superblock with an int16 x int16 mac. The float
  epilogue then runs once per 256 K instead of eight times; the mmul, the nibble
  masks and the A operand are untouched. `make_q4k.py` builds the chunks and an
  exact float64 reference; `make_test.py --q4k` + `check.py <M>`.

  Floor-paired against the shipping kernel in one sitting (one xclbin per
  session - two in a session costs ~4.5 ms of hardware context switch per submit;
  30 submits, 8 rounds, minimum). The session ran slow throughout, so read the
  ratios:

  | build | M | ms | x own floor | T | tok/ms |
  |---|---|---|---|---|---|
  | floor, q4_1 (10.486 MB) | - | 0.407 | - | - | - |
  | floor, **Q4_K (9.699 MB)** | - | **0.382** | - | - | - |
  | q4_1 `TAB_IL` | 16 | 0.484 | 1.19 | 13.5 | 33.1 |
  | q4_1 `TAB_IL` | 24 | 0.674 | 1.66 | 14.5 | 35.6 |
  | **`GEMM_Q4K`** | 16 | 0.420 | 1.10 | 14.6 | 38.1 |
  | **`GEMM_Q4K`** | 24 | 0.565 | 1.48 | 16.2 | 42.5 |
  | **`GEMM_Q4K`** | 28 | **0.625** | 1.64 | **17.1** | **44.8** |

  **1.34x cheaper per token** (12.29 us/token against 16.50, least squares over
  each M sweep) and **1.26x more tokens per unit time** at each family's best M.
  M reaches 28 because the 2.5 KB sub-scale scratch is paid for by the smaller
  chunk and by staying on a 4 KB stack; M = 32 overflows `.bss`.

  Accuracy **1.63e-3 maxrel, cos 0.999999566** against `ref_q4k.bin`. The whole
  residual is that the activation gets one binary point per 256 superblock
  instead of per 32-block - the mmul's C block is shared by four tokens, so the
  SRS shift can be neither per token nor per block. Two traps found the hard way:
  aie2p has **no int32 x int32 elementwise mac** (`mac_elem_64_conf` has no
  32-bit overload, and int32 x int16 only lands in acc64, which at four live
  32-lane accumulators is twice the register file), which is why the partial is
  narrowed first - and that narrowing costs almost nothing (3.66e-3 -> 3.41e-3
  simulated). What did cost was `bf16(mcoef * mn)` per block: 3.7e-3 of a 5.0e-3
  maxrel, fixed by summing `mn * blocksum` in fp32 and scaling by `mcoef` once.
  `GEMM_FAST_EPILOGUE` on top gives 1.4e-2 and is not worth it here.

  **Correction, 2026-09-15: the "M-independent term" above was a fit artifact,
  not a real removable cost.** `GEMM_PER_CALL=2/4` at low M (no L1 pressure)
  moved timing by noise only (99-105% of `pc1`), which should not happen if it
  were per-chunk acquire/call overhead. `GEMV_NULL` built at N =
  2048/4096/6144/8192 fits a near-perfect line against weight volume -
  `floor(MB) = 0.078 ms + 0.0182 ms/MB` (predicts 0.254 ms at N = 8192 against
  0.254 ms measured) - so there IS a real, small, per-`run()` dispatch cost
  (~0.078 ms, same direction as the decode_chain submit/wait floor below, not
  fixable from inside the kernel), but it is already inside the floor every
  ratio above divides by. Refit as time *above each session's own floor* vs M
  (the honest way to isolate anything extra), the intercept goes **negative**
  (-0.116 ms) and a straight line does not fit at all - `q4k_m8` sits 0.008 ms
  above its own floor, the gap opens only past M ~ 12-16. There is no separate
  ~0.26 ms constant to chase.

  **The real next lever, confirmed instead: `GEMM_PREQ=1`, streaming an
  already-quantised activation table from the host.** `prep_host.py` mirrors
  `gemm_q4_prep_il`'s Q4_K math in numpy - the bf16 sign-bit-clear exponent
  trick, the one-shift-per-256-slice `s`, `round_half_even(bf16(x) * 2^s)`
  (exact but for the final int16 SRS, which numpy's `rint` rounds the same way
  the core's `conv_even` does) - and `gemm_q4_copy_group_m{M}` just DMAs the
  result into `tab`: no `reduce_max`, no `to_fixed`, no block-sum tree, on the
  core. Correctness is unmoved (maxrel 1.632e-3 either way). Floor-paired,
  same session, untiled (`n_tiles = 1`, so this is not yet the
  redone-per-output-tile case the ~14% estimate was originally measured on):

  | M | on-core prep | `GEMM_PREQ` | tok/ms |
  |---|---|---|---|
  | 24 | 0.358 ms | **0.319 ms** (11% less) | 67.0 -> **75.2** |
  | 28 | 0.407 ms | **0.354 ms** (13% less) | 68.8 -> **79.1** |

  Real, and close to the original ~14% estimate even without tiling in play.
  It does **not** raise the M ceiling by itself - `GEMM_PREQ`'s table element is
  576 B/token against raw bf16's 512 B/token (larger, since it carries the
  shift and block-sum metadata), so M = 32 fails the same L1 wall. The actual
  M unlock is `GEMM_BPT`'s accumulator tiling; `GEMM_PREQ`'s role there is
  removing the cost of redoing prep once per output tile, which these untiled
  numbers do not exercise.

  **`GEMM_BPT` + `GEMM_PREQ` reaches M = 40 (up from 28) and it is not worth
  it.** `acc_ty` shrinks to `bpt` bands instead of all four, but `acc_depth`
  doubles to 2 whenever `n_tiles > 1`, halving the expected saving (M = 44
  fails the same L1 wall M = 32 does untiled). Correctness holds
  (maxrel 1.632e-3), but **throughput does not improve**: M = 40 tiled measures
  77.4 tok/ms against untiled M = 28's 78.4, in the same session. Holding
  M = 28 fixed and varying only `n_tiles` (1/2/4), time above floor per token
  rises monotonically: 3.39 -> 3.61 -> 4.64 us/token.

  **Audited (same day), which corrected the mechanism.** Three probes at
  M = 28, one session: tiled with the copy intact +0.028 ms over untiled;
  tiled with the copy no-op'd (`GEMM_CFLAGS=-DGEMM_COPY_NULL` - acquire,
  release and x DMA kept, results garbage) **+0.004**; tiled `GEMM_NULL`
  against untiled `GEMM_NULL` **+0.007**. So **it is the copy - 86% - and the
  4x x stream is hidden under the weight stream.** (The "under 1%" byte
  argument compared host totals and ignored that x is broadcast to all 32
  cores from one shim channel - per core it is 516 KB against 303 KB of
  weights at M = 28 tiled - the wrong argument for a conclusion that turned
  out right.) And `acc_depth` need not be 2: `GEMM_ACC_DEPTH=1` costs nothing
  (M = 40: 0.525 vs 0.522) and reaches **M = 52** exactly where the L1 model
  says (that model reproduces every build pass/fail; untiled M = 28 is 65536 B
  to the byte). M = 52 verified (maxrel 1.632e-3) and measured: **78.8 tok/ms
  against untiled M = 28's 78.2 - a wash on throughput all the way to tiling's
  limit**, the extra copies (4x the untiled count) eating exactly what the M
  buys. The earlier 48-56/1245-1452 tok/s projection was sized against the
  Q4_1 epilogue's heavier per-token compute, where a 4x copy count was a
  rounding error.

  **On bytes rather than tok/s, tiled M = 52 wins outright.** Weights stream
  once per pass either way (tiling re-streams x, not w), so the larger M
  amortises the same 9.699 MB further: **338 KB/token at 27.1 GB/s untiled
  M = 28, against 182 KB/token at 14.7 GB/s tiled M = 52** - same wall-clock,
  1.86x less weight DRAM traffic per token, and 12.4 GB/s more bandwidth left
  for a concurrent iGPU decode stream. **Prefer tiled `GEMM_ACC_DEPTH=1` at
  M = 52**; untiled M = 28 stays the simpler reference build. **The bounded fix,
  not done:** the re-stream is free and the copy is the cost, so hand the tile
  body the fifo memory as `tab` (acquire M/4 elements per k-tile; they are
  byte-identical to `tab`'s layout, so L1 is neutral). Projected ~+7% at
  M = 52 - the first configuration where the larger M would show.

  M is capped by L1: the per-token residents are the accumulators (output rows
  per core x 4 B) and the k-tile slice table (576 B int16, 320 B int8), against
  ~49 KB after the weight double buffer, stack and x fifo.
  `GEMM_CFLAGS=-DGEMM_FAST_EPILOGUE` drops the low bf16 half of the fp32 split
  (cos 0.999988564, maxrel 6.5e-3 - ggml's CPU kernel is ~7e-3); without it,
  cos 1.0 and maxrel 1.6e-5.

