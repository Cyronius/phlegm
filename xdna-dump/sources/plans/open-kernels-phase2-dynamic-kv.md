# Open kernels phase 2, item 3: dynamic KV length

**Status: DONE (2026-09-03).** Follows the whole-layer context
(`open-kernels-phase2-whole-layer.md`, commit 3fd8654). Prerequisite for item 4
(the resident driver): before this, the 27B step only ran at position 0 because
`pos` was a CompileTime parameter of `designs/layer_x/ax.py`.

## Results

- **Unit test** (`layer_x/make_test_ax.py` + `compare_ax.py`, layer 2 at
  position 11, attn_chain's inputs, the captured cache re-laid as rows):
  PASS with attn_layer's numbers (knew 0.9999998, og 0.9999993, xres
  0.9999998, xm 0.9999982), routing sane, three replays (xres + kv reloaded,
  `attnpos` re-issued) bit-identical; ax0 1.7 ms warm (2.3 in the step).
  `attn.py` and `attn_layer` on the same record + kernels: still PASS.
- **27B, three greedy tokens** (`make_27b.py --whole-layer --tokens 3`,
  `run_27b_t3.log`): 186 runs, all completed; per-position NPU run time
  157 / 154 / 155 ms (was 165 in the single-step run: no measurable change,
  as expected). Argmax matches the replica at every position (846, 198,
  3710). Position 0: corr 0.999998 as before. Position 1: every layer through
  22 at 1.000000 (7 attention layers reading row 0, 15 linear layers with
  persisted states); position 2: every layer through 23 at 0.999998. So the
  KV / RoPE / mask / state-persistence path is right. **The logits corr at
  position 1 is 0.9945 (position 2: 0.9997), below the 0.9999 bar, and the
  cause is not the cache:** at layer 23 (full attention) the attention
  output nearly cancels the input (max |x| 26.5, max |delta| 28.9, max
  |residual| 2.5), so the delta's ordinary rounding (vs the replica: cos
  0.99999997, maxrel 2.3e-3 of the delta, bf16 og + q4 o_proj) is 2.7 % of
  the residual, and layers 25/26 amplify it (0.987). Verified by replaying
  the replica's layer 23 on the NPU's own inputs (`scratchpad l23.py`): NPU
  cache rows 0/1 vs the replica's k/v 0.999998; the replica's MoE on the
  NPU's residual reproduces the NPU's layer output at 0.9999998. The routing
  also differs at layers 23/25/26 (8th-slot near-ties, p gaps ~1e-4), so
  `decode_step.moe_decode(layer, x, top=...)` gained a routing override and
  `make_27b.py` feeds it the NPU's routing from the previous run; that
  changes the replica's layer-23 output by 8e-2 % and the corr not at all.
  Follow-up (not item 3): the o_proj GEMV's int16 activation blocks against
  an og outlier at such positions, if the position-1 precision matters.
- **Trap found:** DDR patches on buffer args >= 5 carry `+0x80000000` in the
  offset word (the instruction-buffer firmware translates only the first 5
  host args into the AIE address space; mlir-aie `kDDRAIEAddrOffset`). The
  driver's `attnpos` keeps that bit; dropping it read the record from the raw
  host address (wrong cos/sin, and the next run hung).
- Pipeline order matters for the `attnpos` table: the window fill's BD
  blockwrite is the op right before its patch (`bd_write` tracking in
  `attn_patch_table`), the drain and record fills are told apart by their
  placeholder offsets (KV_ROW, arg 5).

No spec impact (the open-kernel engine has no spec yet; item 4 will get one).

## What `pos` touches today

In `ax` (part 0) three things depend on the position:

1. The host sequence issues `pos` pairs of 1 KB fills (K row t, V row t) into
   the attention core's fifo, and drains the new rows into `act[AA_KVNEW]` for
   the host to append.
2. The attention core loops `for _ in range_(pos)` over those pairs.
3. The RoPE cos/sin for the position sit in the per-layer `meta` record inside
   `consts` (host-written, position 0 in the 27B step).

## Design: one stream, three patched words per token

Not per-length streams. The instruction stream is already patched at runtime:
every DDR fill/drain is a BD blockwrite (op 1, 12 words: the 8 BD registers,
word 0 = length in 32-bit words, words 2-3 = 0 for a linear transfer) followed
by a DDR-patch op (op 0x81: arg index at +8, byte offset at +10) that the
firmware resolves against the run's buffer args. `moeroute2` rewrites those
offsets for the experts; the KV length is the same mechanism plus one length
word. So:

- **Cache layout: interleaved rows.** Row t = `[K_t bf16[512] | V_t bf16[512]]`
  (2 KB), `kv` = `MAX_CTX` rows per attention layer, an `InOut` arg. The whole
  window is ONE linear fill from `kv[0]` of `nf * 2048` bytes (the fifo delivers
  it as `2 nf` 1 KB elements, exactly the K_t, V_t order the core consumes; the
  same way the 640 KB weight fills feed 10 KB elements). The new row is ONE
  2 KB drain to `kv[pos * 2048]` (the core already emits `kout` then `vout` as
  consecutive elements). `act[AA_KVNEW]` goes away.
- **Position record table `ptab`** (one BO shared by all layers, 1 KB per
  position, `MAX_CTX` rows): `[i32 pos @0 | i32 nf @4 | cos f32[32] @512 |
  sin f32[32] @640]`, `nf = max(pos, 1)`. The sequence fills it as the second
  meta element from `ptab[pos * 1024]`. Nothing is host-written per token.
- **`consts` meta shrinks to the per-layer part**: `[qn bf16[256] @0 |
  kn @512]` at `CA_META` (1 KB; the slot stays 2 KB).
- **Attention core**: `attn_meta` copies `pos`/`nf` into an `int32[4]` Buffer
  `pb`; the row loop is `range_(pb[1])` (IRON's `_for` index-casts a loaded
  i32); `attn_step` takes `pb`, counts rows in `pb[2]` and returns early for
  `t >= pos` (only ever the one dummy row at position 0: a zero-length BD is
  not an option, so position 0 reads row 0 and masks it). Math unchanged.
- **Build placeholder** `pos = 1`: fill length 512 words, drain offset 2048,
  record offset 1024. `ATTN_POS` is retired for `ax`; one `build_ax0` serves
  every position.
- **Driver directive `attnpos <kernel> <pos>`** (npu-engine `decode.rs`):
  patch table built once from the source stream, three kinds of entry:
  - 0x81 on arg 3 (`kv`) with source offset 0 = the window fill: rewrite
    word 0 of the blockwrite whose BD register is the patch register - 4 to
    `nf * 512`.
  - 0x81 on arg 3 with source offset 2048 = the new-row drain: offset
    `pos * 2048`.
  - 0x81 on arg 5 (`ptab`): offset `pos * 1024`.
  One `sync_to_device` of the instruction BO per token (all attention layers
  share the `ax0` stream, and `pos` is the same for all of them within a
  token), so it is issued once per token, like `load xres`.

Costs: `kv` 8 MB per attention layer at `MAX_CTX = 4096` (80 MB for the 10
layers), `ptab` 4 MB. The K/V read is `pos * 2 KB` per layer per token: 8 MB
at 4096, ~2 ms at the DMA rate, nothing at short lengths.

## Files

- `designs/layer_x/ax.py`: kv `InOut`, `ptab` arg (6 buffer args: within the
  ~8 limit), the one fill / one drain, `pb` Buffer, `range_(pb[1])`.
- `designs/layer_x/layout.py`: `KV_ROW = 2048`, `MAX_CTX`, `PTAB_ROW = 1024`,
  meta layout; drop `AA_KVNEW`.
- `designs/attn/attn.h` + `attn_meta.cc` + `attn_step.cc`: the new record
  offsets, `pb`, the mask. `attn.py` (the standalone design) keeps its own
  static `pos` and old meta layout for its captured test, or is updated to
  the new record: update it (one layout everywhere).
- `npu-engine/src/decode.rs`: `attnpos`.
- `designs/layer_x/make_test_ax.py` + `compare_ax.py` (new): the unit test.
- `designs/decode_chain/make_27b.py`: `--tokens N` (below), per-layer `kv`
  BOs, `ptab.bin`, the new meta, `attnpos` per token.

## Verification

1. **Unit, position 11** (no model needed): `ax0` on the captured layer-2 pool
   (`C:/caps/m0d/000123.bo`) with `attn_chain`'s inputs, the captured 11-row
   cache re-laid as interleaved rows, `ptab` from Python. Compare og, xres, xm
   from `act` and the new row from `kv[11]` against `attn_chain`'s references
   (same tolerances as `attn_layer/compare.py`). Run the layer twice in one
   cfg (the state reload path) to check the stream re-patches cleanly.
2. **Position 0**: the 27B single step (`make_27b.py`, x mode) must still give
   logits corr 0.999998 vs the replica, same argmax (the dummy-row mask).
3. **Multi-token, the real acceptance**: `make_27b.py --tokens 3` runs the
   replica greedily for 3 positions (setting `decode_step.POS` per token: its
   `rope1` reads the module global, not the `pos` argument), writes
   `xres{t}.bin` (the embedding of each token) and `ref_logits{t}.bin`, and
   emits one cfg: per token `load xres`, `attnpos ax0 t`, the 30 layers, ln,
   lm_head, `dump logits{t}`. The linear states and the caches persist in
   their BOs across tokens. `compare_27b.py --tokens 3`: corr + argmax per
   token. Pass = corr > 0.9999 and the same argmax at every position (if the
   argmax ever diverged the later tokens would differ; the reference sequence
   is fixed up front so it is still a like-for-like comparison).
4. Timing: step time at positions 0, 1, 2 from the driver's run log; expect no
   measurable change from 165 ms.

## Out of scope

Prefill (many rows per step) stays host-side / item 4+. Folding the MoE
stream, lm_head split, the resident driver itself (item 4, next).
