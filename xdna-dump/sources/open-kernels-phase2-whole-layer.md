# Open kernels phase 2: the whole layer in one xclbin context

Decision (2026-09-02, after item 5): of the 208 ms decode step, ~60 ms were
context switches (132 dispatches, ~0.45 ms each) and 18 ms the router
dispatch. The whole-layer context removes both. Result: **208 → 165 ms
(6.1 tok/s), 62 dispatches, 2 layer contexts, same logits / argmax / top-5**
(`run_27b_x.log`).

## What a context switch actually is

Found while building it: a multi-part sequence on one xclbin (part A leaves
the cores blocked on their next element, part B delivers it) works when the
two runs are back to back, and hangs when another xclbin's dispatch runs in
between. `lx0 → dn → lx1` hung although every DMA had landed; `lx0 → lx1`
completed in 0.5 ms. **A context switch does not preserve the array state: the
cores restart their program.** That is what the ~0.45 ms buys — a reload —
and why every design so far was one complete program iteration per dispatch.
Two consequences: a layer must be ONE context (so DeltaNet had to move onto
the main cores, which was the plan's stage 2 anyway), and within one context a
core program can be driven by several instruction streams (the driver's
`kernelx` per stream on one xclbin; two kernel objects per context are fine,
verified with rot13).

## Design (`designs/layer_x`)

Two designs, `lx` (linear attention + MoE) and `ax` (full attention + MoE).
Eight main cores (one per column, Tile(c, 2)) run one core program over three
streams each — **w** (10 KB elements from the shim: weights, S slices, the
DeltaNet records, the MoE header, experts), **x** (4 KB elements broadcast
from one shim channel: xn, og as 2 elements, xm, the expert hidden as f32),
**y** (256 B elements to the shim: band results, S' half rows, o, hidden
parts, the block output) — and do every GEMV, the DeltaNet step and the MoE
block. Helpers: norm + router (Tile(0, 3): ln_nr, ln, then the router W
streams through the same fifo), post (Tile(1, 3)), glue (Tile(2, 3)); for
`ax` the attention core (Tile(2, 3)). Shim budget: lx 13 fills / 11 drains,
ax 12 / 11.

Per layer two instruction streams on one xclbin, split where the host must
act: part 0 = everything through the router, `moeroute2` (the driver rewrites
the routed experts' fills from the router output, pool-layout placeholders
with expert j standing in for slot j), part 1 = the MoE. The xclbins of the
parts differ only in timestamps/UUIDs; part 0's serves both.

DeltaNet on the main cores (`dnx.h`): S rides the w stream in 20-row
slices, so a head is 7 slices = 140 rows — the 12 pad rows are zero in DDR
and stay zero (the k/q records are zero-padded to 160 entries) — streamed
twice (pass 1, pass 2), updated **in place** in the state BO (`[conv state |
S]`, 2.3 MB per layer), the pass-2 rows leaving through y as half rows (one
kernel call per half row; ~0.1 ms/layer of lock traffic), o as two y elements
into `act`. The record element is copied out before the slices (release()
frees the OLDEST held element).

Args: `pool, xres (InOut: the residual threads through the layers), consts
(per layer: [lnw | glue side | nw | postln | router W | sgw | out_proj]),
state (InOut), act` — five (the firmware rejects nine).

## Program memory (16 KB) — the design constraint

The main core's IRON program alone was 10 KB with one kernel call site per
stage, plus 2.6 KB of soft-float library from three scalar float ops.
What fit (15.7 KB): every GEMV shape is one runtime-parameterised entry
(`gemv_q4_pool_group_rt`: chunks per band and row split as arguments), the
kernels take ONE scratch buffer each (`ms` for the MoE, `ds` for DeltaNet,
fixed offsets), the routed and shared experts share one 9-iteration loop
(the down band law and acc/combine chosen inside the kernels from the slot
index), every loop is a `range_` loop, all main-core kernels are `-Os`, the
transcendentals in `vecmath.h` are `inline` + noinline (one COMDAT copy per
core program), and no kernel does scalar float arithmetic (vector lanes
instead: broadcast, split, extract lane 0).

Other traps met: a conv-state tap written in bf16 elements against a byte
tensor delivered half-size elements (the glue stalled); `NPU_KEEP_GOING=1`
lets the driver continue after a timed-out run so `dump` shows how far the
cores got; `dump <buf> <file> [size [offset]]`.

## Results

| | dispatches | contexts | step | per linear layer | per attention layer |
|---|---|---|---|---|---|
| item 5 | 132 | 8 | 208 ms | la 2.0 + dn 1.4 + lc 1.4 + rt 0.6 + me 1.85 = 7.3 | al 1.9 + rt 0.6 + me 1.85 = 4.4 |
| whole layer | 62 | 4 | **165 ms** (box 32 % busy vs 26 %) | lx0 4.35 + lx1 1.12 = 5.5 | ax0 2.34 + ax1 1.10 = 3.4 |

Unit test (`layer_x/make_test.py`, layer 0 of the captured decode step): all
of layer_chain's references PASS at the same numbers (xn 0.9999986, residual
0.9999995, S 1.0000000, conv state 0.9999996), the S pad rows stay zero, the
routing equals moe_chain's, the block output matches moe_chain's reference
(cos 0.9999992); lx0 3.4 ms + lx1 0.9 ms warm.

## What is left

- The remaining floor: 62 submits (~8 ms) and ~20 X↔Y context switches
  (~9 ms); lm_head 19 ms; the in-step `lx0` is 4.35 vs 3.4 in the unit test
  (a cold 512 MB pool per layer). Folding the MoE part into part 0 needs the
  0b on-device expert fetch (control packets) — the only thing between the
  router and the experts is the host patch.
- Item 3 (dynamic KV length: `pos` is still a CompileTime parameter of `ax`),
  then item 4 (the resident driver: `L40Backend` over lx/ax).
