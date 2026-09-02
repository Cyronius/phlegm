# Open kernels phase 2, step 1: the MoE block as one dispatch

Kickoff plan for phase 2 of `open-kernels-feasibility.md`. Proposes starting
with plan item 2 (fused MoE) instead of item 1 (fused linear layer), based on
where the time actually goes in the 27B decode step.

## Where the 1.24 s goes (run_27b.log, 2026-09-02, shared box at ~15 % CPU)

One decode step of the pruned 27B = 1622 dispatches, 1239 ms of `run` time
(host submit → wait), i.e. **0.8 tok/s**. Plus ~2.7 s one-time xclbin loading
per process, which a resident driver pays once.

| kernel | dispatches | total ms | avg ms | share |
|---|---|---|---|---|
| gexp (expert up/gate, 512×2048) | 480 | 275 | 0.57 | 22 % |
| gdown (expert down, 2048×512) | 240 | 269 | 1.12 | 22 % |
| sm (silu·mul, 512 elems) | 270 | 111 | 0.41 | 9 % |
| ax (acc += w·y, 2048 elems) | 240 | 106 | 0.44 | 9 % |
| gsu / gsd / fin (shared expert + combine) | 130 | 117 | | 9 % |
| linear-attention chain (ln gqkv gz glue dn post gout) | 201 | 283 | | 23 % |
| full-attention chain (g4kh g512h at) | 30 | 51 | | 4 % |
| rt / lm | 31 | 47 | | 4 % |

Two facts fall out:

1. **The dispatch floor is ~0.4 ms, not 0.18.** `sm` does 512 elements of
   work and costs 0.41 ms; `ax` 2048 elements, 0.44 ms. That is pure
   submit/wait latency. 1622 × 0.4 ≈ 650 ms — **half the step is the floor.**
2. **The routed experts are 61 % of the step and run at 0.6 GB/s.** One expert
   = 5 dispatches ≈ 3.1 ms for 1.8 MB of weights. The same GEMV kernel does
   25–33 GB/s in steady state on the lm_head. The MoE block is 45 dispatches
   per layer × 30 layers = 1350 of the 1622 dispatches.

The linear-attention chain (plan item 1) is 23 % of the step across 10
dispatches/layer; fusing it saves ~120 ms of floor. Fusing the MoE saves
~550 ms of floor and gets the expert bytes onto a streaming path. So: MoE
first.

Byte budget for reference: ~1.6 GB of weights per token (30 × ~36 MB + 540 MB
lm_head). At the 25 GB/s the lm_head kernel already achieves that is 65 ms →
15 tok/s; at the 40 GB/s ceiling, 25 tok/s. The plan's 20–30 tok/s target
stands; the path there is removing dispatches, then bandwidth work (item 5).

## Progress (2026-09-02)

- **Step 0 done** (`decode_chain/floor.cfg`, driver `runx` + timed `submit`):
  the floor is 0.13 ms per submit, a same-context gexp is 0.23 ms, and a
  dispatch that follows one in a *different xclbin context* costs 0.65–0.79 ms
  — the chain's 0.4–0.6 ms average is **context switching**, not submit
  latency. Runlists: 0.07–0.16 ms per run but only within one context. So
  fusion is worth more than the floor arithmetic below says.
- **Step 1 done** (`designs/moe_experts`): all 8 routed experts in one
  dispatch, PASS cos 1.0000000 / maxrel 8.4e-5 on the moe_chain vectors,
  **2.24 ms warm** vs ~25 ms. Wired into `make_27b.py` as
  `copy hdr ← xm, rout; run me wexp hdr acc` (driver `copy` directive).
  **27B decode step: 1622 → 452 dispatches, 1239 → 460 ms (0.8 → 2.2 tok/s),
  logits corr 0.999998, same argmax/top-5, all residuals ≥ 0.999997.** Better
  than the 540 ms estimate because each removed dispatch also removed a
  context switch. Remaining: linear-attention chain ~190 ms (item 1), shared
  expert + combine ~100 ms (step 1b), fused MoE 92 ms, lm_head 23 ms, rest
  ~55 ms.
- Next: step 1b (shared expert + `fin` into the same dispatch: 5 dispatches
  and ~100 ms/step → ~0), then step 2 (0b fetch), then item 1.

## Steps

### Step 0 — size the floor (driver only, hours)

Add a `runx` directive to `npu-engine/src/decode.rs` that appends a generic
`run` to the open `runlist` (today only `layer` can; `run` always submits
alone). Rewrite one layer's 16 `gexp` runs as one runlist and time it. This
tells us how much of the 0.4 ms is per-submit vs per-run and whether runlists
alone are worth using for same-context sequences. Measurement, not product:
whatever the answer, step 1 still fuses.

Trap to respect: the closed layer xclbin timed out after ~3 queued runs
(`decode.rs` header). Open designs may differ; find out.

### Step 1 — `designs/moe_experts`: all 8 routed experts in one dispatch

Host side stays as today (experts sliced by index, since `make_27b.py` already
does that), but concatenated: one BO holding the 8 selected experts' up, gate
and down stripes back to back, plus one small record with the 8 routing
weights. Kernel:

- 8 GEMV cores stream the stripes exactly as `gemv_q4` does now (RS=4 band
  law, same `.cc` entry points), looping over experts with `range_(8)`.
- up and gate for the same expert land on the same core pair so `silu(g)·u`
  can be applied on-core before the down stripes are streamed — or on a ninth
  core fed by object fifos if the program-memory trap (16 KB) bites when the
  silu/exp helpers join the GEMV body. Decide by trying the fused body first.
- Down GEMV output is accumulated on-core: `acc += w[e] · y_e` (today's
  `moe_axpy`), `acc` stays in L1 across the 8 experts, drained once.
- Buffer args: `wexp`, `xm`, `rout`, `acc` — 4, well under the 6–8 limit.

Removes 40 dispatches per layer → 1. Expected per layer: 14 MB of expert
weights at 10–25 GB/s ≈ 0.6–1.4 ms + one floor, versus 25 ms today. Step
total ≈ 1239 − 760 + 30 × ~2 ≈ **540 ms → ~1.9 tok/s.**

Oracle: `moe_chain/compare_moe.py` (block output cos 1.0000000, maxrel
2.8e-5 today) on layer 0 of the captured 3LiF step, then `compare_27b.py` on
the whole 27B step. The hand-sliced experts mean the reference needs no
change.

### Step 1b — fold in the shared expert and the combine

`gsu ×2, sm, gsd, fin` (5 dispatches, 117 ms/step) become the ninth "expert"
in the same dispatch with weight 1 plus the `sigmoid(xm·sgw)` gate, and `fin`'s
`out = xres + acc + gate·shared` becomes the drain. MoE block = router + one
dispatch. ≈ **470 ms → 2.1 tok/s.**

### Step 2 — the 0b fetch (plan item 2 proper)

Router on one core → its 8 indices → DDR-bounced control packets retarget the
stripe BDs (`designs/expert_fetch/ddr_bounce_fetch.mlir`, proven 2026-09-02).
Removes the host slicing and the router dispatch; the whole MoE block becomes
one dispatch reading the full expert pool BO directly. This is where the
resident engine's expert traffic pattern is decided, so it also needs the
pool layout `build_pools.py` emits to be the one the BDs address — check that
before writing the kernel.

### Then plan items 1, 3, 4, 5 in the existing order

After step 2 the step is ~30 × (10 linear + 1 MoE) + 10 × 7 attention + 2 ≈
400 dispatches ≈ 160 ms of floor. Item 1 (fused linear layer) takes that to
~100 dispatches; item 3 (dynamic KV) is required before item 4 (resident
driver, the 27B end to end) can serve more than a fixed position.

## What I need from you

- Agree to MoE-first, or say the linear layer stays first (the plan's item 1
  is also the design with the open core-allocation question, which is a
  reason to do it early if you want that risk retired).
- Step 0 is cheap and independent; I'd do it while designing step 1 unless
  you'd rather skip the measurement.

No spec impact (this repo has no `specs/`; the regression oracle is the
phase-1 chain).
