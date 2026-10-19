# Instrument the 3 unknown per-token cost buckets

## Goal

`bench_e2e_l40.py` currently reports one number: total wall-clock per token
(mean/min/max), compared against FLM's measured 7.05 tok/s baseline. That
number bundles three architecturally distinct costs that need separate
answers before deciding what to optimize next (see
[rust-only-open-engine.md](rust-only-open-engine.md) items 1-2 and
[rust-engine-replacement-feasibility.md](rust-engine-replacement-feasibility.md)):

1. **File/IPC overhead** — Python writes `bench_act.bin`, sends `step ...`
   over stdin, polls stdout for `STEP OK`, reads `bench_hidden.bin` back.
   [rust-only-open-engine.md:55](rust-only-open-engine.md#L55) cites a prior
   estimate of "~38ms of the previous 151ms/token" for this, but that number
   predates the current no-barrier/40-pool benchmark and was never split from
   NPU time directly.
2. **NPU submission/wait time** — the actual `runlist::execute()` +
   `runlist::wait()` calls inside the driver, chunked ≤3 layers with
   ping-pong contexts to dodge the xclbin hw_context timeout
   ([decode.rs:12-15](../../npu-engine/src/decode.rs#L12-L15)). No existing
   measurement isolates this from the file/IPC cost around it.
3. **CPU lm_head matvec** — `full_logits()` in
   [bench_e2e_l40.py:58-63](../../tools/kernel-interp/bench_e2e_l40.py#L58-L63).
   Prior estimate ~15-20ms/token
   ([rust-only-open-engine.md:29](rust-only-open-engine.md#L29)), also never
   measured in this exact (no-barrier, 40-pool) config.

Whichever bucket dominates decides where effort goes next: NPU submission
overhead is a driver/scheduling problem (hard, AMD-driver-shaped); file/IPC
is solved for free by the in-progress Rust single-process port
([rust-only-open-engine.md](rust-only-open-engine.md) item 1); lm_head is a
cheap parallelization fix ([forward.rs:387](../../npu-engine/src/forward.rs#L387)).
Guessing which one to attack first isn't justified without numbers.

## Method

Add timestamps on both sides of the existing stdin/stdout protocol, all
within one process each (steady_clock in C++, `time.time()` in Python) so
results are immune to clock-sync issues — we only need deltas within a side,
plus the two well-defined handoff points (stdin write, stdout readline).

### 1. `decode_driver_nobarrier.cpp` — 4 timestamp prints per step

Pure logging addition inside the existing `step` handler
([decode_driver_nobarrier.cpp:224-261](../../npu-engine/m0/decode_driver_nobarrier.cpp#L224-L261)),
reusing the existing `MARK` pattern's `steady_clock`/"ms since first mark"
convention (same static epoch, so these interleave cleanily with any
existing MARK lines):

- `T recv <ms>` — top of the step handler, before the file read / H2D sync.
- `T h2d <ms>` — right after `act` buffer syncs TO_DEVICE, before the
  runlist/lmhead program loop.
- `T npu <ms>` — right after the program loop finishes (last `wait()` /
  `lmhead` submit returns), before the D2H sync.
- `T d2h <ms>` — right after the D2H sync + hidden-file write, immediately
  before the existing `STEP OK` print.

No control-flow change, no change to `STEP OK`/`STEP ERR` semantics — just
four more printf lines per step. Rebuild with the existing
`build_nobarrier.cmd` (already present, already builds clean against the
checked-in `xrt_coreutil.lib`).

### 2. `bench_e2e_l40.py` — capture the marks, time the Python phases

In the per-step loop ([bench_e2e_l40.py:121-139](../../tools/kernel-interp/bench_e2e_l40.py#L121-L139)):

- Keep `t0 = time.time()`.
- Time `write_act(cur)` directly.
- Time the stdin write+flush separately from the stdout readline loop; while
  reading lines, parse `T <label> <ms>` lines into a per-step dict instead of
  discarding them, and still break on `STEP OK`.
- Time the `bench_hidden.bin` read.
- Time `full_logits(...)` (already isolable, no change needed beyond
  wrapping it).
- Time `sampler.sample(...)`.

From the four C++ marks, derive `dt_h2d = h2d-recv`, `dt_npu = npu-h2d`,
`dt_d2h = d2h-npu` per step — `dt_npu` is the actual NPU submission/wait
time, isolated from file I/O on both sides.

At the end, alongside the existing tok/s summary, print a per-phase table:
mean/min/max ms and % of total wall-clock, for: `write_act`, `ipc_h2d`,
`npu_submit_wait`, `ipc_d2h`, `read_hidden`, `lmhead`, `sample`. Drop the
same warmup steps already dropped for the tok/s number.

## Run

`python bench_e2e_l40.py 40` (a few more tokens than today's default 32, for
steadier percentiles) against the already-built L40 pools in
`npu-engine/m3out/l40` — no rebuild of pools needed, only the driver
`.exe` and the bench script change.

## Output

Append a dated "Findings" section to this file with the phase-breakdown
table and the resulting call on what to optimize next (NPU submission
overhead vs. finishing the Rust single-process port vs. parallelizing
lm_head) — same convention
[rust-engine-replacement-feasibility.md](rust-engine-replacement-feasibility.md)
already uses for dated results.

## Findings (2026-08-31)

Ran `python bench_e2e_l40.py 40` against the resident 40-pool L40 config after
rebuilding `decode_driver_nobarrier.exe` with the four `MARK` timestamps and
wiring `bench_e2e_l40.py` to parse them.

**Current total: 293.6 ms/token = 3.41 tok/s — currently ~2x *slower* than
FLM's measured 7.05 tok/s on the same 40-layer model, not faster.** That's
the real starting line; "beat FLM" needs to close a 2x gap, not a rounding
error.

Phase breakdown (mean over 37 steady-state steps, warmup dropped):

| phase | mean ms | % of total |
|---|---|---|
| `npu_submit_wait` | 192.62 | **65.6%** |
| `lmhead` (CPU) | 48.05 | 16.4% |
| `read_hidden` (file) | 24.20 | 8.2% |
| `ipc_h2d` (file+DMA sync) | 22.36 | 7.6% |
| `write_act` (file) | 3.77 | 1.3% |
| `ipc_d2h` (DMA sync+file) | 1.70 | 0.6% |
| `sample` | 0.64 | 0.2% |
| `send` (stdin) | 0.05 | 0.0% |

**NPU submission/wait dominates outright — 65.6% of every token, more than
3x the CPU lm_head cost and more than 3.5x all file/IPC overhead combined
(~52ms, 18%).** This overturns the framing from the prior conversation: the
CPU-only lm_head was a real, measurable cost (48ms, matching the ballpark of
[rust-only-open-engine.md:29](rust-only-open-engine.md#L29)'s "~15-20ms"
estimate, though notably higher here), but it was never the leading
bottleneck. Neither is the file/subprocess IPC the in-progress Rust port
removes.

**The arithmetic that matters:** even deleting the CPU lm_head cost *and*
all file/IPC overhead entirely (48 + 52 = 100ms saved) only gets total time
to ~194ms/token = 5.15 tok/s — still short of FLM's 7.05 tok/s. **Closing
the gap requires attacking `npu_submit_wait`, full stop.** That cost is the
≤3-layer ping-pong chunking workaround for the layer.xclbin hw_context
timeout ([decode_driver_nobarrier.cpp:1-21](../../npu-engine/m0/decode_driver_nobarrier.cpp#L1-L21)):
40 layers / ≤3 per chunk ≈ 14 sequential `runlist::execute()`+`wait()`
round trips per token, each incurring whatever fixed submission/queue-reset
cost the workaround exists to dodge. High variance (133.8-333.2ms across
steps, ~2.5x) is consistent with per-chunk overhead that isn't constant —
worth a follow-up with per-chunk (not just per-step) marks before concluding
whether this is real AIE compute time or driver/queue fixed cost.

**Revised priority:** the Rust single-process port and lm_head
parallelization are still worth doing (together ~34% of current per-token
time, and required regardless), but neither is the path to beating FLM by
itself. The open question that actually decides "can this beat FLM" is
whether `npu_submit_wait`'s cost is dominated by real compute (bounded,
can't be optimized away) or by the chunking/context-switch workaround
itself (fixable, if a way to avoid or amortize the ≤3-submission limit
exists — e.g. the QoS params already scaffolded in this driver's `xclbin`/
`context` directives, per its own header comment, untried so far).

## Follow-up: per-chunk marks (2026-08-31)

Added a second pair of marks bracketing each `submit` inside the step's
program loop — `MARK c<idx>_<ctx>_n<layers>_start/end` — to split the 192-219ms
`npu_submit_wait` bucket into its ~14 individual ping-ponged
`runlist::execute()`+`wait()` round trips, and re-ran
(`bench_e2e_l40.py 40`).

**Ping-pong context switching costs nothing extra** — first submission on a
context vs. every later submission on that same context: 14.69ms vs 14.85ms,
indistinguishable. This kills the "context-switch tax" hypothesis outright.

**Per-chunk cost is remarkably uniform and scales with layer count, not
submission count alone:**

| chunk | layers | mean ms |
|---|---|---|
| 0-12 (all `n=3`) | 3 | 14.4-16.0 (tight band) |
| 13 (`n=1`, remainder) | 1 | 9.18 |

Fitting `cost = fixed + n_layers × per_layer` from those two data points:
**~6.2ms fixed cost per submit/wait round trip, ~3.0ms of real compute per
layer.** That reproduces the totals cleanly: 14 round trips × 6.2ms ≈ 86ms of
pure submission overhead, 40 layers × 3.0ms ≈ 120ms of compute, sum ≈ 206ms
≈ the measured 192-219ms `npu_submit_wait` (the small residual is logging
overhead from the marks themselves — see caveat below).

**This reframes the bottleneck a third time.** The ≤3-layer chunking
workaround isn't expensive because of the context switch (that's free) — it's
expensive because it multiplies a ~6.2ms fixed per-submission cost by 14
instead of paying it once. If a single runlist could carry all 40 layers (no
timeout), the fixed-cost total would drop from ~86ms to ~6ms, saving roughly
**80ms/token**. Stacked with the lm_head parallelization (~50ms) and removing
file/IPC via the Rust port (~50ms), that's ~180ms of the current 293-324ms
attackable — landing close to **140ms/token ≈ 7.1 tok/s, right at FLM's
measured 7.05 tok/s.** The ~120ms of real per-layer compute is the likely
floor regardless of engine language or process architecture.

**Caveat — measurement overhead is real and should be subtracted mentally:**
the per-chunk marks add 28 extra `printf`+`fflush` stdout writes per token
(2 per chunk × 14 chunks) on top of the existing per-step marks. Sum of
per-chunk means (207.6ms) came in ~11.5ms below the outer `npu_submit_wait`
phase mean (219.1ms) in the same run — consistent with that added logging
cost, not a modeling error. It doesn't change the qualitative conclusion
(uniform per-layer scaling, zero context-switch cost) but means the absolute
ms figures above have a few percent of self-inflicted noise. Total per-token
time also drifted between runs (293.6ms -> 323.8ms) with no config change,
which is its own reminder that single-run NPU timings on this rig have
real variance — worth several runs before treating any of these numbers as
load-bearing for a go/no-go call.

## Resolved: the ≤3 limit is a hard, cumulative per-context layer budget (2026-08-31)

Built `test_chunk_limit.py` to probe the limit directly on real hardware,
reusing the already-loaded 40-pool L40 buffers, single context (no
ping-pong), one step. Safety: driver stdout read on a background thread so a
genuine hang can't block the script — a queue timeout force-kills instead of
blocking on `readline()`. In practice every failure came back as a clean,
near-instant `ERT_CMD_STATE_TIMEOUT` exception (already caught by the
driver's existing try/catch around `wait()`) — no hangs, no hardware wedge,
safe to run repeatedly.

Three runs pin the exact shape of the limit:

| test | result |
|---|---|
| one runlist, 3 layers, 1 submission | **OK** |
| one runlist, 3 layers, then a 2nd 3-layer submission on the *same* context | **FAILS** (2nd submission) |
| one runlist, 4 layers, 1 submission | **FAILS** (1st submission) |
| four separate 1-layer submissions, same context, no switch | submissions 1-3 **OK**, 4th **FAILS** |
| chunk_size=4 with QoS hints (`dma_bandwidth`, `frame_execution_time` maxed) | **FAILS identically** — QoS has no effect |

All four data points agree on one model: **a hw_context has a hard,
cumulative budget of exactly 3 layers of AIE work before it must be handed
to a different hw_context to reset** — consumed by total layers processed,
not by number of separate `execute()`/`wait()` calls. 3-in-one-shot: fine
(uses the whole budget in one call). 3-then-3-more: fails (6 > 3, whether
that's one call or two). 4-in-one-shot: fails immediately (4 > 3 in a single
call). 1+1+1: fine (exactly 3), +1 more: fails. QoS config knobs
(`dma_bandwidth`/`frame_execution_time`, XRT's own header marks these
undocumented — see
[xrt_hw_context.h:46-48](../../npu-engine/deps/XRT/src/runtime_src/core/include/xrt/xrt_hw_context.h#L46-L48))
don't move this boundary at all — this isn't a schedulable QoS trade-off, it's
a hard resource limit baked into this hw_context/kernel combination.

**Conclusion: the current ping-pong-every-3-layers architecture is already
optimal for this constraint.** It spends the full 3-layer budget on every
single context activation before switching — there is no way to submit
fewer than ⌈40/3⌉ = 14 chunks per token without tripping
`ERT_CMD_STATE_TIMEOUT`, and no config lever (QoS, batching, submission
grouping) changes that. **The ~86ms/token fixed submission overhead
identified above is not recoverable at this level — it is a structural cost
of working around a real AIE/driver limit, not an accidental one.** Reaching
it would require a different "layer executor" AIE program that can hold
more than 3 layers of work per hw_context activation — real kernel
engineering (mlir-aie / IRON territory), matching the
[rust-engine-replacement-feasibility.md](rust-engine-replacement-feasibility.md)
Tier 2/3 assessment (months, research-grade, not recommended as a near-term
path) — not a driver-config fix.

**Revised bottom line on "beat FLM":** the lm_head parallelization (~50ms)
and finishing the Rust single-process port (~50ms of file/IPC) remain real,
worth doing, and together could plausibly take current per-token time from
~300ms toward ~200ms (≈5 tok/s). But the ~206ms NPU floor (86ms fixed
submission overhead + ~120ms real compute) is not moving without a kernel
redesign. That likely leaves this open engine short of FLM's measured 7.05
tok/s unless FLM's closed kernel avoids this same per-context budget limit
(plausible — unconfirmed) or 5ish tok/s turns out to be an acceptable
target on its own merits.

## Research: how FLM's closed engine lives with the same 3-layer budget (2026-08-31)

Question from Cyrus: does FLM avoid the per-context layer budget, and if not,
why is it 2x faster? Sources: the closed engine's PE import table
(`dumpbin /imports qwen3_6_moe_npu.dll`), the m0c full-event capture
(RUN/SETARG/START/sync stream, 3-layer model, 6 decode tokens), and the
i3/i4 ELF-stream captures (6- and 8-layer models, 16 decode tokens each),
parsed with `tools/seq-capture/seq_struct.py`.

**FLM does NOT lift the budget — it architecturally routes around it, and
every mechanism it uses is adoptable by our driver without new kernels.**

**1. FLM runs lm_head ON THE NPU every token — and it doubles as the
context-reset barrier.** The m0c capture shows the 542,113,792-byte lm_head
pool (hash `68ee3921…` — byte-identical to our `build_lmhead_pool` output)
H2D'd at the prefill→decode boundary (event 5074), then per token: exactly
one `run::start` on a persistent run + a 1 MB logits D2H (6 tokens, 6
readbacks, buffer `…720`). FLM pays ~10ms of NPU time for the full-vocab
projection (542 MB scan at ~50-60 GB/s NPU DRAM bandwidth ≈ 9-10ms —
consistent) where we pay ~48ms of CPU matvec — and because lm_head lives in
its own xclbin/context, that submission simultaneously resets the layer
context's budget. Our original `decode_driver.cpp` replicated this pattern
as a "wasteful barrier" because we recomputed logits on CPU anyway; FLM's
version of the same submission is *productive*. We already replay the
lm_head kernel byte-exactly (M3 commit 59f8de1), so this is wiring, not
research.

**2. FLM splits layers across two persistent layer kernels — ping-pong,
same as ours.** m0c per token: layer 0 bound to kernel `…C1F60`, layers 1-2
to kernel `…C2080` (pool args confirm the 1+2 split). Even a 3-layer model
ping-pongs. FLM obeys the same ≤3 constraint; it does not batch more layers
per context activation than we do.

**3. FLM reuses runlists via `runlist::reset()` — we create/destroy one per
chunk.** The import table has `?reset@runlist@` plus **timed** waits
(`runlist::wait(duration)`, `run::wait(duration)`); it creates hw_contexts
with the plain ctor — **no QoS overload imported at all** (independently
confirms our QoS dead-end). Our driver constructs and destroys an
`xrt::runlist` per chunk (14x/token); runlist construction/destruction
plausibly carries driver-side allocation cost that `reset()` avoids — a
candidate for a chunk of our measured ~6.2ms fixed per-submission cost.

**4. FLM regenerates control code per token — cheap.** Per decode token it
builds a fresh 480-byte position ELF + a fresh **45,072-byte full-layer
decode program** (module + kernel objects too). Consecutive tokens differ
in only 20 bytes (position words); the 6-layer and 8-layer models use
byte-identical programs (it's a per-layer program — its DDR-patch arg-0
offsets march through `505282560 + k*81920`, the "main proj A" region of
the layer pool per [pools.rs:19-21](../../npu-engine/src/pools.rs#L19-L21);
5 buffer args = the same pool/act/pack/side/state signature our 8KB
executor uses). Object churn per token is affordable at 7 tok/s —
relevant freedom for the Rust port's position patching.

**Where FLM's 2x actually comes from (reconciling with our phase table):**

| cost | ours | FLM's |
|---|---|---|
| lm_head | 48ms CPU | ~10ms NPU, doubles as barrier |
| file/IPC + hidden readback | ~52ms | 0 (in-process) + 1MB logits D2H |
| submission fixed cost | 14 × ~6.2ms serialized | same constraint, but persistent runlists + timed waits ⇒ likely reused/pipelined, largely hidden |
| per-layer NPU compute | ~120ms | ~same hardware, same-shape kernels |

~120ms compute + ~10ms lm_head + small overlapped submission overhead ≈
**140ms/token = 7.1 tok/s ≈ FLM's measured 7.05.** The books balance: FLM
is not doing less work — it hides the overhead we currently pay serially.
Cyrus's instinct ("no real reason we can't match FLM") is confirmed: every
distinguishing mechanism is available to open code.

**Proposed next experiments (in leverage order, driver-only, no new
kernels):**
1. **NPU lm_head in the serve loop** — the `lmhead` directive + pool
   builder already exist; emit logits on-device, D2H 1MB, sample from that.
   Replaces 48ms CPU with ~10ms NPU that also serves as the barrier.
2. **Persistent runlists + `reset()`** between chunks instead of
   create/destroy — direct attack on the 6.2ms fixed cost.
3. **Async ping-pong pipelining** — `execute()` chunk k+1 on the other
   context before `wait()` on chunk k (both XRT calls are async-capable;
   FLM's timed-wait imports point this way). Validates whether the budget
   tolerates overlapped submissions; if yes, most remaining fixed cost
   disappears under NPU execution.

Ceiling if all three land: ~130-145ms/token (7-7.7 tok/s) with the same
kernels — FLM parity from open code, with headroom to beat it via deeper
overlap (act H2D / logits D2H under compute) that FLM may not do.

## RESULT: NPU lm_head lands — 7.38 tok/s, past FLM's 7.05 (2026-08-31)

Implemented experiment 1 (NPU lm_head in the serve loop) and re-ran
`bench_e2e_l40.py 40`. **135.4 ms/token = 7.38 tok/s, vs 293-324 ms
(3.1-3.4 tok/s) before — a 2.2x improvement from one change, and past FLM's
measured 7.05 tok/s on the same model.** Correctness verified: the NPU
logits argmax matches the CPU full-projection argmax on the same hidden.

**Two prerequisite discoveries made this a one-day change:**

1. **The lm_head kernel's output was misunderstood — it already emits the
   FULL vocab.** The repo (and `loglogits`) read the 1 MB logits buffer as
   f32[124160] "odd vocab half". It is actually **bf16[248320]** — full
   vocab, half the bytes (496,640 B, and the buffer's second half is all
   zeros). Proof: decoding as bf16 yields a clean bounded ±8.8 logit
   distribution (f32-mantissa noise would decode to wild exponents), the
   bf16 view's top tokens include *even* indices that the odd-half reading
   makes invisible, and its argmax matches the CPU reference. The old
   f32-odd reading looked plausible only because an f32's high 16 bits are
   the odd-index token's bf16 — the "odd half" was an artifact. Fixed
   understanding, not fixed kernel: `C:/caps/m0c/elf_000003.bin` +
   `lm_head.xclbin` + our byte-exact `pool_lmhead.bin` were already
   everything needed. (`m3out/l40/pool_lmhead.bin` was already built.)
2. **The lm_head kernel does the final RMSNorm on-device** — the act
   buffer's layout `[hidden bf16[2048] | model.norm.weight bf16[2048]]`
   exists precisely so lm_head can normalize and project without CPU
   involvement. No extra work was needed.

Driver change: serve `step` accepts an optional `<logits_buf> <logits_out>`
pair and dumps 496,640 bytes after the program; the serve-program `lmhead`
directive got `MARK lmh_start/end` brackets. Bench change: `serve_config()`
adds the LM xclbin/kernel/pool/logits buffers and one trailing
`lmhead klm logits lmpool act` (which also serves as the trailing
cross-context barrier — FLM's own trick); sampling now reads bf16 logits
directly; the CPU `full_logits` survives only as a step-0 sanity check.

**The second, unplanned effect: removing the CPU matvec HALVED the NPU
chunk cost.** Per-chunk mean fell 14.4-16.0 ms → **6.9-7.0 ms**, and
per-token variance collapsed (129.8-146.8 ms spread vs 244-427 before).
The 48 ms/token numpy matvec was scanning a ~2 GB f32 matrix through the
same DRAM the NPU DMAs weights from — on Strix the CPU and NPU share
memory bandwidth, so the CPU lm_head was throttling every NPU transfer.
Cutting it paid twice: once directly (48 ms → 15.4 ms on-NPU) and once by
freeing bandwidth (~95 ms of layer time → ~95 ms... chunk sum 95.4 ms vs
192-219 before). The earlier "fixed ~6.2ms / per-layer ~3.0ms" model was
measured under this contention; the clean numbers are ~3.3 ms fixed per
submission, ~1.2 ms per layer.

Current phase profile (steady-state means): npu layers+glue 114.2 ms
(84%), of which lmhead 15.4 ms; ipc_h2d 8.0; read_logits 9.1 (file+bf16
convert); write_act 1.6; sample 1.1.

**Remaining attackable, in order:**
- read_logits 9.1 + ipc_h2d 8.0 + write_act 1.6 ≈ **~17 ms of file/IPC**
  that the in-process Rust port eliminates (→ ~2 ms). Lands ~120 ms ≈ 8.3
  tok/s.
- **~46 ms of fixed submission cost** (14 × ~3.3 ms) still serialized —
  experiments 2 (persistent runlists + `reset()`) and 3 (async ping-pong
  pipelining) target this. If most of it hides under execution: ~80-90
  ms/token ≈ **11-12 tok/s**, approaching iGPU class.
- npu_lmhead 15.4 ms is near the DRAM-bandwidth floor for a 542 MB scan
  (~11 ms at ~50 GB/s) — not worth attacking.

## Experiments 2 & 3: runlist reuse (negative) and pipelining (+0.2 tok/s) (2026-08-31)

**Experiment 2 — prebuilt runlists reused per token (`serveq`): NEGATIVE.**
Built all runlists/runs/arg-bindings once at serve setup (legal since every
layer arg is a resident BO; XRT documents runlist re-execution) and
re-executed per token. Result: 156.0 ms/token (6.41 tok/s) vs 134.1 (7.46)
for per-token construction, chunks 8.1 ms vs 6.9 — the baseline reproduced
cleanly in the same session, so this is real, not noise. XRT's re-execute
path evidently revalidates/repatches the cached command chain at a higher
cost than building fresh objects. Mirrors the FLM finding that fresh
ELF/module/kernel per token is affordable: object construction is NOT
where the time is. `serveq` kept in the driver as a documented negative.

**Experiment 3 — pipelined submits (`servep`): +0.2 tok/s, and it proves
we are now device-bound.** Each chunk's runlist is `execute()`d before the
previous chunk's `wait()`, so the predecessor's completion overlaps the
next chunk's host-side construction and submission. **129.3 ms/token =
7.73 tok/s** (serve: 133-134 ms). Correctness verified the strong way: the
greedy token sequence is IDENTICAL between serve and servep — the firmware
serializes cross-context commands in submission order, so the act-buffer
dependency between consecutive chunks is preserved while host latency
hides. The 3-layer budget also tolerates overlapped submission (no
timeouts across 40 tokens x 14 chunks).

The modest delta is the informative part: total NPU phase time only fell
113.6 -> 108.2 ms, so of the ~6.9 ms/chunk only ~1-2 ms was hideable host
overhead — **the rest is real device execution. Per-token device time is
~108 ms (92 ms layers + 16 ms lm_head) and is now the floor for this
kernel set.** At ~50-60 GB/s NPU DRAM bandwidth that implies ~5 GB of
weight traffic per token — decode is memory-bandwidth-bound, the normal
LLM regime. An iGPU on the same SoC shares the same DRAM, so "iGPU-class"
means this same bound; we are effectively there at the architecture level.

**Scoreboard (40L base model, greedy, end-to-end):**

| config | ms/token | tok/s |
|---|---|---|
| CPU lm_head, serial submits (start of day) | 293-324 | 3.1-3.4 |
| NPU lm_head (`serve`) | 133-135 | 7.4-7.5 |
| + pipelined submits (`servep`) | **129.3** | **7.73** |
| FLM closed engine, same model | 141.8 | 7.05 |

**Remaining:** ~19 ms/token of Python-side I/O (act write 1.7 + H2D file
read 7.7 + logits file read+convert 8.9 + sample overhead) that the
in-process Rust port eliminates → projected ~110-112 ms ≈ **9 tok/s**.
Past that, gains require reducing per-token weight traffic (kernel-level
work: batching, quantization of the fp32 logits path, expert caching —
out of scope).

## Position handling: decode is self-contained; the 480B poke is NOT needed (2026-08-31)

Investigated the suspected correctness gap that our decode never advances the
sequence position (the layer ELFs bake it; FLM regenerates a 480B position
ELF per token — 4 seqlen u32s at byte offsets 160/184/208/232, plus a 20-byte
ELF build-id that accounts for ALL other per-token bytes).

Built the mechanism anyway (driver `poketpl <ctx,...> <template>` + step-level
`<pos>` arg: patch template, fresh elf/module/kernel/run per token — FLM's own
object-churn pattern) and A/B tested. **Result: the poke has zero effect on
decode output.** pos 19 vs pos 500 vs no poke: identical greedy token streams,
even with BOTH layer contexts poked (tile memory is per-hw_context, so a
single-context poke was ruled out as the explanation too).

**Explanation, now well-supported: the decode layer kernel tracks its KV
position ON DEVICE, inside the resident 3MB state buffer.**
- A device-side state probe (3 decode steps from zeroed states, then dump)
  shows the full-attn state accumulating (~49 KB, k-region rows filling) with
  no host position input.
- Decisive: M4's replay (`c2b12bf`) was **byte-exact against FLM's captured
  multi-token decode without ever running a 480B poke** — impossible if the
  append position were host-fed, since token 2+ would have mis-indexed.

So FLM's per-token 480B ELF is not a decode-correctness requirement for the
layer kernels (plausibly a bound/limit or a prefill-path parameter). The M4
open item "patch seqlen into the 480B ELF for arbitrary prompt length"
matters for **prefill** programs (which bake T), not decode. `poketpl` stays
in the driver for prefill-side experiments; the bench defaults POKE=0.

Also noted for honesty: the L40 bench states are the pools-only build (all
40 `state_L*.bin` are zeros) — the benchmark free-runs from empty context.
Valid for timing (its purpose) and for these A/B token-identity comparisons,
but the numbers are not a demo of prompt-conditioned generation; that wiring
(real prefill states per request) is the Rust port's item 5, unchanged.

**Rust-port handoff list (proven in the driver, ready to adopt):**
1. lm_head on NPU: `elf_000003.bin` on `lm_head.xclbin`, args
   (3=logits, 4=lmpool, 5=act); output = **bf16[248320]** full vocab in the
   first 496,640 B of the logits BO; the act layout `[hidden | norm.weight]`
   makes it norm-inclusive. Kills the CPU projection AND the DRAM-bandwidth
   contention it caused.
2. Pipelined submits: `execute()` chunk k+1 before `wait()` on chunk k across
   the ping-pong contexts; firmware serializes in submission order (token
   streams identical), so waits are for error detection only.
3. Do NOT bother with: prebuilt-runlist reuse (slower), QoS params (no
   effect), per-token position pokes (decode self-tracks).

## Out of scope here

- Not touching `decode.rs` (the Rust port) — that's tracked in
  [rust-only-open-engine.md](rust-only-open-engine.md) and proceeds in
  parallel per Cyrus's direction.
- Not fixing anything yet — this is measurement only, to make the next
  optimization pick evidence-based instead of a guess.
