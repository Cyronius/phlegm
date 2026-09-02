# Open kernels: replacing FLM's closed xclbins with our own (feasibility, 2026-09-01)

**Status:** phase 0 COMPLETE, spike PASSED — MoE expert fetch is doable without a host round-trip (2026-09-02). Analysis below predates the spike; see the closing section. Supersedes the one-paragraph "Tier 3 — not
recommended" verdict in [[rust-engine-replacement-feasibility]] with what we
now know after M0–M7. Not started.

## Bottom line
Doable, and the upside is bigger than the earlier docs assumed — but it is a
2–4 month project whose single hard problem is **on-device MoE expert
fetching** (routing-dependent DMA from a 512 MB weight pool inside one kernel
invocation), not the math. Everything else is known territory with public
prior art. The reason it is worth it: FLM's decode kernels leave most of the
memory bandwidth on the table, and our host already has a byte-exact oracle
for every kernel we would write.

## What we would have to replace
Per token (decode), FLM runs ONE fused kernel per layer (`layer.xclbin` +
`elf_000005.bin`, 5 args: 512 MB pool, 1 MB act in-place, 2 MB pack, 6 MB
side, 3 MB state) and one lm_head kernel. Inside the fused kernel, per layer:

| piece | shape (per layer, batch 1) | traffic | nature |
|---|---|---|---|
| RMSNorm + input scaling | 2048 | – | trivial |
| linear-attn projections (qkv 8192×2048, z 4096×2048, out 2048×4096) or full-attn (q\|gate 8192×2048, k/v 512×2048, o 2048×4096) | q4_1 | ~14 MB | GEMV, dequant in kernel |
| gated DeltaNet step: conv1d state (bf16 3×8192), S update 32×128×128 fp32, gate/norm | 2 MB state r/w | ~4 MB | outer product + elementwise |
| full attention (every 3rd layer): KV append + softmax over ≤1024 pos, 2 kv-heads×256 | ~1 MB | small | reduction |
| router (2048×256) + top-8 | bf16 | 1 MB | tiny GEMV + sort |
| **8 routed experts**: gate/up 512×2048, down 2048×512 each, q4_1 | 8 × 1.5 MB | ~12.5 MB | GEMV with **data-dependent base addresses** |
| shared expert (3 × 512×2048) + sigmoid gate | q4_1 | 1.5 MB | GEMV |
| residual adds | – | – | trivial |

30 layers ≈ 1 GB of weight traffic per token, plus lm_head (248320×2048 q8 =
0.5 GB). At LPDDR5X-class bandwidth the NPU can realistically pull 30–50 GB/s
(TileFuse and the GEMM-generations paper both find XDNA2 data-movement-bound,
not compute-bound) → **~30–50 ms/token floor ≈ 20–30 tok/s** for this model.
FLM today: ~120 ms of kernel time for 40 layers (~8 GB/s effective) plus the
~86 ms fixed cost of the 3-submissions-per-context cap. Our resident engine
does 7–11 tok/s on the same kernels. An open kernel that (a) is bandwidth-bound
and (b) has no 3-submission cap (that cap is a property of FLM's
hw_context/kernel combination, not of the hardware) could plausibly land at
2–3× FLM. That is a materially different value proposition than "license
cleanliness".

## Prior art (what exists in the open, Sept 2026)
- **IRON / mlir-aie** ([amd/IRON](https://github.com/amd/IRON),
  [Xilinx/mlir-aie](https://github.com/xilinx/mlir-aie)): the toolchain.
  Python-defined dataflow + Peano-compiled C++ core kernels; `--dev npu2` for
  XDNA2; examples include GEMM/GEMV/softmax/MHA and a BF16 Llama-3.2-1B.
- **TileFuse** ([arXiv 2606.11357](https://arxiv.org/html/2606.11357v1)):
  W4A16/W8A16 fused dequant GEMM/GEMV on XDNA2 via IRON, Llama3-8B /
  Gemma-2B / Qwen2.5-3B. Prefill competitive with the iGPU; **decode GEMV
  only ~200 GOPs vs CPU ~300** — i.e. even careful open work has not yet made
  XDNA2 decode fast. Their 64×8 register-block GEMV and memtile weight
  staging are the starting design for our projections.
- **atassis/xdna-engine** ([github](https://github.com/atassis/xdna-engine)):
  Rust host + hand-written IRON kernels (GEMM, GEMV, fused decode, KV cache,
  LayerNorm, MHA, conv), Apache-2.0, Linux only. opt-125m int8 decode 47
  ms/token. Closest structural cousin to phlegm; no MoE, no SSM.
- **OllamaAMDNPU** ([github](https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU)):
  llama.cpp mul_mat offload with mlir-aie's matmul example. Prefill 4× CPU,
  **decode 0.65 t/s at 0.6 % AIE utilisation** — the cautionary tale: naive
  per-op offload with host round-trips is hopeless for decode.
- **Agent-skill LLM deployment** ([arXiv 2606.07586](https://arxiv.org/pdf/2606.07586)):
  Llama-2-7B end-to-end on IRON, "up to 15 tok/s"; reports attention,
  normalization and data movement as the hard parts. No Granite.
- **vegah/LLMNpuTest** ([github](https://github.com/vegah/LLMNpuTest), cloned
  to `C:/code/LLMNpuTest`, Apache-2.0) — the repo Josh found, and the most
  useful prior art by a wide margin: **own IRON/Peano kernels against FLM's
  q4nx format, on the identical machine** (Ryzen AI 9 HX 370, XRT 2.21.0,
  driver 32.0.20102.3930 — our driver build), Windows. Granite-4.2-3B is the
  dense variant (no Mamba/MoE), so it does not touch our two hard problems,
  but it settles most of the rest:
  - **Toolchain works on this exact hardware/driver**: mlir-aie 1.4.2.dev16 +
    Peano 21.0.0 (no Chess/Vitis), custom xclbins load and run. Phase-0 risk
    (4) is retired; what remains is the install itself.
  - **q4_1 / q8 dequant-in-kernel GEMV in OUR file layout** (`w = code*d + m`,
    16-lane row interleave consumed as-is: "16 consecutive nibbles are 16
    different rows at one k … no gather"). Their lm_head (248320×1024 q8,
    270 MB): **7.2 ms, 37.7 GB/s, fp32-exact**, 8 cores, one dispatch; with
    the memtile leg (24 cores) 2.24× faster. Our lm_head is 2× the bytes.
  - **Dispatch cost model, measured**: `t = 177.9 µs + bytes / 39.3 GB/s`,
    hw_context switch 563 µs, one shim stream per core caps a design at 8
    cores, memtile split/join reaches 24 (24 beats 32). Whole-layer fusion:
    9 ops in 3 dispatches, 2.97 ms vs 4.15 ms unfused.
  - **Bandwidth ceiling corrected**: ~39–47 GB/s is per *agent*; NPU + CPU
    together pull ~75 GB/s (LPDDR5X-7500 ≈ 120 GB/s). A hybrid split of one
    matmul gave 1.37×. That raises our per-token floor estimate's ceiling.
  - **Trap catalogue** (CLAUDE.md there): device pin to `npu2` or silent NPU1
    fallback; floor rounding default; no fp32 vector multiply on AIE2P
    (returns zero silently — split into two bf16 halves); 2 in / 2 out DMA
    streams per core; 128-byte shim transfers deliver zeros; scalar float =
    1617× slow; arg-shape mismatch hangs with no diagnostic. Weeks of
    hardware pain we get to skip.
  - Status: "arithmetic done and checked; integration not" — no token loop,
    Python-only dispatch. Their designs are driven by static instruction
    streams; nothing there does data-dependent addressing, so the
    expert-fetch spike is still ours to run.
  - Note on the official Windows path: mlir-aie ships wheels for
    Ubuntu/WSL only; the documented flow is *build xclbins in WSL, run them
    from Windows XRT* (docs/buildHostWin.md). vegah has a native
    `iron_env.cmd` build. Either works for us — our `xrt.rs` shim already
    loads xclbins on Windows; WSL exists on this box.
- **No public DeltaNet or MoE-routing AIE kernel** exists that I can find.
  Both the earlier Tier-3 note and this survey agree.

## What is actually hard, in order
1. **Routed-expert fetch inside one kernel.** The core problem. Decode needs
   the top-8 expert ids (computed on-device from the router) to select which
   ~1.5 MB slabs of the 512 MB pool to stream next — a data-dependent DMA
   address, inside a static-dataflow program. Options, cheapest first:
   (a) host round-trip per layer (router output → host → patch BO offsets →
   submit experts): ~6 ms × 30 layers ≈ 180 ms/token. Dead on arrival —
   this is exactly OllamaAMDNPU's failure mode.
   (b) two-phase kernel with an on-device "BD patch": AIE2/AIE2P shim and
   memtile DMAs are programmed by buffer descriptors that the control
   sequence writes; a core can write DMA registers of its own tile, and the
   runtime-sequence dialect exposes address patching. Whether a *core* can
   retarget a *shim* DMA mid-run without the host is the spike to run first.
   FLM's kernel evidently does it (one submission per layer, pool as a
   single 512 MB arg, in-place act). If (b) works, the design closes.
   (c) speculative prefetch: stream all 256 experts' weights — 400 MB/layer,
   no.
   (d) predict experts from the previous token / the pre-router hidden
   state early in the layer (hierarchical routing tricks) — research, no.
   **Spike (1–2 weeks): IRON kernel that reads 8 indices from a small BO and
   DMAs the 8 indexed slabs out of a 512 MB BO into memtiles, then GEMVs.
   If this cannot be made to work without a host round-trip, the open-kernel
   project is a no-go for MoE decode and the answer reverts to "keep FLM's
   fused kernel".**
2. **Bandwidth-bound q4_1 GEMV with in-kernel dequant.** Known art
   (TileFuse), but nobody has published XDNA2 GEMV near the DRAM roofline.
   Our format is fixed (q4_1 block-32, bf16 d/m, 16-lane nibble interleave —
   already documented byte-exactly), so the dequant micro-kernel is
   deterministic work. lm_head alone is 0.5 GB/token: this kernel decides
   whether we beat FLM.
3. **Gated DeltaNet decode step.** Small compute, awkward shapes (32 heads ×
   128×128 fp32 state = 2 MB read+write per layer), sigmoid/softplus/L2-norm
   in the vector unit, plus the conv1d state shift. No prior art; our
   `forward.rs` + captured per-layer states are an exact spec and oracle.
4. **Toolchain on this machine.** Retired as a *risk* by LLMNpuTest (same
   driver build, custom xclbins run). Still a task: mlir-aie wheel in WSL
   (or vegah's native build), Peano, `iron.set_current_device(npu2)`; first
   design = their `rot13`/`q4nx_unpack` smoke tests, then their lm_head
   re-pointed at our 248320×2048 q8 pool.
5. **The rest** (RMSNorm, gates, residuals, KV append + softmax, router +
   top-8, prefill GEMMs) is standard IRON-example territory.

## Why this project is easier for us than for anyone else
- **Byte-exact oracle per kernel.** `C:/caps/m0c` + `pf_t11_full` hold FLM's
  real inputs and outputs for every op and every layer boundary; the host
  engine already compares hashes/corr. Each new kernel gets a pass/fail
  against the closed one on real data — the thing every prior-art project
  lacked.
- **Host is done.** Pools, packs, states, decode-as-prefill, sampler,
  tokenizer, HTTP server, resident device management all exist; a new
  kernel drops in as a different xclbin + arg list.
- **Format is fixed and documented** (q4_1, q8 lm_head, state layouts, pack
  layout). No format design work.

## Honest note on "you're good at optimizing these kernels"
I can do the analysis (roofline, tiling, DMA/buffer planning, the dequant
micro-kernel, the DeltaNet vector code) and I can write IRON/Peano code. What
I cannot shortcut is the empirical loop: AIE kernel performance is set by
data movement details that only show up on hardware, the compile-run-measure
cycle is minutes, and observability is poor. Nothing in this repo so far is an
AIE kernel I have optimized; the closed kernels were reverse-engineered, not
written. Plan for the work to be dominated by measurement, and for the first
month to produce correct-but-slow kernels.

## Effort estimate (agent wall-clock, revised after LLMNpuTest)
Human-engineer weeks in the first draft; this repo's own calibration is
M0–M7 done in ~6 days against a 3–5 month estimate. What does not compress:
hardware-serial experiments (one NPU, timeouts instead of stack traces),
compile cycles, the shared machine, and decision latency.

| phase | output | wall clock |
|---|---|---|
| 0a | toolchain: mlir-aie + Peano in WSL, smoke design runs from Windows XRT via our shim | ½–1 day |
| 0b | **expert-fetch spike**: 8 indices in a small BO select 8 slabs of a 512 MB BO inside one dispatch, GEMV on them, verified | 1–3 days; go/no-go |
| 1 | correct open decode: lm_head + q4_1 projections (port vegah's GEMV to our shapes), DeltaNet step, attention, router+experts, each verified vs `C:/caps` oracle; slow | 1–2 weeks |
| 2 | fused per-layer kernel, memtile leg, no 3-submission cap → target 2–3× FLM | 2–3 weeks |
| 3 | prefill kernels; decode-as-prefill until then | 1–2 weeks |

If 0b fails (no on-device data-dependent fetch without a host round-trip),
the MoE decode kernel stays closed and only lm_head/projections go open —
days lost, not months.

## Recommendation
Do 0a and 0b now, in that order, alongside the perf/HTTP items. LLMNpuTest
removes the toolchain risk and hands us the GEMV, the dispatch model and the
trap list; the expert-fetch spike is the only real unknown left and it is
days, not weeks. Do not start the projection/lm_head port before the spike
resolves — if MoE decode cannot go open, the project's shape changes.

## Phase 0a DONE + spike reconnaissance (2026-09-01 evening)

**0a: our own kernel runs on the NPU through phlegm.** mlir-aie 1.4.2 + Peano in
WSL; `xclbinutil` and `aiebu-asm` (missing from the wheel, no sudo in WSL)
built from XRT master in a throwaway Docker container. ROT13 (vegah's kernel)
builds in 5 s and round-trips byte-exact through BOTH loading paths — the
classic `xclbin + insts.bin` (new `kernelx`/`run` directives + shim support)
and FLM's `xrt::elf` path (IRON's `insts.elf` loads unchanged). 0.45–0.6 ms per
dispatch. Commit `open-kernels phase 0a`. Details: `tools/open-kernels/README.md`.

**How FLM fetches routed experts — read off its own control code.** Census of
`elf_000005.bin` (the fused decode layer kernel, 49 KB of aie2 txn ops:
315 blockwrite / 315 ddr-patch / 284 write / 278 maskwrite / 276 tct):
- 240 DDR-patched DMA descriptors reference 240 *distinct* pool offsets, all in
  the shared-expert/attention region (≥ 503316480) — the static weights.
- Exactly **32 descriptors reference the routed-expert region, all at pool
  offset 0**, and **zero** reference the down-expert region. 32 = 8 experts ×
  4 gate/up stripes. They sit in the shim DMAs of columns 0, 1, 6, 7, BD ids
  {4,5,6,7,12,13,14,15}, written once (blockwrite 0x1d080/0x1d0a0/…, length
  0x5000 words) and **never enqueued by the txn** — every task-queue write
  (0x1d214/0x1d21c ← 0x8000000x) names BDs 1, 2, 9, 10 only (the static
  weight ping-pong stream).
- Therefore the expert BDs are retargeted *and* triggered from inside the
  array: the only in-array path to a shim tile's DMA registers is the stream
  network's tile-control port (control packets). That is the mechanism to
  reproduce.

**Spike design (0b), two steps:**
1. Host-sourced control packets (shim DMA → shim `TileControl:0`) that rewrite
   a shim BD's address registers and push its task queue, then verify the
   right slab arrived — proves the register/packet encoding with mlir-aie's
   existing ctrlpkt infrastructure (`test/npu-xrt/add_one_ctrl_packet` shows
   the syntax: `aie.packet_flow(id) { aie.packet_source<%tile, DMA:0>
   aie.packet_dest<%tile, TileControl:0> }`).
2. Core-sourced: `aie.packet_source<%core_tile, DMA:1>` →
   `aie.packet_dest<%shim, TileControl:0>`; the core writes control-packet
   words (BD addr lo/hi for base + idx*slab, then task-queue push) into a
   local buffer and its tile DMA sends them; slab streams back into the core
   via a circuit flow; core checksums it. Repeat for 8 indices read from a
   host BO. If the router/placer accepts a core-sourced TileControl
   destination and the hardware honours it, MoE decode can go open.
Register map: AIE-ML NOC-module DMA regs (BD n at 0x1D000 + 0x20·n; MM2S
task queues at 0x1D214 / 0x1D21C; ctrl at 0x1D210 / 0x1D218 — all confirmed
by FLM's own txn stream above).


## Phase 0b RESULT (2026-09-02): expert-fetch spike PASSED — GO

The go/no-go question — can an expert's weight slab be fetched by a
runtime-computed index with no host round-trip per fetch — is answered YES on
this hardware.

**Proof** (`designs/expert_fetch/ddr_bounce_fetch.mlir`): a shim DMA
descriptor pointing at a 4MB DDR "pool" is configured but never enqueued by
the runtime sequence. Control packets — the same words a core writes, here
staged through DDR and streamed into the shim's own control port by a shim
DMA — rewrite that descriptor's DDR address to slab `idx` and push it to its
task queue. The core sums whatever slab arrives. Two indices tested:
idx 3 -> 16384, idx 7 -> 32768, both exact, ERT state 4, 0.45-0.57 ms.
Changing only `idx` (a runtime value) changes which slab is fetched. That is
routed-expert selection.

**What the bisection established, in order (all on-device):**
1. core -> core control packet (write a lock value): PASS
2. host -> core, and host -> shim, control packets (mlir-aie's
   add_one_ctrl_packet retargeted at the shim; write + readback of shim
   registers and DMA descriptors): PASS
3. core/array -> shim control port DIRECTLY: FAILS (times out). The AIE2
   target model only lets a shim switchbox reach its own TileControl port
   from South/FIFO, and in practice the array->shim-ctrl route does not
   deliver. This is the one real constraint.
4. Resolution: bounce the packet words through DDR. The array writes them
   with an ordinary S2MM; the shim's own MM2S streams them back into its
   control port (a legal shim-DMA -> shim-TileControl packet_flow). This is
   almost certainly how FLM does it too — its shim descriptors 4-7/12-15 are
   written-but-not-enqueued exactly so control packets can retask them.

**Cost note:** the DDR bounce is one extra small DMA per fetch group, not per
byte — negligible against the ~1.5 MB an expert's weights already stream.
The per-token decode still issues its layer kernels once; expert selection
rides the same control-packet channel FLM uses.

**Driver support added:** `kernelx` (classic xclbin+insts kernel), `run`
(generic submit), `ctrlpkt` (build BD-retarget packet words from a live BO's
device address, with an enqueue-only mode); xrt-shim gained `bo.address()`
and `run.get_ctrl_scratchpad_bo()`.

**Revised verdict:** every unknown that gated the project is now retired —
toolchain (0a), and data-dependent on-device fetch (0b). What remains is
engineering, not research: the q4_1 GEMV with in-kernel dequant, the DeltaNet
step, attention, and wiring per-layer kernels behind phlegm's resident
driver, each checkable against the C:/caps oracle. Phase 1 (open + correct
decode) is the next milestone.
