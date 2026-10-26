# ArtLite prefill implementation audit

2026-10-24. Source audit of ArtLite commit
`a07a7b8893cb4a835352669d237609ba017e8ab1`, in the ignored local checkout
`target/npu-dev/artlite`. Source links below are pinned upstream paths. This is
implementation evidence, not an isolated measurement of each optimization.

## Both engines batch tokens

Phlegm's validated compact Llama image logs `block prefill route: T = 256`
and a layer-major, stage-major schedule. Projection matmuls batch 256 tokens;
flash attention handles the whole prompt per layer. It is not sequential GEMV
prefill. The current C++ path still prepares/packs activations on the host and
usually reads projection results back for host processing. Some stages retain
device outputs; inspect the `y_dev` branch rather than assuming every result
returns to the host. Source: external runtime `Core::gemm_phase`, core.cpp
around 4413–4505, base OFLM 478dbea3 plus the tracked standalone PR209 patch.

## Reusable techniques proved by ArtLite source

1. **Use the fast matrix operand format explicitly.** All prefill projection
   paths dequantize weights into `bfp16ebs8`; norms supply activations in the same
   panel format. Its shared-exponent 8-bit mantissas reach the fast matrix unit.
   This is a different rounding scheme from native BF16. The MAC helper uses
   8x8x8 products, four independent accumulators, and block streams, with separate
   restricted C load/store pointers. The source reports II 33 for its split
   loop; that schedule is an upstream report, not independently remeasured here.
   [GEMM planner](https://github.com/artlite-project/artlite/blob/a07a7b8893cb4a835352669d237609ba017e8ab1/src/artlite/operators/prefill/gemm.py),
   source paths `gemm/mac.py`, `gemm/gemm_q.py` in the same tree.
2. **Make intermediate layouts the next operator's input.** Norms write packed
   BFP panels; QKV feeds attention prep and the decode-compatible KV cache;
   flash attention produces O-projection panels; gate/up fuses SiLU and writes
   down-projection panels. These buffers stay on one device. Residual adds and
   norms run on the NPU, avoiding the host packing/readback cycle in our current
   general projection route. Llama uses a BF16 residual with FP32 delta sums.
   Source: `operators/llama/prefill.py`, `_ops_runlist` (around 230–285).
3. **Amortize configuration inside one hardware context.** A library ELF holds
   distinct designs once. Host-built control programs reconfigure operators in
   the same context, run each operator across sub-chunks consecutively, and split
   only at whole-layer boundaries to respect the driver watchdog. Configuration
   still costs time: one context does not imply no reconfiguration. Sources:
   `operators/prefill/chunks.py`, `operators/fused/`.
4. **Use larger GEMM items selectively.** Llama chunks reach 4096 tokens. QKV
   remains in 256-token items; other projections use 1024-token items when the
   chunk divides evenly. Norm blocks reach 2048. The code selects sizes by
   operator; simply increasing every tile is not the technique. Source:
   `operators/llama/prefill.py`, `SUB`, `BIG`, `_t`, `_ops`.
5. **Reuse weights and activations where they live.** GemmQ pairs neighboring
   cores to split dequantization and exchange panels through shared memory.
   GemmQd dedicates one core per column to dequantization and broadcasts panels
   to up to three MAC rows. For tile-major Q4_1 and T=256, planner selection is
   GemmQ, not the dedicated-dequant variant. Pin streams and alternating C tiles
   to separate 16 KiB memory regions; retain activation strips in the memtile
   across output passes. Split long-K output projections when that makes strips
   resident. Sources: `gemm/gemm_q.py`, `gemm/gemm_qd.py::gemm_mac_rows`,
   `prefill/gemm.py::resident_k_part`.

All source paths above are relative to `src/artlite/` in the pinned checkout.
The exact runtime-selected plan still needs a captured per-projection plan;
static planner rules alone do not establish which fraction of total time each
technique saves. Do not infer a speedup from upstream comments on other models.

## Local evidence and constraints

Measured on Strix Halo MAX+395, Fedora44 kernel7.2.11, amdxdna0.10.0,
firmware1.1.2.65, default power mode. Phlegm XRT2.27/MLIR-AIE1.4.2/
Peano21.0.0.2026080301; ArtLite MLIR-AIE1.4.5.dev56+g1a647f1/
Peano22.0.0.2026102001. Same Llama3.2 1B Q4_1 GGUF input, but arithmetic and
head representations differ. Existing paired development samples at 1K/4K/8K:
Phlegm prefill771/717/586 versus ArtLite9592/9302/7779 tokens/s. These do not
isolate individual techniques or establish matched-quality parity.

Our standalone uncompensated BFP conversion ran in 1.983 ms but failed:
relative error0.009321872 versus source (limit0.005) and0.008950170 versus BF16
operands (limit0.0001). It was rejected, not promoted. See
[the owning experiment report](gguf-prefill-linux.md). ArtLite's own gates use
model KL and several prompts, and its Llama prefill uses single BFP Q/K/P;
Phlegm's Llama high/low probability contract must remain intact. Our recorded
ArtLite sequential quality checks are not a prefill quality acceptance.

## Next experiments, not accepted changes

Capture actual per-phase plans and timings first. Separate configuration-only,
matmul-only and host packing/readback contributions with paired controls. Then
investigate accurate fast-matrix arithmetic (for example residual compensation,
subject to register/stack and independent numerical gates), and device-resident
norm/activation/layout pipelines. Retain failure cases and long-context gates.
Do not copy LGPL implementation code into a differently licensed kernel without
reviewing the license obligations; these notes describe techniques only.

## Transfer experiments

The2026-10-25 continuation in [the owning report](gguf-prefill-linux.md) records
a numerically passing compensated-BFP projection with1.51x isolated throughput
after moving conversions to producers and fusing dequant into panel output.
Neighbor sharing, naive residency and unrolling were also tested and retained
as slower results. These are not full-model improvements. Canonical activation
residency is currently disabled; a memtile-repeat probe builds correctly but
did not speed up this compute-heavy shape.


# Pipeline gaps after HH-only failed to accelerate the engine

2026-10-26. Read-only source analysis; no NPU or resource jobs. ArtLite pinned
`a07a7b8893cb4a835352669d237609ba017e8ab1`, local
`target/npu-dev/artlite/src/artlite/`; runtime paths below refer to
`target/npu-dev/reference-runtime-next/src/open_qwen36/`. Runtime is dirty
experimental source, not an upstream commit attribution. Root reports clean
HH-only/compensated engine ratio 0.978436703; this audit does not remeasure it.

## Concrete residency gaps

- `core.cpp:3534`: existing device QKV prep requires old `fmt==bfp16`.
  The new native `[N,T]` F32 HILO projection cannot use the old c-tiled prep
  layout. Thus its QKV reads back for host transpose/RoPE/Q high-low packing
  and KV writes (`3583–3695`). Simply removing the format guard is incorrect.
- `core.cpp:3558–3565`: raw output and fused residual/norm shortcuts also
  require old bfp16. HILO uses separate transpose/add/norm operations.
- `core.cpp:3766–3775`: flash output is read back, then O input is host packed
  and uploaded through generic `gemm_phase`.
- `core.cpp:3813–3840`: gate/up are separate GEMMs. Each output reads back and
  transposes; host computes double-precision exp-based SiLU times up, then
  down input is packed/uploaded.
- `core.cpp:4422–4505`: `gemm_phase` already supports `y_dev` retained output,
  but always host-prepares/uploads its input. A device input BO path is absent.
  Its stage-in overlaps the preceding run; timers cannot simply be added.

At T256/H2048/F8192, gate/up readbacks are 8 MiB each, and down HILO input
upload is 4.5 MiB: a device gate/up -> SiLU+HILO pack -> down chain removes
20.5 MiB of host/device synchronization payload per block/layer. This is a
byte calculation, NOT measured latency or a speedup bound. It additionally
removes host transposes, SiLU and down packing. A separate glue kernel adds
configuration/dispatch cost, so the first useful experiment must compare the
complete chain, not just SiLU kernel time.

Minimal prototype: retain gate/up native F32 outputs in BOs using the existing
`y_dev` structure; add one SiLU+HILO-pack operator writing the current explicit
input ABI; allow down to consume that BO without `stage_in`. Keep C++ as owner.
Do not initially redesign all projections or residual storage. Device F32/tanh
SiLU is not bit-equivalent to host double exp, so stage it under an explicit
quality mode and independently test source/operand errors and whole-model
300/8192/boundary gates. An exact encoder needs host rollover/tie behavior.

## What ArtLite actually does

`operators/llama/prefill.py:230–284` constructs device-BO chains. Gate/up writes
HP, down reads HP (`274–279`); O and down outputs feed residual+norm operators
which write the next packed activation (`269–284`).
`operators/prefill/gemm.py:229–250` combines gate/up weights into 8-gate/8-up
row groups, enables SiLU and packed C. `operators/gemm/gemm_q.py:254–294` performs
the SiLU epilogue and directly emits BFP panels. These are real implementation
paths, not timing attribution. `operators/prefill/norm.py:28–39` uses BF16
residual inputs/updates and reciprocal high/low arithmetic; copying that
numerical policy would change our double-residual contract. Residency here
means persistent device BOs, not all activations/weights permanently in SRAM.

## Does T1024 dequantize weights only once?

For tile-major Q4_1, **no**: source disproves a 4x whole-operator dequant reuse
advantage over four T256 items.

- `llama/prefill.py:42–54,131–145`: QKV item T256; O/gate_up/down item T1024
  when the prompt chunk divides 1024. Total chunks reach4096.
- `prefill/gemm.py:78–91` chooses the class through `gemm_mac_rows`.
  `gemm/gemm_qd.py:606–620` returns0 (GemmQ) for tile-major Q4 O, gate_up and
  down even above256. QKV remains256 and also selects GemmQ. This statement
  is conditional on the actual image being tile-major Q4_1, as in the audited
  Q4_1 path; planar quantizations choose different operators.
- `gemm/gemm_q.py:83–89,653–674`: GemmQ uses4 core rows and64 tokens/core,
  with64 output weight rows per pass. The m64 candidates fit its buffer-budget
  formula for Q4_1; T1024 gives `mc=1024/(4*64)=4`.
- `gemm_q.py:677–684,1130–1155`: `pair_fn` repeats dequant inside every
  token-chunk x output-pass iteration. Thus T1024 repeats each weight sweep
  four times, just as four T256 items do. Larger items amortize item/config
  overhead; they do not retain the entire dequantized matrix across1024 tokens.

There IS a separate **2x dequant work-sharing** distinction for the common
256-token weight sweep. GemmQ pairs neighboring token rows: each converts half
of each 64-output-row weight tile, then exchanges its half through local/shared
FIFO (`gemm_q.py:1082–1084,1130–1155`). Each complete weight value is converted
once per pair, twice across4 token rows. Our compensated64
`design.py:97–128` has4 independent row producers, each converts all64 rows
before MAC. For a projection W with N*K scalar weights and256 tokens:

- ArtLite GemmQ: approximately2*N*K scalar weight dequantizations.
- Current compensated64:4*N*K scalar weight dequantizations.
- Over1024 tokens:8*N*K versus16*N*K, respectively; no new4x gain from BIG.

These are logical operation counts excluding padding, header work and numerical
limbs. Both broadcast packed weight input per feature column; this difference
is conversion duplication, not2x external weight-DRAM traffic. Our high/low
producer also performs compensation-specific rounding/packing absent in its
single-BFP path. Previous naive neighbor-sharing failure remains valid; this
source distinction is not evidence that repeating that design will speed up.
The compact-producer agent owns producer scheduling; no overlapping edits here.

Current compensated64 `design.py:19,102–123,157–166` fixes T256 and serializes
producer/MAC per tile, with a task drain barrier per output feature band. A
naive T1024 change still re-fills weights inside the ntile loop. Any redesigned
larger-item reuse requires explicit L1/accumulator/BD analysis and measurement.
No host385ms subtraction, 1.42x ceiling, or speedup estimate follows from this
audit. The engine result calls for stage/dispatch attribution before promotion.
