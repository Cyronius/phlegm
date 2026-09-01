# Open Qwen NPU executor — implementation plan

## Goal

An **open-source execution engine** that runs the Qwen3.6-MoE family on the
Ryzen AI XDNA2 NPU, including Cyrus's pruned **interval-3** model *as published*
(no re-pruning). It reuses AMD/FLM's closed **xclbin kernels** for the actual
NPU compute; everything above them — weight loading, per-op orchestration, the
DeltaNet/MoE/attention scheduling, the interval handling that FLM gets wrong — is
open. (Open-sourcing the kernels themselves is a later, separate effort; see
"Openness ceiling".)

The engine exists because FLM's closed host engine **overflows to NaN on
interval-3** (proven: Tier-0b/c/d in `rust-engine-replacement-feasibility.md`).
The op *schedule* and the *kernels* are correct — only FLM's closed host-side
scaling is broken — so an open host that drives the same kernels correctly is the
whole fix, and it generalizes to any prune/finetune.

## What we already have (assets)

- **Kernels** (`src/xclbins/Qwen3.6-35B-A3B-NPU2/`): `mm`, `dequant_mm` (q4nx
  GEMM), `attn`, `conv` (DeltaNet causal conv), `GateDeltaNet_prefill` (the
  linear-attention recurrence), `lm_head`, `layer`. Fine-grained, per-op.
- **Captured control sequences**: `C:\caps\{i4,i3}` — 821 / 433 real per-op
  control blobs in submission order (the elf capture). The replay material.
- **Interception harness**: `tools/seq-capture/` proxy over `xrt_coreutil.dll`
  (elf + bo hooks) — lets us capture golden inputs/outputs of any op live.
- **Golden reference**: `tools/golden-ref/reference.py` — correct logits + per-
  layer activation magnitudes for the exact slices, via transformers `qwen3_5_moe`.
- **Open FLM headers** (spec, not impl): `npu_instr_utils.hpp` (ctrlcode
  encode/interpret), `q4_npu_eXpress.hpp` / `dequant.hpp` (q4nx block format),
  `causal_lm.hpp` (the engine interface), `npu_utils`.
- Q4NX format understood enough to slice (`tools/seq-capture/slice_keep.py`).

## Architecture

```
  tokenizer + sampler (trivial; reuse or port)
        |
  Qwen3.5-MoE forward orchestration        <-- OPEN, the core of the project
   (embed -> [DeltaNet | full-attn] + MoE per layer -> norm -> lm_head)
        |
  per-op dispatch: Op -> Backend
        |                         \
   CPU/torch backend               NPU backend (XRT)
   (reference, always correct)     loads xclbin + drives ctrlcode + BOs
```

- **One forward, two backends.** Every op (dequant_mm, attn, conv, GateDeltaNet,
  lm_head, elementwise/norm/router) is a dispatch call. The CPU backend is the
  always-correct reference (validated against `golden-ref`); the NPU backend runs
  the matching xclbin. Mixed execution (some ops CPU, some NPU) is allowed and is
  how we bring the NPU up one kernel at a time.
- **Correctness oracle at every step.** transformers/golden-ref gives whole-model
  logits; per-op we capture FLM's own input/output for a kernel (bo hook) and
  require our NPU op to match it bit-for-bit (or within quant tolerance).

## Risk-ordered milestones (prove the hard thing first)

- **M0 — kernel reachability (THE concept proof).**
  - **Step 1 DONE (2026-08-29): NPU reachable from our own process.** Standalone
    C++ probe (`npu-engine/m0/`, XRT headers vendored from open repo + gendef'd
    import lib over the system `xrt_coreutil.dll`) opened device[0] =
    **"NPU Strix"**, loaded FLM's `mm.xclbin`, registered it, created an
    `xrt::hw_context`, enumerated the kernel — no FLM engine involved. Kernel ABI
    revealed: **`MLIR_AIE(opcode, instr, ninstr, bo0..bo4)`** — the classic AIE
    transaction convention (opcode + instruction-buffer BO + length + up to 5 data
    BOs). This is how *every* FLM kernel is driven.
  - **Drive recipe — extracted from FLM's OPEN `npu_utils_xrt.hpp`:** the full,
    exact way to run any kernel:
    1. instruction sequence = a `uint32_t[]` **transaction blob** → `aiebu`
       assembler (`aiebu_assembler_buffer_type_blob_instr_transaction`) → an ELF.
       (Our elf-capture already dumped these assembled ELFs — so we can skip aiebu
       and feed a captured ELF straight to `xrt::elf`.)
    2. `xrt::elf(blob)` → `xrt::module(elf)` → `xrt::ext::kernel(hw_ctx, module,
       "MLIR_AIE")`.
    3. `xrt::run r(kernel);` then **`r.set_arg(0, 3)`** (opcode=3),
       `r.set_arg(1, 0)`, `r.set_arg(2, 0)` (instr/ninstr unused — the module
       carries the code), and **data BOs at args 3,4,5,…** in order.
    4. `r.start(); r.wait();` sync BOs H2D before / D2H after.
  - **Step 2a DONE (2026-08-29): run-correlation capture built + verified.**
    Extended the proxy with `run::set_arg_at_index(bo)` (@413) and `run::start()`
    (@423) hooks + a unified `events.tsv` (one counter across elf/bo/setarg/start).
    `analyze_events.py` reconstructs discrete ops: a 1-token 3LiF run →
    **406 ops, 379 ELFs**, each with its ELF, arg3–7 buffer bindings, sizes, and
    H2D/D2H roles. Verified the ABI live: every `MLIR_AIE` run binds **5 data BOs
    at args 3,4,5,6,7**.
  - **What an FLM op actually is (learned):** a fused op over **large resident
    weight pools** — arg3 is typically a 512 MB (or 542 MB) weight buffer, plus
    small activation buffers (1–6 MB) and a control ELF (384 B … 586 KB depending
    on op). Not a standalone small GEMM. Buffers are heavily reused across ops
    (same bo pointers).
  - **Step 2b PARTIAL (2026-08-29): replay path EXECUTES on the NPU, byte-match
    pending.** Built `npu-engine/m0/m0_replay.cpp` — loads a captured ELF →
    `xrt::module` → `xrt::ext::kernel("MLIR_AIE")`, allocs `xrt::ext::bo` per arg
    (FLM's pattern), binds opcode=3/instr=0/ninstr=0 + data BOs, `start()`/
    `wait()`, compares output BO to the captured bytes. It runs: `attn` produced
    262 KB of real output, `GateDeltaNet_prefill` reached state=4 (completed) — so
    the full drive path works from our own process. But no op **byte-matches** FLM
    yet. Two causes identified:
    * **ELF/buffer linkage is heuristic, not deterministic.** 379 ELFs < 406 ops
      ⇒ ELFs (and buffers) are reused across ops, so "ELF just before the op" /
      "nearest sync" pick the wrong control code or stale bytes for some ops.
    * **Likely BO-address patching.** FLM's `buffer.hpp` has commented-out logic
      re-allocating BOs to avoid device addresses `0x6000_0000–0x7FFF_FFFF` — the
      control code may not be fully address-independent, so a replay BO at the
      wrong address computes wrong regardless of ELF.
    (First false-positive lesson: op#269's output was 100% zeros — `fnv1a(zeros)`
    matched a zeroed un-run BO. Always test on ops with **non-zero** output.)
  - **Step 2c DONE (2026-08-29): M0 COMPLETE — verified numeric match on the NPU.**
    Added object-graph hooks (module-ctor @68, ext::kernel-ctor @62, run-ctor @74)
    + a run::start dump of every bound buffer (RUNARG, via `bo::size` @421 +
    `g_bo_hostptr`). `analyze_op.py` links each run → its EXACT elf
    (run→kernel→module→elf, pointer identity verified) and exact inputs. Replayed
    **op#311** (elf idx13) on `mm.xclbin`: `run state=4 (completed)`, output arg3
    = `670930a564906f22` == FLM's captured output, **byte-exact**, 523874 non-zero
    bytes. Every other xclbin timed out ⇒ `mm` is definitively this op's kernel.
    So: **our own open process drives FLM's closed kernel on the NPU and
    reproduces its exact result.** The concept is proven end-to-end.
  - **Key lessons:** (1) XRT auto-patches arg-BO device addresses into the control
    code — no manual address handling needed (a wrong-xclbin run still wrote to
    our buffer). (2) buffers are often **in-place** (arg pre-loaded with input,
    kernel overwrites it) — classify by comparing pre-exec (RUNARG) vs post-exec
    (D2H) hashes, not by D2H-presence alone. (3) only the matching xclbin
    *completes*; wrong ones time out (state 8) — a free kernel-identity oracle.

### M1 — per-kernel correctness

**Step 1 DONE (2026-08-30): all 4 activation kernels drive + reproduce FLM's
exact output.** `map_kernels.py` maps each control-ELF → its xclbin by replay
(only the matching kernel completes, state 4). Byte-verified with real non-zero
data: `mm` (op#311, 523874 B), `GateDeltaNet_prefill` (op#296, 860992 B), `conv`
(op#282, 225731 B), `attn` (op#3552, 22477 B). The whole activation path —
matmul, DeltaNet conv + gated recurrence, attention — runs correctly from our own
process. (Output-arg correlation caveat: "expected output = first D2H of that bo
after start" occasionally grabs a later op's reuse of the same buffer address —
so a real computed output matches while a stale sibling arg shows a spurious
mismatch. Tighten by bounding the D2H window to before the bo's next bind.)

**Step 2 DONE (2026-08-30): weight matmuls verified — whole decode path covered.**
Re-captured with a 600 MB cap + content-addressed dedup dumps (`blob_<size>_<hash>.bin`,
so the shared 512 MB weight pool is written once). Findings:
- **`mm` handles weight matmuls too** — op#255 (512 MB weight, arg4 activation →
  arg3 output) byte-matched (`31c16cad…`, 177383 B). So `mm` is the general
  matmul for both activation×activation and activation×dequantized-weight.
- **`dequant_mm` verified** — op#3329 byte-matched (`f4c73bad…`, 32706 B); used
  for the on-the-fly q4-weight matmuls (the later ops), vs `mm` for
  already-dequantized ones. In-place output (non-zero pre-content, overwritten).
- **`lm_head`** never completed any tested op and no distinct >600 MB vocab weight
  exists in the capture — so in this pruned 3-layer single-token decode the final
  projection folds into `mm`/`dequant_mm` with a weight in a 512 MB pool. Not a
  blocker; revisit on a full-model capture if a distinct lm_head op appears.

**RESULT: 5 kernels byte-verified from open code — `mm` (activation + weight),
`dequant_mm`, `conv`, `GateDeltaNet_prefill`, `attn` — covering the entire
per-token NPU compute. We can reproduce FLM's whole decode computation ourselves.**

**Step 3 (the deep half): interpret each op's math vs torch.** For each verified
kernel, decode its buffers (dtype/shape/layout/tiling — bf16 activations, q4nx
packed weights) and match a torch reference (using `tools/golden-ref` as the
whole-model oracle) so we know exactly what each op computes. That understanding
is what M2 ports into the Rust forward orchestration.

**Step 3 STATUS (2026-08-30): linear-attn layer + MoE block FULLY interpreted
and verified end-to-end.** Tooling in `tools/kernel-interp/` (q4nx.py dequant,
hf_fetch.py ranged HF reference downloads, moe_forward.py, op_table.py).
Capture m0d = 11-token prompt prefill (token ids recovered via embedding cosine
match = exactly 1.0; saved `prompt_token_ids.npy`).

*Q4NX weight format (byte-exact vs HF reference):* 5120B chunk = 256 bf16
scales + 256 bf16 mins (q4_1, block=32 along in-dim, planar) + 4096B nibbles in
16-lane interleave `nib[(r//16)*4096 + bc*512 + i*16 + r%16] = elem(r, bc*32+i)`.
Standard matmul tiling: chunk c → rows `64*(c//per_band)+32*(c%2)`,
cols `1024*((c//8)%(in//1024))+256*((c//2)%4)`, `per_band=in//128` (verified
in=2048 and 4096). lm_head is q8_0-style: 8704B chunk = 8192 int8 + scales.
MoE expert tiling differs: gate_up = 8 alternating 163840B bands
[up_k|gate_k]×4 per expert, band = 32 chunks over 128 rows × 2048 cols
(`rows=32*(c%4), cols=256*(c//4)`); down = 128 contiguous chunks / expert
(`rows=128*(c//8)+32*(c%4), cols=256*((c//4)%2)`). Dequant maxerr vs HF bf16 =
quantization bound (~0.004-0.03) on every tensor tested.

*Weight pools:* one 512MB pool PER LINEAR LAYER (836f…=L0, 64f0…=L1), q4nx
chunks verbatim: [gate_up experts interleaved 0..335MB][down experts
335MB..][share_up 503316480][share_gate 503971840][share_down 504627200]
[linear qkv 505282560][z-gate 515768320] (z-gate = q4nx "self_attn.gate_proj"
on linear layers = HF in_proj_z). 6MB side pool per linear layer:
[conv1d bf16 @0][ssm_norm @65536][alpha_proj bf16 @66048][beta_proj @197120]
[out_proj q4 @328192]. Small norms/router/embed live CPU-side only.

*Per-op math of the linear-attn layer (all numerically verified vs captured
buffers; activations = dense row-major bf16 [T,dim] in padded pools):*
1. qkv op (elf ~50KB, mm-class): `qkv = x @ Wqkv^T` [T,8192=q2048|k2048|v4096],
   written into conv buffer at **row offset 3** (= causal-conv left pad, K=4).
2. z op: `z = silu(x @ Wz^T)` [T,4096] — SiLU fused in kernel.
3. conv op: depthwise conv k=4 over qkv + SiLU + **per-head L2-norm fused on
   q,k sections** (v left un-normed).
4. CPU: `decay = exp(-exp(A_log)·softplus(x@Wa^T + dt_bias))`,
   `beta = sigmoid(x@Wb^T)` → fp32 [512,32] each; decay @0, beta @65536B of a
   1MB buffer.
5. GateDeltaNet op: gated delta rule per v-head (32 heads, k-head = h//2):
   `S *= decay; delta = beta·(v − S^T k); S += k⊗delta; o = (S^T q)/√128`.
   Output raw (pre-norm, ungated) [T,4096].
6. CPU: `o' = RMSNorm128(o)·ssm_norm_w · silu(z)` (gating+norm on CPU).
7. out op (mm): `attn_out = o' @ Wout^T` (no fused residual).
8. CPU: residual add.

*MoE block (verified):* CPU computes router softmax→top-8→renorm (HF
`mlp.gate.weight`, bf16, CPU-side), gathers per-expert token batches. Per
expert pair: 5200B-elf op = fused gate_up matmul → h [b,1024] in
**[u₀₋₁₂₇|g₀₋₁₂₇|u₁₂₈₋₂₅₅|g₁₂₈₋₂₅₅…] band-interleaved order**; CPU applies
silu(g)·u → H2D [b,512]; 8192B-elf op = down matmul → [b,2048] dense
(unscaled); CPU scatter-accumulates with routing weights. Shared expert runs
as its own NPU op group (fused output verified = shared(x), corr 0.996+);
final `moe_out = Σ rw_e·expert_e + sigmoid(x@sgate_w)·shared`.
**End-to-end layer-0 CPU replication (pool-dequantized weights) matches FLM's
captured layer-1 input at corr 0.990-0.997 per token** — routing decisions,
both residuals, all op semantics confirmed. NPU-vs-torch residual noise is
bf16-accumulation-order only (~1% median per op).

*Capture-tooling caveat discovered:* ~21 runs in m0d have no captured START
(reused run objects skip the hook) — e.g. the shared-expert gate_up + one more
op hide between op311 and op341. Op reconstruction must treat SETARG groups
without START as real runs. Three tiled scratch buffers around the shared
expert (elf14-out f9455c77, elf15-out f0842ef7, op341-a4 0c896405) carry
[11×2048]-sized data in an uncracked element permutation — not needed for M2
(our engine owns its buffer choreography; kernel I/O contracts above are what
matter).

*Full-attn layer (m0c, layer "2" = orig layer 3) — VERIFIED (2026-08-30):*
- 4 NPU projection ops (all plain matmuls, no fused activation): q [T,4096],
  gate [T,4096] (q_proj stores planar [q 4096 | gate 4096] in the FILE — HF
  interleaves per head [q256|g256]×16), k [T,512], v [T,512] (2 KV heads).
- CPU then does: q/k head-RMSNorm (norm weights stored EFFECTIVE = 1+HF's
  zero-centered weight!), partial RoPE (rotary dim 64 of 256, half-split
  pairs (i, i+32), theta 1e7, positions 0..T-1), packs KV (k' @0, v @byte
  1073152 of the 3MB cache buffer, both bf16 [T,512]).
- **Prefill attention itself + sigmoid(gate) + o_proj + residual + post-norm
  all run on CPU** — the NPU attn/o_proj ops execute on dummy zero buffers in
  prefill (kernel presumably decode-only). CPU uploads the finished normed MoE
  input (verified: softmax(q'k'^T/16 + causal)v · sigmoid(g) @ Wo^T chain
  matches at corr 0.9998).
- The earlier "attn op#3552 byte-verified 22477B" was actually matching the
  KV INPUT roundtrip (a5 unchanged) — the attn kernel's real math is still
  UNVERIFIED and decode-path-only. Revisit in decode interpretation.

*lm_head + final norm — VERIFIED:* q8 chunks (8704B = 512B bf16 scales j=bc*32+r
+ 8192 int8 in the same 16-lane byte interleave as q4 nibbles), FILE raster
chunk order, dequant maxerr = quant bound vs HF over 1024 rows. At the
prefill→decode boundary FLM uploads the whole 517MB lm_head pool + reads back /
re-uploads the 3 layer state buffers (GDN states + KV pack). First-decode
logits: fp32, split across (at least) two 1MB buffers **interleaved by vocab
row parity** (captured buffer b -> vocab row 2b+1; lstsq fit residual 0.4%).
lm_head input = rms(x)/sqrt-normed hidden UPLOADED unweighted; model.norm
weight folded in afterwards (recovered-hidden corr 0.99997 with
buf*norm_w).

*FILE vs POOL layouts (critical discovery):* the q4nx FILE stores every tensor
in PLAIN RASTER chunk order (chunk f -> rows 32*(f//(in/256)), cols
256*(f%(in/256)); experts contiguous per expert; lm_head same with in=2048).
All exotic tilings observed in the pools (per-band kgroup swizzle for standard
matmuls; [up|gate] 163840B band interleave + intra-band c%4 rowtile order for
experts) are applied by FLM's LOADER at model-load time. Two-probe "verbatim"
byte checks are fooled by this permutation (it fixes group-leading chunks) —
compare full regions. For the Rust engine: file reads are trivial raster;
pool-building must replicate the load-time permutations (laws documented
above + in tools/kernel-interp/moe_forward.py / full_forward.py).

*Whole-model CPU forward (tools/kernel-interp/full_forward.py):* runs the full
3LiF model (embed -> L,L,F -> norm -> lm_head) purely from model_3LiF.q4nx.
Matches captured NPU activations at corr 0.97 (L2 input) degrading to ~0.4-0.5
at logits — pure accumulation drift (bf16 NPU rounding + top-8 routing flips on
near-ties), each individual op verified at 0.99+. This is the M2 reference
implementation skeleton.

*DECODE PATH interpreted (2026-08-30, m0c blocks after ev5076):*
- Architecture: decode is FULLY FUSED. Per linear/attn layer ONE kernel run
  with 5 data args: a3 = the layer's 512MB weight pool, a4 = shared 1MB
  activation buffer (in-place hidden), a5 = 2MB param pack (uploaded at init),
  a6 = the 6MB side pool, a7 = 3MB state. Plus a separate lm_head kernel
  (args: logits-out 1MB, 517MB lm_head pool, activation). Kernel ctrlcode:
  53840B ELF shared by the layer kernels (⇒ almost certainly `layer.xclbin`),
  586KB ELF for lm_head (⇒ `lm_head.xclbin`) — replay-confirm which xclbins
  complete these ELFs next time on the NPU box. The per-token 480B ELF is a
  TEMPLATE with the sequence length patched as u32 at byte offsets
  160/184/208/232 (values 12,13,… as cache grows) — decode ctrlcode generation
  for our engine is "patch 4 ints".
- Per-token host loop: CPU samples token, uploads a4 = [row0: embed(token) |
  row1: model.norm.weight] (byte-verified), kernels run all 3 layers + final
  rms-norm(·w)+lm_head on device, D2H one 1MB fp32 logits buffer (odd vocab
  rows, v=2b+1; the even half never appears in a sync event — likely read via
  a persistently-mapped pointer, minor open item). First decode block is
  special: layers are skipped and the CPU uploads prefill's final rms-normed
  hidden directly (its logits = lm_head(upload·w), corr 1.0000 verified).
  NOTE the first generated tokens (<think>-family/newlines) look
  template-FORCED, not sampled from the logits.
- 3MB linear-layer state buffer layout (byte-verified): [0:49152] conv state
  bf16 [3,8192] = last 3 tokens' post-qkv (pre-conv) — matched prefill qkv
  rows exactly; [49152:2146304] GDN state fp32 [32,128,128] (S[h,dk,dv]) —
  matched my recomputed 11-token delta-rule state at corr 1.00000 (also
  independently confirms the GDN prefill math + that GDN_prefill's a3 output
  IS the state). L2 state = the known KV pack (k@0, v@byte 1073152).
  At the prefill→decode boundary FLM D2H/H2Ds all three states + uploads the
  517MB lm_head pool.
- CPU replication of decode steps (tools/kernel-interp/decode_step.py):
  structurally consistent — corr ~0.6 single step, stable ~0.45 over 5 CHAINED
  steps with my own state updates (no collapse ⇒ state-update semantics
  right). The gap to 1.0 is numeric drift amplified by softmax/routing (same
  signature as the from-scratch prefill composite: per-op 0.95-0.999,
  composite ~0.5). Sensitivity sweep confirmed: qk-l2norm required, conv-state
  order [t-3,t-2,t-1] required, shared-expert sigmoid gate DOMINATES the
  hidden (removing it → corr 0). Exact equality is only expected when driving
  FLM's own kernels (M3): replay-verify decode kernels byte-exact from our
  process is the remaining hard proof.
- Open minor items: even-row logits readback path; the master run
  (0x…3914F38)'s creation args (constructed before capture window); whether
  long prompts use the attn xclbin for prefill (our capture: 11 tokens, CPU
  attention).
- **M1 — per-kernel correctness.** Same, for each kernel we need: `dequant_mm`,
  `attn`, `conv`, `GateDeltaNet_prefill`, `lm_head`. Each validated vs torch and
  vs a live bo-captured FLM input/output pair.
- **M2 — full forward on CPU backend. CORE DONE (2026-08-30).** Rust engine at
  `npu-engine/` (src/q4nx.rs = verified dequant, src/forward.rs = config-driven
  forward with prefill + stateful decode; layer schedule auto-derived from the
  q4nx tensor names). Parity vs the capture-validated Python reference:
  prefill corr 1.000000 / identical argmax (4.3s for the whole 3LiF forward
  incl. lm_head, scalar single-thread); decode block-2 exact, chained blocks
  0.999+. **THE M2 RESULT: `run` on the interval-3 slices — model_5Li3
  (FLM: 3.4e38 overflow) and model_6Li3 (FLM: 100% NaN logits, "////////")
  — produces finite, healthy logits (absmax 7.4 / 9.0, residual stream ~4.5),
  matching the HF golden-reference scale. The interval-3 bug is fixed by
  construction in the open host.** Commands: `open-qwen-npu prefill|decode`
  (parity vs tools/kernel-interp/rust_ref + C:/caps/m0c states),
  `open-qwen-npu run <model.q4nx> <ids,csv>`.
  Remaining M2 polish: tokenizer + sampler + generate loop; run Cyrus's real
  30-layer model (needs GGUF→q4nx conversion — separate track); perf
  (parallelize matmuls / cache dequant).
- **M3 — full interval-3 forward on NPU. LARGELY DONE (2026-08-30).**
  - `m3_chain.exe` (npu-engine/m0/m3_chain.cpp): script-driven multi-run NPU
    harness (persistent BOs, multiple xclbin contexts) — the decode driver.
  - **Pool builder (tools/kernel-interp/build_pools.py) is byte-exact**:
    rebuilds FLM's L0 512MB pool bit-for-bit from the q4nx file. Laws: std
    matmul perm (+cols mod in_dim; share_down uses std law, expert down its
    own banding), expert stripe transpose, full-attn layout
    [q-half|k|v|gate-half|o] at 505282560+ (decoded from ctrlcode addresses,
    verified by BYTE-EXACT q/gate/k/v prefill-op replays vs captured outputs),
    lm_head = 128-row supertile transpose (k = 32*(v//128)+4*(col//256)+
    (v//32)%4; decoded via one-hot linear probes of the kernel).
  - **lm_head.xclbin driven from OUR process with OUR built pool reproduces
    FLM's captured logits BYTE-EXACTLY** (hash d5a4a650…, block-1 target).
    The lm_head kernel rms-normalizes row0 itself and multiplies row1
    (=model.norm.weight riding in the act buffer).
  - layer.xclbin (fused whole-layer decode kernel): drivable, deterministic;
    with pre-normed input its state update and in-place output verified at
    0.999+ vs CPU (y = x + Δattn + Δmoe, both phases every invocation).
    Input stage = windowed dynamic power-of-2 normalization (state invariant
    for input scales ~[1/4, 4]; saturates below). 2MB pack =
    [input_ln @0 | postln @4096 | shared_gate @8192 | router @12288].
    480B per-token ELF = 4 seqlen pokes into AIE tile addresses
    (0x06202400/0x06309000/0x06402400/0x06509000).
  - **Fixed a REAL reference bug found by the NPU comparison: file ssm_a is
    pre-baked −exp(A_log); Python+Rust were double-exponentiating** — this
    was most of the earlier "drift" (per-op decode verification now 0.9977+).
  - **DECODE CHAIN BYTE-EXACT (2026-08-30): the full decode step (3 fused
    layer kernels + lm_head) driven from OUR process with OUR built pools
    reproduces FLM's captured logits BYTE-EXACTLY across 3 consecutive tokens
    with persistent in-place state carry** (blocks 2/3/4, hashes match,
    identical argmax). The earlier "mismatch" was an experiment bug: block 1
    is lm_head-only on the prefill hidden (act buffer 000896 = raw residual
    hidden, NOT an embedding); the full layer chain starts at block 2 with
    act 000904 = [embed(sampled tok) | model.norm.weight]. The layer kernel
    rms-normalizes+ln-weights row0 itself using the pack (input_ln @0) — that
    resolves the "raw embedding" puzzle. L1(linear) and L2(full-attn) use the
    SAME 53840B ctrlcode ELF ⇒ the fused decode kernel is a config-agnostic
    universal layer executor; the interval schedule lives in orchestration.
  - **INTERVAL-3 ON NPU — DONE (2026-08-30): the 5Li3 (L,L,F,L,L) model runs
    on the NPU via open code and produces FINITE healthy logits (absmax 7.85,
    zero NaN) where FLM overflows to NaN.** run_5li3_npu.py builds all NPU
    buffers for any slice from the q4nx file + serializes our CPU-computed
    prefill states (corr 0.9999 vs captured format). Pool builder byte-exact
    (build_pools.py: pool/pack/side all verified vs 3LiF captures). Full
    5-layer + lm_head decode step ran on the NPU (state=4 every run).
    Practical note: run each layer as a separate process (carry act via file)
    — a single-process 5-layer chain times out on in-process XRT resource
    accumulation (each layer works standalone; needs proper runlist/context
    lifecycle in the real engine, an M4 detail not a correctness issue).
- **M4 — productionize. STARTED (2026-08-30): autoregressive interval-3
  generation runs on the NPU.**
  - `npu-engine/m0/decode_driver.cpp`: single-process decode driver. Kernels
    created once and reused (fresh xrt::run per submit); layers batched into
    xrt::runlist submissions. **KEY HW FINDING: the layer.xclbin hw_context
    times out after ~3 consecutive submissions (runlist capped at 3 runs);
    a submission on a DIFFERENT context (lm_head) between chunks resets it.**
    So decode = layers in runlist chunks of ≤3 with a cross-context barrier
    between chunks. (Why FLM's 40-layer models don't hit this is an XRT
    version/queue-config question — an M4 perf item, not correctness; likely
    an ERT command-queue-depth setting.) 3-layer 3LiF decode via the driver
    is byte-exact vs FLM.
  - `tools/kernel-interp/generate_npu.py`: **autoregressive generate loop.
    The interval-3 model (5Li3) generates FINITE tokens on the NPU** — NPU
    runs all 5 transformer layers per token (states carried in place), CPU
    reads the final hidden, computes full logits, samples greedily, feeds the
    next embedding. 5 tokens generated, all finite (absmax 10-13, zero NaN),
    sequence evolves (no NaN-pin); the sampled token matches the pure-CPU
    argmax. FLM emits constant-NaN "////////" here.
  - Fixed a real builder bug: `build_pools.build_lmhead_pool` used the std
    matmul perm; lm_head needs the 128-row supertile transpose. With it
    fixed, NPU lm_head output matches CPU at corr 0.999983 (same argmax).
  - Sampling note: the NPU lm_head kernel emits only the ODD vocab half
    (v=2b+1) into its 1MB buffer; the even half is a separate FLM pass. The
    generate loop sidesteps this by computing full logits on CPU from the
    NPU's final hidden (lm_head is a plain matmul, not interval-specific).
  - **RESIDENT SERVER DONE: ~0.1s/token.** `decode_driver.cpp` `serve` mode
    keeps pools AND per-layer states resident across tokens (no disk
    round-trip); reads `step <act_in> <hidden_out>` per token over stdin. A
    trailing cross-context barrier per step keeps the layer queue under its
    ~3-cap across step boundaries. `generate_npu.py` caches the dequantized
    lm_head [248320,2048] once (sampling = one matmul). 8 interval-3 tokens
    generated at ~0.1s/token, all finite (absmax 10-13), states resident,
    same token ids as the disk-carried + pure-CPU paths. Startup ~7s (2.5GB
    pools + 2GB lm_head cache from disk).
  - Remaining M4: tokenizer + real sampler (temperature/top-p); full 30-layer
    model (pool builder already general); resolve the ERT queue-depth cap so
    all layers fit one runlist (removes the barrier hack + its wasted lm_head
    passes); port the driver to the Rust core via the C-ABI xrt-shim;
    FLM-compatible HTTP server; GGUF→q4nx converter for Cyrus's published 27B.

## Decisions

- **Language: RUST (decided 2026-08-29).** Core engine in Rust for the open-
  community story and safety. XRT is reached via FFI to its **C API** (the
  undecorated `xrt*` exports — cf. `tools/seq-capture/xrt_coreutil_orig.capi.def`,
  62 C symbols), not the mangled C++ API. The ctrlcode/q4nx helpers get ported to
  Rust (spec: open `npu_instr_utils.hpp` / `dequant.hpp`). **M0 exception:** prove
  kernel-reachability first with the lowest-friction tool available on this box
  (pyxrt if it exists here, else a tiny C probe against the XRT C API), before
  building the Rust FFI layer — de-risk before investing.
- **Weights:** load FLM's `q4nx` directly (reuse the open dequant/reorder from
  `dequant.hpp`) so we run the *same* weights FLM does; GGUF import later.
- **XRT access = Rust core + thin C++ `xrt-shim` (C ABI) over FLM's `npu_utils`.**
  The kernels are driven by the modern XRT **C++** ELF/module/ext::kernel flow
  (`xrt::device→xclbin→hw_context→elf→module→ext::kernel→run`, per
  `npu_instr_utils.hpp`), which has no clean C bindings — so rather than
  reimplement it in Rust, wrap FLM's already-working `npu_utils` in a small
  `extern "C"` shim and FFI to that from Rust. Reuses the open wrapper verbatim;
  keeps engine logic in Rust.
- **Platform:** on **Windows** (Cyrus's box) NPU access must go through XRT
  (`xrt_coreutil.dll`); `amdxdna_accel.h` is a **Linux** DRM-ioctl UAPI, so a
  driver-direct (XRT-free) backend is a *Linux-only* future option, not the
  Windows path.

## Dev-env resolution (M0 unblocked)

- XRT C++ headers: **not on the box, but XRT is open (git reachable)** — vendor
  the `xrt/*.h` + `experimental/xrt_{elf,module,ext}.h` matching runtime
  **32.00.20102.3931** (from the XRT repo's AIE/amdxdna branch or a Ryzen AI SW
  install). Import lib already exists (`tools/seq-capture` gendef'd it).
- `pyxrt`: **not available** (not on PyPI, no Ryzen AI SDK here) → M0 is native
  (C++ shim probe), not a Python prototype.
- Rust 1.92 + cargo present; crate scaffolded at `npu-engine/`.

## Dev-environment gaps to close before M0

- XRT **SDK headers + import lib** on the Ryzen box (only `xrt_coreutil.dll`
  runtime is present; need `xrt/` headers + `.lib`, from the Ryzen AI / XRT SDK
  or `gendef` on the DLL). `pyxrt` is not installed (would enable a Python M0
  prototype if we go that route).
- Confirm we can open an hw-context on the NPU and load an xclbin from our own
  process (the proxy proved the DLL loads; M0 proves *we* can drive it).

## Openness ceiling (honest, for the README one-liner)

Open host + orchestration + scheduling + model support; **AMD's xclbin kernels
stay closed for now** (reused under `src/xclbins/.../TERMS.md`, revenue-capped).
That already removes the closed, buggy, lock-in layer. Fully-open AIE kernels is a
separate research effort (no open DeltaNet AIE kernel exists) and hardware-
bandwidth-capped — not needed to prove the concept or run the model.

## Validation / spec traceability

Golden reference (`tools/golden-ref`) is the whole-model oracle; per-op torch +
live bo-captures are the kernel oracles. Every milestone has a concrete numeric
pass/fail against one of these — no "looks right".
