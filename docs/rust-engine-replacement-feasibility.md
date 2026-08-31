# Feasibility: Rust replacement for the closed FLM engine lib (qwen3_6_moe_npu)

Question: how involved is replacing `src/lib/{xrt,hrx}/qwen3_6_moe_npu.{dll,so}` with an
open Rust implementation, reusing the shipped closed xclbin kernels and cribbing model
math from open source (llama.cpp)?

Verdict: **more tractable than "rewrite an inference engine" sounds — Tier 1 below is
plausibly 4–8 weeks — because three of the four hard-sounding layers are already open,
and the pruned model's per-layer shapes are identical to the official 35B's, enabling a
capture-and-replay strategy for the one closed layer.** But it is still the slow road to
running the pruned model; Tier 0 (days) may fix the actual blocker directly.

## What the closed engine actually does (~2.4 MB per backend)

Behind the 9-virtual `causal_lm` ABI ([causal_lm.hpp:15](../../src/include/causal_lm.hpp#L15),
[qwen3_6_moe_npu.hpp:65](../../src/include/models/qwen3_6_moe/qwen3_6_moe_npu.hpp#L65)):

1. **q4nx weight loading** → XRT buffer objects (format already reverse-understood:
   safetensors container, header dumped 2026-08-26).
2. **Instruction-sequence generation** — the DMA/tiling choreography each xclbin
   expects, per op and shape. This is the real IP and the *only* genuinely closed
   logic layer (e.g. `Gemm::generate_seq` hides behind a pimpl —
   [gemm.hpp:47-49](../../src/include/modules/gemm.hpp#L47-L49)).
3. **Layer scheduling** — which sequence runs per layer, driven by
   `num_hidden_layers` / `full_attention_interval`. **The interval-3 bug lives here.**
4. **CPU-side math** — sampler is open; norms/router/DeltaNet-decode-recurrence are
   presumed CPU/AVX inside the engine (`immintrin.h` in the public header;
   `GateDeltaNet_prefill.xclbin` is named *prefill*-only, suggesting decode recurrence
   runs on CPU).
5. KV cache + DeltaNet state management, checkpoint/restore.

## What is already open (verified in-repo, 2026-08-27)

| Layer | Where | State |
|---|---|---|
| AIE2 ctrlcode encoding, **disassembler, sequence differ** | [npu_instr_utils.hpp](../../src/include/npu_utils/npu_instr_utils.hpp) (735 lines, header-only) + `instr_utils/npu_cmd_*.hpp` (write, dma, wait, maskwrite, issue_token, preemption) | Open, MIT. Cites mlir-aie sources for the format. Can load a sequence from file and print/diff it. |
| Kernel execution / driver | [npu_utils_xrt.hpp](../../src/include/npu_utils/npu_utils_xrt.hpp) (663 lines), [amdxdna_accel.h](../../src/include/npu_utils/amdxdna_accel.h) in-repo | Open. XRT and hrx backends. |
| Kernels | `src/xclbins/Qwen3.6-35B-A3B-NPU2/` — 9 **op-level** xclbins: `mm`, `dequant_mm`, `attn`, `GateDeltaNet_prefill`, `layer`, `lm_head`, `conv`, `vision_attn`, `vision_mm` | Shipped in-repo, closed, revenue-capped license (TERMS.md). Op granularity, not one monolithic overlay. |
| Engine ABI | `causal_lm.hpp` — forward/prefill/load_weights/set_context_length/clear_context/kv-cache getters/checkpoint/restore | Open, tiny. |
| Direct-drive harness | [test.cpp](../../src/test/qwen3_6_moe_npu/test.cpp) (125 lines) instantiates the engine without CLI/server | Open — natural capture point. |
| Model math reference | llama.cpp (MIT — license-compatible): gated DeltaNet recurrence, gated GQA, MoE routing, RMSNorm; it runs the pruned GGUF *today* on CPU/iGPU | ggml has **no** XDNA backend — nothing NPU-side to take, only correctness references. |

**The moat, precisely located:** not the instruction format (open), not the driver
(open), not kernel availability (shipped) — only the *sequence-generation policies*:
what choreography each xclbin expects for a given shape.

## The decisive advantage in this case

The prune keeps per-layer tensor shapes untouched — only layer count/order changed. A
replacement engine therefore **does not need general sequence generation**. It needs the
fixed sequences for: DeltaNet-layer decode, full-attn-layer decode, both prefill
variants, embedding, lm_head — at the exact shapes the closed engine already generates.
Weight offsets are explicit parameters of `generate_seq`, i.e. the sequences are
templates parameterized by offsets.

→ **Capture-and-replay:** link the closed `.lib` from a C++ harness (test.cpp as the
base), capture the emitted `npu_sequence` per op, disassemble/verify with the in-repo
tooling, then replay from Rust in `[L,L,F]×10` order instead of `[L,L,L,F]×10`.

## Tiers

### Tier 0 — locate/fix the interval-3 defect (days) ← do this first regardless

**Interception boundary confirmed (2026-08-27).** The engine ships as a prebuilt DLL
(`src/lib/{xrt,hrx}/qwen3_6_moe_npu.dll`, ~2.4 MB); only a header is open. Its PE
import table imports exactly one non-system, NPU-touching DLL: **`xrt_coreutil.dll`**
(plus `q4_npu_eXpress.dll`). The layer loop + scheduling are baked into the engine DLL,
so **a header hook on `npu_app` does NOT work** — editing `npu_utils_xrt.hpp` recompiles
only `flm.exe`, never the engine's own template instantiations. All NPU submission
instead funnels through `xrt_coreutil.dll`, which is the real capture point.

Chosen approach — **`xrt_coreutil` proxy shim** (built, see `tools/seq-capture/`):
- Proxy `xrt_coreutil.dll` forwards all 541 real exports to a renamed
  `xrt_coreutil_orig.dll`; Detour-hooks `xrt::elf::elf(const void*, size_t)` (export
  `??0elf@xrt@@QEAA@PEBX_K@Z`, @41) to dump every control-code blob in submission order,
  plus `run::start`/`run::wait` for ordering. No engine rebuild, no Ghidra.
- Run the interval-4 (working) and interval-3 (broken) models with the same prompt,
  then `seq_diff.py` aligns the two ordered op streams. A missing/extra/reordered layer
  op is the interval-3 scheduler defect made visible. Then disassemble the divergent
  `.seq` with the in-repo differ (`npu_sequence::interpret()`) for byte-level proof.
- Requires the Windows build env (MSVC + vcpkg Detours) and the test machine
  (HX PRO 370) to run a live capture — neither is set up yet; that is the only blocker.

Fallback if the shim is inconclusive: Ghidra pass over `qwen3_6_moe_npu.dll` anchored on
the `full_attention_interval` string (bug smells like a hardcoded `%4`/table where the
config value should be). A small binary patch may unblock the published model
immediately (private use; interop RE — don't redistribute the patched binary).

Minimum outcome: turns the ROCm/FastFlowLM issue from "degenerate output repro" into
"here is the defect", which is the highest-leverage path to an official fix.

**RESULT (2026-08-27) — capture built, ran, and diffed on the HX370.** The
`xrt_coreutil` proxy (`tools/seq-capture/`) works: it captured the live per-op
control sequences of two real runs through the closed engine —
`model_8Li4` (interval-4, 8L, coherent output: 821 ops) and `model_6Li3`
(interval-3, 6L, **reproduced the degenerate "////////" output**: 433 ops).

- Raw-byte diff is useless: per-layer weight-offset patching makes ~489 of 821
  ops byte-unique (`seq_diff.py`). **`seq_struct.py`** parses the AIE ctrlcode and
  fingerprints each op by command structure with the patched weight pointers
  masked → collapses to **16 structural op types**.
- **Decisive finding: the interval-3 bug is NOT in NPU op scheduling.** Every
  structural op type has identical counts in both runs; the ordered structural
  streams are identical except for pure-length runs of the two decode-loop ops
  (fewer iterations for 6 layers vs 8). No wrong op, no wrong shape, no reorder,
  nothing missing. Interval-3 executes the *same op sequence, in the same order*
  as the working interval-4 config.
- **Therefore the defect is CPU-side** — the logic keyed on
  `full_attention_interval` that XRT never sees: MoE routing, RoPE/position
  indices, the DeltaNet decode recurrence, or per-layer weight-pointer
  computation. This overturns the original Tier-0 hypothesis (a scheduling bug
  visible in a sequence diff).

### Tier-0b — TENSOR-DATA CAPTURE RAN (2026-08-27): the defect is a NaN blowup

The `xrt::bo` data path was hooked and both configs re-run. The engine imports
exactly three data-path symbols from `xrt_coreutil.dll` — `?map@bo` (get host
ptr), `?sync@bo` (flush a region H2D / D2H), `??1bo` (dtor); there is **no
`bo::write` import**, so map+sync is the whole path. The proxy now records, per
sync, `(dir, size, offset, fnv1a-hash)` and optionally dumps the bytes
(`FLM_BO_CAPTURE_DIR`, `FLM_BO_DUMP_MAX`). Tooling: `xrt_shim.cpp` (bo hooks
added), `bo_capture.ps1` (variant swap + run), `analyze_bo.py`, `scan_nan.py`.
Corpora: `C:\caps\bo_{i3,i4}_{meta,dump}`.

**Finding — interval-3 produces NaN, interval-4 does not:**
- The NPU moves every tensor through fixed **1 MB DMA staging tiles**, so real
  activations sit inside 1 MB syncs (offset 0). The final logits are fp32,
  vocab=248320 (~970 KB) → they fit one tile and are directly identifiable.
- **Broken (6Li3):** the first logits tensor (sync idx 910, immediately after the
  128 MB lm_head weight load at idx 900–901 = the prefill→decode boundary) is
  **100 % NaN** over its active region. Every subsequent decode step returns the
  *same* NaN logit tensor (`ee2486…`) → argmax is pinned → the `////////`
  output. So the blowup is present **by the end of prefill**, before any token is
  decoded, and is total, not drift.
- **Healthy (8Li4):** at the same structural point the logits are fully finite
  (±10 range), **zero NaN anywhere in the run**, and the argmax changes every
  decode step (39536 → 51403 → 37174). Confirmed varied, non-degenerate.
- Noise ruled out: a recurring 128 KB buffer reads as 65.6 % "bf16-NaN" in
  **both** runs (a non-float / int8 buffer whose bit patterns hit the bf16 NaN
  exponent). Only the 100 %-NaN 524 KB logit tensors are unique to the broken run.

**Conclusion:** interval-3 is not mis-scheduled (op sequence identical) and not
merely mis-routed — the CPU-side numerics **overflow to NaN during the prefill
forward pass**. No fully-NaN float activation is captured *before* the first
logits, but per-layer prefill activations aren't individually typeable at the bo
boundary (small tensors embedded in shared 1 MB tiles, decode is token-level not
layer-level), so the earliest-NaN layer can't be named from this capture alone.

### Tier-0c — DEPTH BISECTION RAN (2026-08-27): explosive activation growth at the interval-3 F→DeltaNet boundary

Built minimal single-variable slices with `slice_keep.py` (keep an explicit
old-layer list, renumber, config derived) and classified each by decode logit
behavior (collapse signature = the constant NaN logit tile `ee24861f195ff4b1`;
healthy = distinct logits per decode step). **Slices are cut from
`model.q4nx.orig`, never the live `model.q4nx`** (a capture run swaps a variant
into the live name).

| Variant | layer_types | interval | max\|logit\| | verdict |
|---|---|---|---|---|
| 3LiF | `[L,L,F]` | 3 | **10.6** | healthy |
| 4Li3 | `[L,L,F,L]` | 3 | **3.4e38** (fp32 max!) | "healthy" but at overflow edge |
| 5Li3 | `[L,L,F,L,L]` | 3 | **NaN** | collapse (`ee2486…` ×12) |
| 4LiF | `[L,L,L,F]` | 4 | ~10 | healthy |
| 8Li4 | `[L,L,L,F,L,L,L,F]` | 4 | ~10 | healthy (2 full-attn, 3 DeltaNet after each) |

**Mechanism (well-supported):** activations grow **geometrically through DeltaNet
(linear-attention) layers that follow the interval-3 full-attention layer** —
~10 → 3.4e38 across a *single* post-full-attention DeltaNet layer, then → NaN at
the second. A `[L,L,F]` stack is fine only because its full-attention layer is
last and the final RMSNorm rescales it before lm_head. Under **interval-4** the
same F→DeltaNet chain stays ~10 through three following DeltaNet layers, so the
fault is specific to how the **interval-3 full-attention layer's output feeds the
following DeltaNet recurrence** — a scale / normalization / state-index computed
from `full_attention_interval` that is wrong for interval=3, producing an
fp32 overflow (not a logic/scheduling error).

**Not yet distinguished:** whether the interval-3 full-attention layer emits
over-large values (masked by RMSNorm when it is the last layer) or the following
DeltaNet layer applies a wrong huge factor — both fit the ladder. The bo boundary
can't type per-layer prefill activations to settle it; Ghidra on the DeltaNet /
full-attention scale math (or an engine-side activation print) would.

### Tier-0d — GOLDEN REFERENCE RAN (2026-08-28): overflow is the engine's, not the math's

Ran the real **Qwen/Qwen3.6-35B-A3B** (built-in `qwen3_5_moe` arch in transformers
5.16.1 — gated DeltaNet `linear_attn.*` + periodic `self_attn.*` full-attention +
MoE) through the official forward, sliced to the exact failing configs, on a
RunPod RTX PRO 4000 Blackwell (`tools/golden-ref/reference.py`, terminated after).
All three interval-3 slices are **finite and well-scaled** — per-layer hidden
absmax grows only 3.92 → 4.03 → 4.81 across the F→DeltaNet→DeltaNet chain, logits
~8–10, sane argmax — exactly where the **engine** blows to 3.4e38 (4Li3) then NaN
(5Li3). So the interval-3 architecture is numerically fine; the closed engine's
DeltaNet-after-interval-3-full-attention scaling is the defect, confirmed against
ground truth. The reference logits + per-layer norms are the oracle/spec for a
Tier-1 host replacement (which reuses the working xclbin kernels). Weight source
settled: real HF bf16 weights, no q4nx dequant needed.

**Minimal repro for AMD/FLM:** `config_5Li3.json` + `model_5Li3.q4nx`
(`[L,L,F,L,L]`, interval 3) — smallest config that emits NaN logits; the
interval-4 sibling `4LiF` with the same early layers is finite. Report as a
**numeric-overflow bug**: "interval-3 activations blow past fp32 max through the
DeltaNet layers after a full-attention layer; interval-4 identical pipeline stays
bounded." Repros rebuild in ~1 min each:
`python slice_keep.py <model_dir> 5Li3 3 0,1,3,4,5`.

### Tier 1 — Rust replay engine (4–8 weeks)
Open crate implementing the `causal_lm` role for the Qwen3.6-MoE family shapes:
1. q4nx loader in Rust (~1 wk — safetensors container, format understood).
2. XRT/xdna FFI: bindgen over `xrt_coreutil` C API or `amdxdna_accel.h` ioctls
   (~1–2 wk; Windows uses XRT-on-MCDM, Linux can go straight to the mainline UAPI).
3. Port `npu_sequence` encoding to Rust (~1 wk, mechanical — spec is open).
4. C++ capture harness + captured-sequence corpus (~1 wk).
5. CPU-side math ported with llama.cpp as reference — DeltaNet decode recurrence,
   router, norms, sampler (~2–3 wk; the recurrence is the hairy part).
6. Integration/debug on hardware (open-ended; NPU debugging is slow).

Runs any layer count/order at 35B-family shapes → the pruned model, other prunes,
finetunes. **Risks:** sequences may embed more dynamic state than weight offsets
(BO address patching — check how npu_utils_xrt patches at submit); unknown CPU/NPU
split for DeltaNet decode until captured; sync-token/preemption semantics.

### Tier 2 — real sequence generation (+2–3 months on Tier 1)
Re-derive tiling policies from a captured corpus across shape sweeps → arbitrary
shapes within each xclbin's limits. This is the version worth announcing as a
community project: an open engine for FLM's NPU kernels.

### Tier 3 — open kernels too (6–12+ months, research)
IRON/mlir-aie kernels replacing the xclbins. No open DeltaNet AIE kernel exists; MoE
routing on static-dataflow overlays is unpublished; NPU DRAM bandwidth (~50 GB/s
class) caps the payoff regardless. Not recommended.

## Caveats

- **License travels with the kernels.** Tiers 0–2 still ship against the closed
  xclbins: TERMS.md's USD 10M revenue cap applies to any deployment, open engine or
  not. Publishing the *Rust code* is clean; publishing a *captured sequence corpus*
  is grayer (arguably derived from the closed lib) — prefer "capture locally at
  first run" in the tool design.
- **Perf ceiling unchanged.** Baseline 35B measured ~6.6 tok/s on the HX PRO 370;
  expect roughly +25–35% from the prune, not a different class of speed.
- **Cheapest path to running the model is still not this.** Re-prune by whole
  [L,L,L,F] blocks (interval 4, proven working) or an AMD fix via the Tier-0-fueled
  issue both beat weeks of engine work if the goal is just "my model runs".

## Relation to existing plan

[pruned-qwen36-support.md](pruned-qwen36-support.md) Phase 2 (converter) is untouched
by all of this and needed in every scenario. Tier 0 here supersedes that plan's
option (a) "file issue with repro" by strengthening it; Tier 1 replaces option (c)
"wait for AMD" with "don't wait".
