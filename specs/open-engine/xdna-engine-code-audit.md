# xdna-engine: pinned source audit

Inspected 2026-10-25. Exact clone commit:
`35e4adf527131d99077e87117ad3f5d07a25b034`, local
`target/npu-dev/xdna-engine`. The checkout was clean when inspected. No AGENTS.md or CLAUDE.md
was found. Root license is Apache-2.0; individual AIE files inspected carry
`Apache-2.0 WITH LLVM-exception`. NOTICE attributes the engine to Taimuraz
Kaitmazov and lists separate model notices. No install/setup scripts or NPU
jobs were run. Source links below are pinned to this commit.

## What can transfer

**Device-resident operator sequences are real code, not just a README goal.**
[The prefill generator](https://github.com/atassis/xdna-engine/blob/35e4adf527131d99077e87117ad3f5d07a25b034/designs/decode_fused/gen_llm_prefill.py)
constructs `OperatorSequence(..., share_designs=True, extra_runlists=...)` around
all layer operations. Its precision knobs explicitly distinguish single-BFP
emulation, native BF16, and FP32 versus BF16 partial sums. This is not an
arithmetic-identical faster GEMM by default. The source reports sizable K/V
changes from BFP emulation; Phlegm must retain its own stricter gates.

[The Rust prefill path](https://github.com/atassis/xdna-engine/blob/35e4adf527131d99077e87117ad3f5d07a25b034/rust/npu-models/src/llm/npu_prefill.rs)
shares one device-allocated arena between decode and prefill, checks the arena
layout, writes embeddings/positions/masks, then dispatches named segments with
no intermediate readback. Tail padding is causal and kept within the cache
bound. Its reported large prefill speedup compares batching with its own
per-token path; Phlegm already batches, so that is not another multiplicative
win available here.

**Persistent-context full-ELF dispatch has an explicit C++ implementation.**
[The XRT shim](https://github.com/atassis/xdna-engine/blob/35e4adf527131d99077e87117ad3f5d07a25b034/rust/npu-xrt/shim/xrt_shim.cpp)
contains `shim_elf_ctx_open` / `shim_elf_kernel_rebind`: create a context from a
base ELF once, construct `xrt::module` from a variant ELF and bind
`xrt::ext::kernel(ctx,module,name)`. `shim_elf_resident_open_named` shares the
same context and ELF handle between named sequences, keeps `xrt::run` and
arguments bound, and only flushes a control scratchpad before `run.start()`.
This is reusable from C++; adopting its Rust engine is unnecessary. It does
not prove that arbitrary Phlegm xclbins with different static configurations
can be interchanged safely: the fused ELF/configuration builder is part of the
contract.

**Its resident MLP design is a concrete fusion reference.**
[`rmlp_design.py`](https://github.com/atassis/xdna-engine/blob/35e4adf527131d99077e87117ad3f5d07a25b034/designs/resident_forward/rmlp_design.py)
keeps x in memtiles, performs pre/post RMSNorm, gate/up, GELU*up, down,
residual and layer scaling in one command, with weights streaming. It uses
idle weight blocks for normalization and a cross-column sum-of-squares cascade.
This is useful as a topology example; its Gemma dimensions/activation and
rounding are not a drop-in Llama kernel.

## Findings that prevent misleading shortcuts

- **Residency corroborates our negative result.** The prefill generator leaves
  `A_RESIDENT` off. Its source reports that removing 3.92 GiB from 16 GiB of
  transfers gave about zero latency change over ten alternating pairs. This is
  an upstream reported measurement on its stack, not independently remeasured.
  Phlegm's proper memtile-repeat experiment was likewise correct but about 2%
  slower for the current shape. Lower byte count alone is not a speedup.
- **No exact BF16 fast-matrix escape hatch.**
  [`gemm_bf16xbfp16.cc`](https://github.com/atassis/xdna-engine/blob/35e4adf527131d99077e87117ad3f5d07a25b034/aie_kernels/gemm-bf16xbfp16/gemm_bf16xbfp16.cc)
  accepts BF16 activations but acknowledges they convert internally to BFP
  before the hardware matrix unit. It is a single-pump primitive, not an
  already optimized compensated Q4 replacement.
- **Native int4 group dequant is an interesting different algorithm, but not
  a validated GGUF Q4_1 replacement.**
  [`gemm_int8xint4_dequant.cc`](https://github.com/atassis/xdna-engine/blob/35e4adf527131d99077e87117ad3f5d07a25b034/aie_kernels/gemm-int8xint4-dequant/gemm_int8xint4_dequant.cc)
  accumulates int8 x signed-int4 products into int32, then scales each K group
  and sums FP32 partials. It explicitly excludes asymmetric/zero-point
  correction and labels device numerical correctness unverified. Its output
  is BF16, and its scale multiply can use emulated FP32 operations. Directly
  transplanting it would change format, arithmetic and precision.
- **Do not interpret all weights as on-chip.** The resident-forward README's
  wording overstates residency.
  [`rforward.py`](https://github.com/atassis/xdna-engine/blob/35e4adf527131d99077e87117ad3f5d07a25b034/designs/resident_forward/rforward.py)
  defines a large weight-stream argument, with 64-bit addressing past 8 GiB;
  `rmlp_design.py` explicitly says weights stream while activations are in
  memtiles. It is a command/dataflow improvement, not SRAM large enough for a
  12B model.
- **Tile registry evidence is mixed.**
  [`gemm_tile_registry.py`](https://github.com/atassis/xdna-engine/blob/35e4adf527131d99077e87117ad3f5d07a25b034/designs/decode_fused/gemm_tile_registry.py)
  keys precision/FP32-accumulator mode as well as shape and rejects absent
  shapes. The checked JSON has 173 entries: 15 `sweep`, 80 `seed`, 78 `assumed`.
  There is no exact 256x2048x8192 entry. Do not treat all entries as measured or
  transfer timing from its different hardware/precision.

## Stack and reproducibility limits

The README names Ryzen AI 9 465/Krackan under Linux. Its driver lock records an
amdxdna build reporting Gorgon Point, firmware 1.1.2.64. These names and their
absolute speeds do not establish equivalence to our Strix Halo MAX+ 395.

`toolchain.lock` pins MLIR-AIE fork 498f452a811a065a22c503ed8da13dafed5f1f91,
MLIR distro 24.0.0.2026091506+e4fcd128, Peano fork
aa4acd25694884aadad7bdb1479b72ec0399744c (LLVM 22 seed), and IRON
2f7df42ca571d557514fcf233da460f655234af4. That is a different stack from our
MLIR-AIE 1.4.2/Peano 21 kernels and should remain isolated if reproduced.
The repository's mirror contains only selected decode operators, not GEMM or
OperatorSequence. Fetching its exact IRON submodule pin from the declared
public remote failed with `upload-pack: not our ref`. Therefore the exact GEMM
and fused sequence builder behind this checkout was not available for a full
source audit here. No setup scripts were executed to conceal that gap.

## Feasibility follow-up and next experiment

1. **Native-int4 affine feasibility was tested and failed the operand gate.**
   A subsequent bounded CPU study used the existing synthetic `check_gemm`
   distribution at N=64, T=64, K=2048, seeds 212–214. It evaluated per-32-K
   `d*(q_signed @ X)+(m+8*d)*sum(X)`, with `q_signed=q-8`, using exact X and
   two/three int8 activation limbs. Ideal float64 restoration isolated the
   algorithm from hardware accumulation error. Against the existing BF16
   operand reference, even exact-X affine output had relative error
   0.001708–0.001721, exceeding the unchanged 0.0001 limit. All modes passed
   the separate 0.005 FP16-source limit. Extra activation limbs did not repair
   the nonlinear per-weight BF16 rounding residual. Pure affine native-int4
   hardware work is therefore not justified under this operand contract;
   a residual correction would need a new feasibility and cost analysis.
   This is a measured CPU failure, not a device result or a model-quality test.
   Reproduction script and raw results remain local:
   `target/npu-dev/parallel-int4/study_affine_int4.py` and `result-k16.json`;
   result SHA256
   `f5d3ab8273d7ea9490f0804a435347ede5152813a9f9ac12abcc112a9a79d827`.
2. **Execution-pipeline experiment:** once the exact public IRON source is
   available or an equivalent verified compiler path is chosen, build a tiny
   two-operator full ELF with a shared arena and named variants. Bind those
   variants in one C++ context through the shim pattern, measure configuration
   and host-copy savings, and require unchanged arithmetic/output bytes.
   Only then expand to norm/packed-panel/GEMM or gate-up/down chains. Current
   Phlegm already shares a context across projection shapes, so eliminating
   context switches alone cannot plausibly close the full 12x prefill gap.

No speedup is claimed from this source audit. At the audit checkpoint, a
separate local kernel experiment reported approximately 2.82 ms for the
compensated Q4 projection versus 4.27 ms native. That historical projection
measurement is not an xdna-engine comparison or a full-model speedup. Current
quality gates and performance evidence belong to
[Linux GGUF prefill](gguf-prefill-linux.md) and the
[active performance plan](../../.plans/artlite-performance.md).
