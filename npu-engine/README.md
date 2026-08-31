# open-qwen-npu (working name)

An **open** execution engine for the Qwen3.6-MoE family on the AMD Ryzen AI
(XDNA2) NPU — including hybrid-attention pruned variants (interval-3) that the
current closed FLM engine mis-executes. It reuses AMD's NPU kernels and makes the
host-side orchestration open, correct, and extensible.

> Status: **proof-of-concept, in development.** Not usable yet — see
> [`.claude/plans/npu-open-engine.md`](../docs/npu-open-engine.md) for
> the plan and current milestone. Reuses FLM's closed xclbin kernels for now
> (open kernels are a later goal).

## Prior art for open XDNA2 kernels (surveyed 2026-08-30)

If/when we pursue writing our own kernels instead of reusing FLM's closed
`.xclbin`s, these are the existing projects worth reading first rather than
starting from raw MLIR-AIE examples:

- **[xdna-engine](https://github.com/atassis/xdna-engine)** — closest
  architectural sibling to this crate: Rust + XRT + hand-written AIE kernels
  via MLIR-AIE/IRON, targeting XDNA2/Strix. Resident dataflow, fused decode,
  KV cache, multi-precision GEMM/GEMV, CPU fallback for unported ops. Same
  "own the host, write real kernels" philosophy as this project.
- **[OllamaAMDNPU](https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU)** —
  a llama.cpp fork with an XDNA2 NPU backend (Ryzen AI MAX): compiles kernels
  via IRON, caches the resulting `.xclbin`s, dispatches via XRT. Matmul
  offload only. Reports the NPU running at **0.6% AIE utilization** —
  concrete evidence that "compiles and runs" is a long way from "uses the
  hardware," i.e. correctness and performance are separate projects. There's
  a matching open feature-request on upstream llama.cpp:
  [ggml-org/llama.cpp#21725](https://github.com/ggml-org/llama.cpp/issues/21725)
  (not merged anywhere).
- **[TileFuse](https://arxiv.org/abs/2606.11357)** (arXiv, June 2026, no
  public code found yet) — describes almost exactly our `mm.xclbin`/
  `dequant_mm.xclbin`: a fused unpack+dequant+GEMM/GEMV kernel for W4A16/W8A16
  on XDNA2, tiled to 32K dims, full 4×8 AIE array. Watch for code — would be
  a direct replacement candidate for two of our five reused kernels.
- **[Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie)** (IRON) — the AMD
  compiler/kernel-authoring toolchain underneath all three projects above.
  Actively maintained (1.2 shipped 2026-08); mature enough now that
  independent people are shipping real, if unoptimized, inference backends
  on it — not just toy examples.
- **[open-xdna](https://github.com/Scottcjn/open-xdna)** — same toolchain
  lineage but targets XDNA1 (Phoenix/Hawk Point), the wrong generation of
  silicon for our XDNA2 (Strix Point) test machine. Low relevance here.

**What none of them have:** a gated-DeltaNet / linear-attention recurrence
kernel, or MoE expert-routing kernel, for AIE. Every project above targets
the generic dense-decoder case (matmul, standard softmax attention). Josh's
model's hybrid linear-attn + full-attn + MoE architecture, fused the way
FLM's `layer.xclbin` does it, isn't published anywhere — that piece stays
novel work regardless of what we borrow.
