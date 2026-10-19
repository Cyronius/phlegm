# Feasibility: open-source NPU loader for Qwen3.6-27B-A2.8B on AMD XDNA2

Research date: 2026-08-27. Target model: [Cyronius/Qwen3.6-27B-A2.8B](https://huggingface.co/Cyronius/Qwen3.6-27B-A2.8B). Goal: offload it to a Ryzen AI NPU with an open-source loader, Rust preferred; assess whether plugging into FastFlowLM makes sense given its closed components.

## Headline facts that change the premise

1. **AMD acquired FastFlowLM (2026-07-17).** The repo moved to the ROCm org ([github.com/ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM)) with v1.0.0 on 2026-08-11, and **the entire C++ runtime is now MIT-licensed source** — runner, server, model graph code (`modeling_llama3.cpp`, `automodel.cpp`, etc.), CMake builds for Linux and Windows. The "closed cpp components" era ended a month ago. ([flm_story.md](https://raw.githubusercontent.com/ROCm/FastFlowLM/main/flm_story.md), [Phoronix](https://www.phoronix.com/news/FastFlowLM-1.0))
2. **What remains closed:** the hand-tuned AIE kernels, shipped only as precompiled `.xclbin` binaries in `xclbins/`, and the NPU firmware (`npu.sbin`, redistributed via linux-firmware). No stated plan to open the kernels.
   - Licensing wrinkle: README now says the kernel binaries are "completely free for any use, including commercial use," but [TERMS.md](https://github.com/ROCm/FastFlowLM/blob/main/TERMS.md) still carries a USD 10M revenue cap. Contradictory; if commercial use matters, get written clarification (info@fastflowlm.com).
3. **FLM already runs the base model of this prune.** `qwen3.6-moe:35b-a3b` is in the FLM registry (Q4_K, 256k context, moved from Q4_1 to Q4_K in v1.0.3 released today). That means the closed kernels already implement the hard parts of this architecture: Gated DeltaNet linear attention, gated GQA attention, and 256-expert MoE FFNs.
4. **The model is a depth-only prune of that exact base.** Cyronius removed 10 of 40 layers (1 of every 3 DeltaNet layers; all 10 full-attention layers kept) from Qwen/Qwen3.6-35B-A3B, then LoRA-healed. Per-layer shapes and expert config (256 experts, 8+1 active) are **unchanged**. Files: Q4_K_M GGUF (16.5 GB), IQ3-experts GGUF (12 GB), bf16 safetensors (~55 GB). Runs on stock recent llama.cpp (`--spec-type draft-mtp` for the MTP head).
5. **An open weight converter exists:** [FastFlowLM/FLM_Q4NX_Converter](https://github.com/FastFlowLM/FLM_Q4NX_Converter) (Python) converts GGUF → FLM's tile-aligned `.q4nx` format. [Issue #501](https://github.com/FastFlowLM/FastFlowLM/issues/501) asks for custom-MoE q4nx support and has no answer — this is an open community gap.

## The three difficulty tiers

### Tier 1 — run this model in FLM as-is (days to ~2 weeks)

Because the prune preserves per-layer shapes, the existing closed kernels should execute it; only the *loader/manifest* layer needs to accept 30 layers instead of 40.

Work items:
- Convert the model (GGUF or bf16) to `.q4nx` with FLM_Q4NX_Converter. Risk: converter may lag the v1.0.3 Q4_K MoE layout; may need patching (it's open Python).
- Author the model manifest (quantization, context, tokenizer, memory plan) by cloning the `qwen3.6-moe:35b-a3b` package from `huggingface.co/FastFlowLM` and editing layer count.
- If the MIT runtime hardcodes depth per model tag, patch it — source is available.
- Drop/ignore the MTP head (FLM has no speculative-decode path; llama.cpp keeps that advantage).

This is exactly "an open source loader that plugs into FastFlowLM," and post-acquisition it's not just feasible, it's upstreamable to a ROCm-org repo.

### Tier 2 — a proper open loader/converter tool, Rust (2–6 weeks)

A Rust CLI that takes any Qwen3.6-MoE-family GGUF/safetensors (including pruned variants), emits `.q4nx` + manifest, and validates against the runtime. Pairs with a PR to ROCm/FastFlowLM making depth/expert-count manifest-driven. Rust is fine here — this tool is pure file-format work (GGUF parsing, Q4_K → q4nx tile repacking); the NPU memory allocation is FLM's job at runtime (XRT `xrt::bo` over the amdxdna driver), not the loader's. This is the highest community-value-per-effort option and directly answers issue #501.

Caveat: `.q4nx` tile layout is only documented by the converter's source; treat that as the spec.

### Tier 3 — fully open stack, no closed kernels (many months, research-grade)

For the ideological version (no closed xclbins at all):

- **Memory/dispatch layer — genuinely easy.** Linux `amdxdna` driver is mainline since 6.14 (GPL-2.0, `/dev/accel/accel0`). The UAPI is one small header ([amdxdna_accel.h](https://raw.githubusercontent.com/torvalds/linux/master/include/uapi/drm/amdxdna_accel.h)): `CREATE_BO`/`SYNC_BO`/`EXEC_CMD`/`WAIT_CMD` + mmap. Bindgen it from Rust; the Apache-2.0 XRT shim in [amd/xdna-driver](https://github.com/amd/xdna-driver) `src/shim/` is the reference implementation. The only Rust crate ([xrt-rs](https://crates.io/crates/xrt), AGPL, last release June 2024) won't save meaningful work — direct ioctls are cleaner. **The "loader that allocates RAM on AMD shared mem" part is a few days of Rust.** Windows has no open path (closed MCDM driver; usable via `xrt_coreutil.dll` but not open).
- **Kernels — the real cost.** Open toolchain is real: [mlir-aie/IRON](https://github.com/Xilinx/mlir-aie) + [Peano/llvm-aie](https://github.com/Xilinx/llvm-aie) (Apache/MIT), and [amd/IRON](https://github.com/amd/IRON) now ships open GEMM/GEMV/softmax/MHA-GQA/RMSNorm/RoPE kernels plus an end-to-end Llama 3.2 1B example. Open GEMM hits 38 TOPS int8 on XDNA2 (76% of peak, [arXiv 2512.13282](https://arxiv.org/pdf/2512.13282)). But **nobody has open AIE kernels for Gated DeltaNet**, and **MoE routing on the NPU is unpublished territory** — static dataflow overlays + xclbin reconfiguration overhead make dynamic expert dispatch structurally awkward ([TileFuse, arXiv 2606.11357](https://arxiv.org/html/2606.11357)).
- **Physics caps the payoff.** Measured NPU↔DRAM bandwidth is ~50 GB/s on XDNA2 vs ~218 GB/s practical for the Strix Halo iGPU ([Chips and Cheese](https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling)). Decode is bandwidth-bound: 2.83B active params at ~4.5 bpw ≈ 1.6 GB/token → ~30 tok/s theoretical NPU ceiling before overheads, vs ~4× that on the iGPU. Every open effort ([OllamaAMDNPU](https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU), TileFuse, [open-xdna](https://github.com/Scottcjn/open-xdna)) converged on the same conclusion: **NPU wins prefill and power efficiency, iGPU wins decode; hybrid NPU-prefill/iGPU-decode is the sane architecture.**

## Ecosystem notes

- The NPU stack is not ROCm-the-runtime (HIP/GPU) — it's XDNA/XRT/mlir-aie. But FLM now lives in the ROCm GitHub org, so Tier 1/2 contributions do land inside the ROCm umbrella.
- llama.cpp upstream rejected an NPU backend as a maintenance liability ([issue #14377](https://github.com/ggml-org/llama.cpp/issues/14377), closed stale).
- Model memory budget: ~19–20 GB total for Q4_K_M + 32K context (KV is tiny — only 10 attention layers cache KV, ~20 KB/token). Fits 32 GB shared-memory machines comfortably.

## Recommendation

Do Tier 1 now to get the model running; grow it into Tier 2 (Rust GGUF→q4nx converter + variable-depth runtime PR) as the community contribution. Skip Tier 3 unless the goal is research or the closed-binary licensing ambiguity turns out to be a real blocker — and if you do go there, target NPU prefill only and let the iGPU decode.
