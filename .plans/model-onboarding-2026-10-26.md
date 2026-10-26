# Model onboarding: October 26, 2026

The requested Hugging Face models span several incompatible decoder designs. A
model name or matching hidden size is not evidence that its weights, state update,
router and attention arithmetic work with an existing kernel set. Keep the NPU
path and its batched prefill for the models that match; build new families for the
others instead of routing them to a superficially similar recipe.

## Inventory from published configs

| Model | Decoder and current route | Next prerequisite |
| --- | --- | --- |
| [Ornith 1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/config.json) | Qwen3.5 MoE, 2048/40/256 experts; existing `qwen36moe` geometry and 256-row GEMM prefill | Convert weights to the supported container, then full-model NPU quality gate |
| [Nex N2.5 mini](https://huggingface.co/nex-agi/Nex-N2.5-mini/blob/main/config.json) | Same Qwen3.5 MoE text geometry | Same; vision tower is separate |
| [KAT Coder V2.5 Dev](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev/blob/main/config.json) | Same Qwen3.5 MoE text geometry | Same; vision tower is separate |
| [Nanbeige 4.3 33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/blob/main/config.json) | Llama dense parser derives its 7168/22016 geometry | Extend the down projection past the recipe's two-piece L1 split, then validate new dimensions and attention tuple |
| [LFM2.5 24B A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/config.json) | `lfm2_moe`: short convolution plus MoE | New combined LFM2/MoE recipe and expert-bias routing |
| [flint](https://huggingface.co/RecursiveSI/flint/blob/main/config.json) | `flint`: full/sliding attention plus MoE | New layer schedule, sliding state, learned softmax and MoE route |
| [Xing 4.0](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/blob/main/config.json) | `xing4_0`: MLA, mHC and MoE | New attention, residual and routing kernels |
| [Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/config.json) | `muse_glimmer`: multimodal custom text decoder | New text family, then image path |
| [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/config.json) | `nemotron_h`: hybrid Mamba/attention MoE | Mamba state, expert route and matching quantization |
| [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B/blob/main/config.json) | `gemma4`: local/global attention MoE | New text family and expert route; image path separately |
| [Underdog Saluki 27B](https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0) | Qwen3.8-derived GGUF, published with 2-bit weights | Confirm metadata/packing and add a supported low-bit NPU route or convert weights with quality gate |
| [K2 Horizon MoVA](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/main/config.json) | `k2_horizon` model type, but MoVA/MoE rather than the existing dense K2 | New MoVA router/expert recipe; dense K2 parser now refuses this config |
| [Raon 1 mini](https://huggingface.co/KRAFTON/Raon-1-mini-preview/blob/main/config.json) | `raon1_mini`: RDA/MLA, mHC, MoE, MXFP8 | New recurrence, attention, routing and format support |
| [Hy Memory](https://huggingface.co/tencent/Hy-Memory-30B-A3B/blob/main/config.json) | `hy_v3`: MoE and custom routing | New Hy V3 text family and expert route |

## Changes and verification

- The Qwen3.5 MoE config parser now reads a published VLM's `text_config` and
  validates its decoder model type. Flattened container configs retain their
  existing spec hash. A focused test checks the resulting kernel hash and the
  manifest's 256-row batched GEMM route.
- Dense K2 derivation now refuses MoVA/MoE configs instead of dropping experts.
- Manual offline checks on the Qwen3.5 MoE nested config matched the checked-in
  35B kernel hash and produced both full- and linear-attention 256-row block routes.
  Command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=open_kernels python -c` with
  inline spec-hash and manifest assertions (PASS). `git diff --check --` on the
  changed paths passed. `PYTHONPATH=open_kernels python -m pytest -q
  specs/open-engine/tests/test_spec_derive.py specs/open-engine/tests/test_k2.py`
  could not run: this environment has no pytest installation.
- No weights for these models were present locally. No download, conversion,
  kernel build, NPU inference or speed result was attempted. Hardware acceptance
  for even the matching Qwen3.5 models remains open.

Next action: convert one Qwen3.5 MoE model into the existing supported container,
run the full-model quality gate on the NPU, then evaluate block prefill and decode
with Dibs reservations. Follow with Nanbeige's three-way down projection split.
