# Gemma 4 multimodal NPU kernel progress

The three Gemma 4 FLM entries remain `missing_recipe` in the registry matrix.
The work here covers matrix multiplication, vision attention, and audio convolution primitives;
it does not yet execute a vision or audio tower.

Public q4nx package headers were inspected without downloading model payloads.
E2B and E4B vision matrices use FLM BF16 `[N/64,K/256,16384]` tiles. Their
audio matrices use the same layout. The 12B multimodal matrices use row-major
BF16. `gemma4_mm_source.py` converts either representation to the Phlegm
pre-tiled B layout while preserving BF16 bits. The bundle builder found and
compiled all 15 unique matrix shapes across the E2B vision/audio, E4B
vision/audio, and unified 12B image/audio headers.

| Primitive | Source | NPU check on 2026-10-27 |
| --- | --- | --- |
| Vision and audio BF16 matrix multiplication, M=256 | `open_kernels/designs/gemma4_mm/`, `build_gemma4_multimodal_mm.py` | Five representative matrices passed: K768×N768, K1024×N2048, K768×N2560, K640×N3840, K6912×N3840; maximum relative error `2.51e-6` |
| Bidirectional vision attention, 12 heads×64 lanes | `open_kernels/designs/whisper_fa/`, `gemma4_vision_fa_check.py` | Unit-scale image compiled for 2520 valid patches padded to 2560; NPU check passed at maximum relative error `0.0227` and cosine `0.999747` |
| Audio 5-tap depthwise convolution, 256 frames×1024 channels | `open_kernels/designs/gemma4_audio_dw/`, `gemma4_audio_dw_pack.py` | BF16 inputs and weights, FP32 output passed with maximum relative error `4.05e-8` |
| Audio stride-2, pad-1 3×3 subsampling convolutions | `open_kernels/gemma4_audio_subsample.py`, `gemma4_audio_subsample_check.py` | Both 1→128 and 128→32 channel stages passed 256-pixel NPU checks; maximum relative errors `3.11e-8` and `5.40e-7` |

The vision attention check covers SDPA after Q/K/V normalization and vision
RoPE. The remaining multimodal kernel gaps include those normalization and
2D RoPE operations, audio attention with relative position terms, image
patch processing, and tower normalization and activation. Model-level work
also needs tensor-by-tensor weight export, a scheduler connecting these
operations, real-package numerical checks, and image/audio output acceptance.
