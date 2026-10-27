# Gemma 4 multimodal NPU kernel progress

The three Gemma 4 FLM entries remain `missing_recipe` in the registry matrix.
The work here covers matrix multiplication and one audio convolution primitive;
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
| Audio 5-tap depthwise convolution, 256 frames×1024 channels | `open_kernels/designs/gemma4_audio_dw/`, `gemma4_audio_dw_pack.py` | BF16 inputs and weights, FP32 output passed with maximum relative error `4.05e-8` |

The remaining multimodal kernel gaps include vision attention, audio attention
with relative position terms, the audio 2D subsampling convolutions, image
patch processing, and tower normalization and activation. Model-level work
also needs tensor-by-tensor weight export, a scheduler connecting these
operations, real-package numerical checks, and image/audio output acceptance.
