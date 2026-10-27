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
vision/audio, and unified 12B image/audio headers. The v2 matrix bundle also
builds two im2col shapes for the audio subsampling convolutions, 17 images total.
`build_gemma4_multimodal_ops.py` builds 36 additional operation images into a
manifest with image hashes.

| Primitive | Source | NPU check on 2026-10-27 |
| --- | --- | --- |
| Vision and audio BF16 matrix multiplication, M=256 | `open_kernels/designs/gemma4_mm/`, `build_gemma4_multimodal_mm.py` | Five representative matrices passed: K768×N768, K1024×N2048, K768×N2560, K640×N3840, K6912×N3840; maximum relative error `2.51e-6` |
| Bidirectional vision attention, 12 heads×64 lanes | `open_kernels/designs/whisper_fa/`, `gemma4_vision_fa_check.py` | Unit-scale image compiled for 2520 valid patches padded to 2560; low-magnitude Q/K passed (`0.0227` maximum relative error), but normalized-magnitude Q/K failed (`0.0961` maximum relative error against `0.04` limit); native BF16 matmul variant produced nonfinite output |
| Vision per-head Q/K/V RMSNorm, width 64 | `open_kernels/designs/gemma4_head_norm/` | 12 learned-scale and 12 unit-scale heads passed with exact BF16 output |
| Vision axial 2D RoPE, 24 Q/K heads×64 lanes | `open_kernels/designs/gemma4_vision_rope/` | Position pairs (23,47) and (1023,2047) passed with exact BF16 output |
| Vision FP32 online attention, 64-lane query with 2520 keys | `open_kernels/designs/gemma4_attention/`, `gemma4_vision_online_check.py` | 1024+1024+472 rows carried across NPU images; normalized-magnitude Q/K passed with maximum absolute error `3.37e-7` |
| Vision 768-wide RMSNorm/residual and 3072-wide GELU gate | `open_kernels/designs/gemma4_hidden_norm/`, `gemma4_gelu_mul/` | Both passed; RMSNorm/residual exact BF16, GELU gate maximum absolute error `3.81e-6` |
| Vision projected-patch plus learned x/y position vectors, 256×768 panel | `open_kernels/designs/gemma4_vision_pos_add/`, `gemma4_vision_position_pack.py` | FLM `[2,12,40,16384]` position table decoded to `[2,10240,768]`; NPU x+y and patch+position additions passed with exact BF16 output, including 40 padded rows |
| Vision/audio clipped linear input and output, widths 256 through 4096 | `open_kernels/designs/gemma4_clip/`, `gemma4_clip_reference.py` | BF16 pre-linear and FP32 GEMM-to-BF16 post-linear boundary images compiled for every matrix-output width in the multimodal plan; representative 768 pre and 1024/1536/3840/4096 post NPU checks passed exactly |
| Vision 3×3 patch pooling, 2520 patches→280 tokens×768 channels | `open_kernels/designs/gemma4_vision_pool/`, `gemma4_vision_pool_pack.py` | Full 280-token NPU check passed exactly against FP32 sum/9, BF16 average boundary, and √768 output scaling; one empty spatial group checked |
| Audio 5-tap depthwise convolution, 256 frames×1024 channels | `open_kernels/designs/gemma4_audio_dw/`, `gemma4_audio_dw_pack.py` | BF16 inputs and weights, FP32 output passed with maximum relative error `4.05e-8` |
| Audio stride-2, pad-1 3×3 subsampling convolutions | `open_kernels/gemma4_audio_subsample.py`, `gemma4_audio_subsample_check.py` | Both 1→128 and 128→32 channel stages passed 256-pixel NPU checks; maximum relative errors `3.11e-8` and `5.40e-7` |
| Audio subsampler LayerNorm/ReLU, 128 and 32 channels | `open_kernels/designs/gemma4_audio_subnorm/` | Both passed with exact BF16 output |
| Audio relative local attention, 8 heads×12 queries×24 context rows | `open_kernels/designs/gemma4_audio_attn/`, `gemma4_audio_attention_pack.py` | NPU computes projected-relative-key scores and shift, softcap 50, masks, softmax and value reduction; maximum relative error `1.21e-6` |
| Audio learned Q and fixed K scaling, 8 heads×12/24 rows×128 lanes | `open_kernels/designs/gemma4_audio_qk_scale/`, `gemma4_audio_qk_scale_pack.py` | BF16 learned per-dimension softplus, model Q/K factors and FP32 output passed full 8-head NPU check exactly |
| Audio hidden RMSNorm/residual, width 1024; pre-projection RMSNorm, width 1536; GLU/SiLU activations | `open_kernels/designs/gemma4_hidden_norm/`, `gemma4_audio_act/` | 1024- and 1536-wide norm passed exactly; 1024-wide GLU, 1024/4096-wide SiLU passed exactly at BF16 output precision |

The single-query vision online path is numerically accurate, while the
bidirectional image has a precision gap at realistic normalized inputs. The
remaining multimodal kernel gaps include image preprocessing, position-table
gathering and model-specific residual/normalization wiring. The audio
relative-key projection uses the compiled 1024×1024 matrix shape; its output
still needs model-level packing and clipped-linear handling. Model-level work
also needs tensor-by-tensor weight export, a scheduler connecting these
operations, real-package numerical checks, and image/audio output acceptance.
