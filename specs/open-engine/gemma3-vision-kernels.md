# Gemma 3 4B vision projector NPU kernels

The four Gemma 3-family 4B entries in the pinned FLM registry still need a
vision-tower route. This work covers the projector after the SigLIP tower,
not the tower itself. The Google/Hugging Face Gemma 3 projector performs 4×4
average pooling from a 64×64 patch grid to 256 tokens, applies RMSNorm with
`1 + weight`, then projects from 1152 vision channels into the 2560-channel
text space. See the [Gemma 3 projector source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py).

`gemma3_vision_pool_pack.py` groups the 4096 row-major SigLIP patches into
256 spatial 4×4 groups. The pool/norm NPU image accepts BF16[256,16,1152]
and a BF16[1152] normalization weight; it emits BF16[256,1152]. The projector
matrix image uses the existing BF16 pre-tiled GEMM at M256×K1152×N2560 and
emits FP32[256,2560]. `build_gemma3_vision_projector.py` builds both images
into a hashed manifest.

On 2026-10-27 the full 256-token NPU pool/norm check passed with maximum
relative error `0.00278` and `99.97%` exact BF16 elements. The composed
pool/norm-to-GEMM check passed with maximum relative matrix error
`1.01e-6` against an FP64 matrix reference. The spatial pack test checks
several corner and interior groups.

Remaining work includes the SigLIP patch embed and transformer layers,
real FLM tensor-layout conversion for the projector weight, variant-specific
geometry checks, and image-to-text output acceptance. These 4B entries remain
text-only recipe routes in the coverage matrix until the full vision path is
validated.
