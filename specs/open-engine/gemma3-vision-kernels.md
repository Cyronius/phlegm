# Gemma 3 4B vision NPU kernels

The four Gemma 3-family 4B entries in the pinned FLM registry still need a
vision-tower route. This work covers patch embedding and the projector around
the SigLIP transformer; the transformer itself remains open. The Google/Hugging
Face Gemma 3 projector performs 4×4 average pooling from a 64×64 patch grid
to 256 tokens, applies RMSNorm with
`1 + weight`, then projects from 1152 vision channels into the 2560-channel
text space. See the [Gemma 3 projector source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py).

`gemma3_vision_pool_pack.py` groups the 4096 row-major SigLIP patches into
256 spatial 4×4 groups. The pool/norm NPU image accepts BF16[256,16,1152]
and a BF16[1152] normalization weight; it emits BF16[256,1152]. The projector
matrix image uses the existing BF16 pre-tiled GEMM at M256×K1152×N2560 and
emits FP32[256,2560]. `build_gemma3_vision_projector.py` builds both images
into a hashed manifest.

The [SigLIP vision embedding source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py)
uses a 14×14 stride-14 RGB Conv2d followed by learned position addition.
`gemma3_vision_patch_pack.py` lowers an 896×896 CHW BF16 image into 4096
patch rows with 588 input values padded to 640. The Conv2d weight is padded
from [1152,588] to [1280,640] and converted into the pre-tiled GEMM layout.
`build_gemma3_vision_patch_embed.py` builds an M256×K640×N1280 GEMM and a
256-row boundary image. The boundary adds the BF16 Conv2d bias to the FP32
GEMM result, rounds to BF16, adds the learned BF16 position vector, and
rounds again. Bias and position share one packed input stream to fit the
tile's two input DMA channels. A 4096-patch image needs 16 panel launches.

On 2026-10-27 the full 256-token NPU pool/norm check passed with maximum
relative error `0.00278` and `99.97%` exact BF16 elements. The composed
pool/norm-to-GEMM check passed with maximum relative matrix error
`1.01e-6` against an FP64 matrix reference. The spatial pack test checks
several corner and interior groups.

The patch embedding bundle `target/npu-dev/gemma3_vision_patch_bundle_v2`
compiled both images. A nonzero 256-patch panel starting at patch 256 passed
the NPU check: GEMM maximum relative error `6.44e-7` against the FP64
source-pixel/weight reference, boundary result 100% exact BF16 against its
FP32 NPU input, and composed output maximum relative error `0.00164`.
`test_gemma3_vision_patch_pack.py` checks RGB/channel/patch ordering and
weight padding.

Remaining work includes the SigLIP transformer layers, real FLM tensor-layout
conversion for patch and projector weights, variant-specific geometry checks,
and image-to-text output acceptance. These 4B entries remain
text-only recipe routes in the coverage matrix until the full vision path is
validated.
