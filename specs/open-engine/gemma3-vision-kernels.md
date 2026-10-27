# Gemma 3 4B vision NPU kernels

The four Gemma 3-family 4B entries in the pinned FLM registry still need a
validated vision-tower route. The NPU images now cover patch embedding, a
complete SigLIP encoder layer, and the projector. The Google/Hugging
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

The [SigLIP encoder source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py)
uses LayerNorm, Q/K/V and output projections, then a two-matrix MLP within
each transformer layer. The 4B vision geometry used here is 1152 channels,
16 heads, 27 layers, and 4304 MLP channels; these dimensions also appear in a
[public repack's configuration](https://huggingface.co/onnx-community/gemma-3-4b-it-ONNX/blob/main/config.json).
`build_gemma3_vision_tower_mm.py` builds a hashed five-image bundle:
LayerNorm[256,1152], QKV M256×K1152×N3584, attention output
M256×K1152×N1280, FC1 M256×K1152×N4352, and FC2 M256×K4352×N1280.
`gemma3_vision_tower_weights.py` pads logical matrices and concatenates Q/K/V
in that order. This bundle is in `target/npu-dev/gemma3_vision_tower_mm_bundle_v2`.

The NPU LayerNorm check passed across 256 rows including a constant row,
with maximum relative error `0.00140` and `99.98%` exact BF16 values. All
four matrix images passed sampled FP64 checks across first, interior and last
rows and first/last columns; their maximum sampled relative errors were
`4.72e-7`, `5.65e-7`, `6.04e-7`, and `9.91e-7` respectively. Padded output
columns were zero. The matrix source remains synthetic.

`build_gemma3_vision_post_mm.py` builds four BF16 bias boundary images. QKV
uses 3456 logical channels; attention output and FC2 use 1152; FC1 uses
4304 and applies SigLIP's tanh GELU after the BF16 bias round. Their NPU
checks passed with maximum relative errors `2.05e-5`, `0`, `5.09e-6`, and
`0` respectively against BF16 references, with zeroed padded channels.
The earlier 1152-channel two-addend image also serves attention and MLP
residual boundaries when packed with bias and residual instead of bias and
position; this NPU check matched all BF16 outputs exactly.

`build_gemma3_vision_attention.py` adds three images for SigLIP's global,
bidirectional attention: Q×Kᵀ at M256×K128×N4096, 4096-key scaled BF16
softmax, and P×V at M256×K4096×N256. `gemma3_vision_attention_pack.py`
extracts the 72 channels of a selected Q/K/V head and pads its matrix
dimensions. The scale is `1/sqrt(72)`; attention is over every one of the
4096 patches. A whole image requires 16 heads and 16 query panels per head.
On NPU, head 5 and query rows 256–511 passed against a full FP64/BF16 CPU
reference: QK maximum relative error `1.27e-7`, 99.9888% softmax BF16
matches with maximum absolute error `3.81e-6`, PV maximum relative error
`1.47e-6`, and composed output maximum relative error `0.000182`.
The 184 padded output channels were zero.

`gemma3_vision_attention_run.py` schedules all 16 heads and 16 query panels
from a BF16[4096,3584] QKV input. The full synthetic 4096-query run completed
768 NPU kernel dispatches. Nine sampled head/row CPU comparisons passed, with
maximum relative error `0.00347`. It produces an interleaved
BF16[4096,1152] attention output for the layer's output projection.

`gemma3_vision_layer_run.py` composes one complete 4096-token SigLIP encoder
layer with synthetic BF16 weights: both LayerNorms, QKV, all attention panels,
output projection and residual, FC1/GELU, FC2 and residual. NPU attention
samples passed with maximum relative error `0.00132`; three downstream
row checks using captured attention matched the BF16 CPU reconstruction
exactly. This establishes the kernel schedule for one layer, not a real
checkpoint result.

`gemma3_vision_source.py` reads the BF16 tensors from FLM's
`vision_weight.q4nx`, validates the 4B shapes, and undoes FLM's 64×256
matrix tiling. It exposes patch embedding, all 27 encoder layers, final
post-LayerNorm, and projector weights. The layer runner accepts
`--vision-weights`, `--input` (BF16[4096,1152] patch embeddings), and
`--layer-index` for a real-weight layer check. The tile inverse and padding
rejection have CPU tests.

`gemma3_vision_tower_run.py` now schedules the 16 patch panels, 27 encoder
layers, 16 post-LayerNorm panels, one 4×4 spatial pool/norm image, and one
projector matrix image. It alternates two hidden-state files and reuses one
layer scratch directory. Input is FLM's preprocessed BF16 CHW[3,896,896]
image, `vision_weight.q4nx`, and its `config.json`. Output is a
BF16[256,2560] projected image-token file alongside the
FP32 projector output. The 27-layer chaining test checks every layer index
and the final input to the projector.

The runner checks `config.json` against 17 vision geometry fields, all 439
required q4nx tensor headers and byte extents, and all 17 NPU image paths
before dispatch. The published FLM configs for
[Gemma3 4B](https://huggingface.co/FastFlowLM/Gemma3-4B-NPU2/blob/a5ad9dbc7bdf9cdf83b4ddfd951372953da0969a/config.json),
[Translategemma 4B](https://huggingface.co/Atomic-Germ/Translategemma-4B-Instruct-NPU2/blob/main/config.json),
[Medgemma 4B](https://huggingface.co/OpenFlowLM/medgemma-4b-it-NPU2/blob/main/config.json),
and [Medgemma 1.5 4B](https://huggingface.co/OpenFlowLM/medgemma-1.5-4b-it-NPU2/blob/main/config.json)
passed the same geometry check. The public
Gemma3 4B `vision_weight.q4nx` at FastFlowLM revision
`a5ad9dbc7bdf9cdf83b4ddfd951372953da0969a` has SHA-256
`2ad25d01edb6b31bab7e487e496f4652cc098738d8a726d69854fb5408d351e6`.
Its 439 required tensor headers passed, and all 27 layers decoded with zero
padding and finite values. The other three variants' weight payloads have
not been checked.

For the new full-image path, a synthetic 4096-patch NPU run passed FP64/BF16
checks at patches 0, 257, 2047, and 4095 with exact BF16 results. The
synthetic final post-LayerNorm, pool/norm and projector path also passed NPU
checks: sampled post-LayerNorm rows were exact BF16, pool/norm maximum
relative error was `0.003125`, and sampled projector matrix maximum relative
error was `6.62e-7`. These checks establish the endpoints of the scheduled
tower; they do not validate propagation through a real 27-layer checkpoint.

With the public Gemma3 4B FLM weights and a deterministic normalized RGB
image, all 4096 patch embeddings ran on NPU; four sampled patches matched the
FP64/BF16 CPU calculation exactly. Encoder layer 0 completed all 16 heads and
both residual paths. Three sampled QKV rows reached maximum relative error
`7.58e-5`, all 16 heads at three rows reached `0.00129`, and three sampled
layer-output rows reached `0.000481`. The real final post-LayerNorm,
pool/norm, and projector weights also passed NPU arithmetic checks when fed
that layer-0 output: sampled LayerNorm rows were exact BF16, pool/norm maximum
relative error was `0.000635`, and sampled projector matrix maximum relative
error was `3.90e-7`. This last probe uses layer-0 output as input to the
final stages; it does not represent the 27-layer model output.

Remaining work includes real-weight 27-layer execution, the other three
variants' weight checks, and image-to-text output acceptance. These 4B entries remain
text-only recipe routes in the coverage matrix until the full vision path is
validated.
