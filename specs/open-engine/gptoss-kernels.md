# GPT-OSS 20B NPU kernel source progress

Both `gpt-oss:20b` and `gpt-oss-sg:20b` remain `missing_recipe` in the pinned
46-entry FLM matrix. Phlegm has open NPU primitives for Q/K/V and output
projection, RoPE, attention state and BF16 finalization, router, expert MXFP4
decoding and GEMV, gating, normalization, residual, and q4 LM head. These include
synthetic NPU checks, including a one-token layer composition. The long-context
finalizer takes the 64×68 FP32 online-softmax state and produces the 4096 BF16
values consumed by the output GEMV. It passed exact BF16 checks against both
synthetic state and a saved 4097-row attention result. A 4097-row attention to
output-projection and residual composition using this NPU handoff passed with
maximum relative residual error `4.71e-4` on 2026-10-27. This is kernel
coverage, not a deployable package or model-output acceptance.

`build_gptoss_kernels.py` compiled 13 GPT-OSS images into a hashed manifest:
input norm, q4 Q/K/V and output projections, YaRN RoPE, sliding and carried
full attention, BF16 state finalization, residual addition, top-4 routing,
MXFP4 expert projection, expert gate and mix, and the full 201,088-logit q4
LM head. The bundle's 4097-row full-attention-to-residual composition passed
on NPU, and its full LM head passed against an FP64 decoded-weight reference
with maximum relative error `5.69e-6`.
The complete synthetic one-token layer, using the bundle's full-attention,
finalizer, norm, router, four experts, mixer and residual images at position
4096, also passed against the FP64 layer reference with maximum relative error
`3.07e-7`. This checks image composition; it does not use public model weights.

`recipes/gptoss_source.py` validates the complete 435-tensor public GPT-OSS
20B q4nx header: all 24 layers, both router layouts, the expert payloads,
attention projections, embedding, final norm, and LM head. The validator
checks dtype, geometry, exact byte length, and disjoint extents. The source
adapter reads only one embedding row, projection call, expert, bias, or head
band at a time; the complete expert payload is not loaded to export one call.
The public `Atomic-Germ/GPT-OSS-20B-NPU2` and
`FastFlowLM/GPT-OSS-Safeguard-20b-NPU2` headers passed all 435 checks on
2026-10-27. Their required tensor names, dtypes, and shapes match; the
validator read bounded headers and no weight payloads.

The source adapter now builds a complete expert GEMV input by inserting the
selected expert's 2880 BF16 bias values into bytes 128–191 of each band's
first MXFP4 chunk. The layer exporter records both the raw projection and
the bias-filled NPU stream. A 90-band NPU projection using this source stream
passed an independent FP64 decoded-MXFP4 reference with maximum relative
error `6.45e-7`. The check used fixture weights and nonzero fixture bias;
real model payload arithmetic is still pending.

The source adapter also packs each 16-band Q/K/V call as BF16[2880]
activation, 64 zero padding values, and the selected 512 BF16 bias values.
This stream passed the 16-band Q NPU image at maximum relative error
`9.86e-7` against the decoded q4 reference. It packs the router activation
with the layer's 32 BF16 router biases; the NPU top-four check selected
experts `[17, 9, 3, 26]` and reached maximum logit error `5.33e-6`.
`gptoss_qkv_run.py` now schedules all ten 16-band Q/K/V image calls for one
layer from the source adapter, then writes the 4608-channel BF16 Q/K stream
for RoPE and the 512-channel BF16 V stream. Its complete ten-call NPU fixture
check passed an independently decoded q4 reference with maximum relative
error `1.11e-8`. It has not yet run with a public model weight payload.
`gptoss_o_run.py` takes the BF16[4096] attention result, schedules all six
output projection bands, discards the final 192 padded channels, and passes
the 2880 projected values, source bias, and residual through the NPU residual
image. Its seven-call fixture check matched an independent decoded q4
projection and FP32 residual reference at maximum relative error `1.68e-6`.
`gptoss_attention_run.py` now composes the source QKV runner with YaRN RoPE,
carried sink attention, and NPU BF16 finalization for one token. It validates
the supplied GPT-OSS config against the fixed image geometry and derives full
versus sliding mode from the layer schedule. Fixture checks passed all 13 NPU
calls for layer 1 at position 0 (full, exact BF16 attention) and layer 0 at
position 128 (sliding, maximum relative error `1.09e-6`). The new key and
value matched the independent BF16 reference exactly in both cases. The
published FLM GPT-OSS 20B config uses this alternating schedule and geometry:
<https://huggingface.co/FastFlowLM/GPT-OSS-20B-NPU2/blob/main/config.json>.
These checks use fixture activations and weights to verify the source/kernel
ABI. The per-layer cache and 24-layer token schedule still need to consume
these inputs with actual model weights.

Remaining model work includes a complete layer and token scheduler, real
container weight export and NPU arithmetic checks, and model-output
acceptance. Keep both registry entries at `missing_recipe` until those checks
pass. Matching header geometry does not imply the variants share weight
contents or output behavior.
