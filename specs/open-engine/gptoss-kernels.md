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

Remaining model work includes a complete layer and token scheduler, real
container weight export and NPU arithmetic checks, and model-output
acceptance. Keep both registry entries at `missing_recipe` until those checks
pass. Matching header geometry does not imply the variants share weight
contents or output behavior.
