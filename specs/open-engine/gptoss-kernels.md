# GPT-OSS 20B NPU kernel source progress

Both `gpt-oss:20b` and `gpt-oss-sg:20b` remain `missing_recipe` in the pinned
46-entry FLM matrix. Phlegm has open NPU primitives for Q/K/V and output
projection, RoPE, attention state, router, expert MXFP4 decoding and GEMV,
gating, normalization, residual, and q4 LM head. These primitives include
synthetic NPU checks, including a one-token layer composition. This is kernel
coverage, not a deployable package or model-output acceptance.

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
