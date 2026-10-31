# Gemma 4 text NPU kernel progress

The target is the three Gemma 4 entries in the pinned 46-model FLM registry
listed in `flm-kernel-matrix.md`: E2B, E4B, and 12B. The matrix continues to
mark all three `missing_recipe`; the work below supplies tested primitives and
compositions, not an exported model or native generation path.

| Operation | Phlegm source | NPU check on 2026-10-27 |
| --- | --- | --- |
| Per-head Q/K/V RMSNorm, widths 256/512 | `open_kernels/designs/gemma4_head_norm/` | Eight head shapes with learned and unit scales, 16 runs passed |
| Local and proportional full RoPE | `open_kernels/designs/gemma4_rope/` | Six Q+K layouts at positions 0, 127, 4096, 65535, 24 runs passed |
| Decode attention with online softmax state | `open_kernels/designs/gemma4_attention/` | Eight width/chunk images passed; 512-wide state carried through 4096 rows |
| Composed norm, RoPE, GQA attention | `open_kernels/designs/gemma4_attention/check_chain.py` | All six E2B/E4B/12B local/full head layouts passed at position 127 |
| q4 full-vocabulary LM head | `open_kernels/designs/lm_head_q4/` and `open_kernels/gemma4_lm_head_check.py` | Full 262144-logit images passed at K=1536, 2560, 3840; 12B's FLM source uses q8 instead |
| 12B q8 full-vocabulary LM head | `open_kernels/designs/lm_head_q8/`, `gemma4_q8_head_check.py` | Complete 262144-logit image passed at maximum relative error `4.43e-6` against independently decoded synthetic q8 chunks |
| Final logit softcap | `open_kernels/designs/gemma4_logit_cap/` | Full 262144 logits passed against `30*tanh(logit/30)` |
| Composed head and softcap | `open_kernels/gemma4_logits_check.py` | All three text widths passed across the full vocabulary |
| E2B/E4B per-layer input GELU gate | `open_kernels/designs/gemma4_ple_gate/` | BF16-rounded 256-value NPU check matched exactly |
| Hidden RMSNorm and residual | `open_kernels/designs/gemma4_hidden_norm/` | E2B/E4B/12B widths passed with residual and plain norm modes |
| Text layer scheduler plan | `open_kernels/gemma4_text_plan.py` | Published E2B/E4B/12B geometry, shared-K/V providers, K=V routing, MLP widths, and local/full sequence checked by eight CPU tests |
| q4 text Q/K/V/O and MLP projections | `open_kernels/build_gemma4_text_projections.py` | Published geometries compiled; E2B Q and split down, plus 12B split down, passed representative NPU arithmetic checks |
| E2B/E4B BF16 per-layer projections | `open_kernels/designs/gemma4_bf16_gemv/` | All four source-layout geometries compiled and passed representative NPU arithmetic checks (maximum relative error `7.27e-7`) |
| FLM text weight mapping | `open_kernels/gemma4_source.py` | Public E2B, E4B, and 12B q4nx headers passed 597, 727, and 668 required tensor shape/byte checks; 12B's signed q4 and q8 16-row tile repackers passed exact synthetic value checks |
| Bounded source reads | `open_kernels/gemma4_text_source.py` | E2B/E4B/12B public headers passed complete required-tensor and nonoverlapping byte-extent checks; synthetic reads cover striped q4 projection bands, 12B signed q4 and q8 head bands, BF16 per-layer matrices, and scaled embedding rows |
| Source-format Q projection on NPU | `open_kernels/gemma4_source_projection_check.py` | Complete layer-zero Q projection passed from E2B, E4B, and 12B source-format fixtures; maximum relative errors `5.27e-6`, `6.76e-6`, and `9.34e-6` against independent FP64 source decoders |
| 12B source-format q8 head on NPU | `open_kernels/gemma4_source_head_check.py` | One 128-logit band passed at maximum relative error `5.14e-6` against an independent decoder of FLM's 16-row q8 source tiles |
| 12B local/full decoder layer | `open_kernels/gemma4_12b_layer_run.py` | Layer 0 local and layer 5 full K=V paths passed zero-weight NPU wiring at positions 0 and 1, 162 calls total; residual and cache rows matched exactly |
| Per-layer context norm and token mix | `open_kernels/designs/gemma4_ple_input/` | E2B/E4B with-token and context-only variants compiled and passed NPU checks |
| Full-width MLP GELU-tanh and up product | `open_kernels/designs/gemma4_gelu_mul/` | All four 6144/10240/12288/15360-wide images compiled and passed NPU checks (maximum absolute error `3.05e-5`) |

The attention state image processes 1, 128, 512, or 1024 rows per call. Its
FP32 carry output supports consecutive full-attention chunks. The caller
selects the grouped K/V head and passes only allowed local-window rows. The
4096-row synthetic check reached maximum absolute output error `3.54e-7`
against FP64 attention. The six composed attention checks reached maximum
error `5.67e-6` when conditioned on the actual NPU norm/RoPE outputs. One E4B
local query norm value rounded to an adjacent BF16 value relative to the FP64
oracle; the corresponding end-to-end attention difference was `0.00505`.

The full q4 head checks used alternating synthetic quantized bands in every
vocabulary row. Maximum relative errors were `6.64e-6` (E2B), `6.27e-6`
(E4B), and `8.29e-6` (12B geometry). The actual 12B FLM head is q8, so that
third check does not validate its source format. After the NPU softcap, maximum absolute errors
were `4.01e-5`, `4.05e-5`, and `7.06e-5` respectively. These checks use the
generic 5120-byte q4_1 pool format; the actual Gemma 4 package's source
layout was checked against the public q4nx headers. The source adapter reads
one projection or vocabulary band at a time, so it does not need to map a full
matrix into a separate array. Numerical checks using actual source weight
bytes remain pending.

The text projection inventory has 310 E2B, 384 E4B, and 328 12B logical
matrices. E2B and E4B include three BF16 per-layer matrices per layer in
FLM's `[N/32,K,32]` layout: model projection, input gate, and projection
back to hidden width. This mapping follows the FLM q4nx converter and all
three actual public q4nx package headers; numeric source bytes remain unchecked.
The inventory omits K/V projections in shared layers and the full-attention V
projection in 12B. A 12288-wide E2B down projection exceeded AIE tile memory
as one call; the plan splits it into two 6144-wide calls. E4B's 10240-wide
and 12B's 15360-wide down projections similarly use two calls. The complete
12B, E2B, and E4B all-q4 projection image sets compiled from pinned configuration
fixtures on 2026-10-27. Corrected E2B and E4B text kernel bundles compiled
with BF16 PLE projection images. The corrected 12B bundle also compiled with
a q8 LM head. The representative arithmetic checks above cover source-layout
families and selected shapes, not every layer or real weight payload.

## Dense recipe route from openflowlm-next #225

The `gemma4` family in `open_kernels/recipes/gemma4.py` was imported on
2026-10-28 from openflowlm-next PR #225 (head `beccd46`). It builds the fused
`designs/dense/dx.py` layer once per layer type, for 12B only. Requirement,
upstream hardware result and open items are in `OPEN-FAMILY-GEMMA4` in
`spec.md`; import validation is in `open_kernels/UPSTREAM.md`. Upstream
measured that route on the GGUF container on Strix Point. Phlegm ran it on
Strix Halo on 2026-10-29. Greedy decode matched an independent reference, and
layer 0 confirmed the 1024-row window at positions 1023-1025. The same day, a
block prefill route (`gemma4.gemm_route`) gave the same greedy tokens at 147
instead of 325 ms per prompt token and passed the same window check. It is on
by default for the GGUF route; `DENSE_GGUF_GEMM=0` at export leaves it out. The
GGUF route also stores its weights in 5120-byte chunks by default, with the
same output bits and about 11% faster decode. Block prefill's attention then
moved to NPU matrix products per kv head: 56 instead of 134 ms per prompt
token in adjacent runs, the same greedy tokens and a lighter KL tail. Those
runs had one host thread, because the dev environment exports
`OMP_NUM_THREADS=1`. At 8 threads, with bit-identical host stages, one
hardware context for all attention products and a layer-major schedule, the
same prompt takes 22-23 ms per token with every prefill logit unchanged. On
2026-10-30 the default prefill moved its five projections to bfp16 GEMMs
(`designs/gemm_bfp16`), with the q4 route kept as `--prefill-mode lean`. In
adjacent runs under heavier load, that took 12-13 against 29 ms per prompt
token. Both greedy runs were unchanged. Next-token loss was within 0.3% of
lean's, top-1 agreement with the reference was 2.5 points lower, and the KL
distribution was about 10 times wider. Later that day each layer ran step by
step over every block, with the host's work under the next dispatch, and the
bfp16 route's GELU moved to fp32. The scores of Q's two halves also became one
product. The schedule and the stacked scores left every logit unchanged.
Fast took 9.2-10.0 s for the 1450-token prompt (6-7 ms per token) against
16.1-16.3 s on the old stages, and lean 28.4 against 37.5 s. Both gates still
pass with the fp32 GELU. Cutting four host copies, again without changing a
logit, took fast to 9.4-9.7 s. That evening the GGUF view's pools moved to a
4608-byte Q4_0-native chunk, which drops the min plane because the cores form
`m = -8 d` themselves. Every logit stayed the same and decode took 6-7% less
time, 238-240 against 255-256 ms per token. Decode is bound by the weight
stream, not by the GEMV arithmetic. The measured results and the decode
profile are in `OPEN-FAMILY-GEMMA4`.

The two implementations agree on 12B's layer types, global geometry, K=V
before separate norms, unit score scale, V norm, proportional RoPE pairing and
the whole-stream output scale. The per-operation images above can serve as
independent references when the recipe's two attention points go through
catalogue validation. Three differences matter for that work:

- The fused route keeps the residual stream in FP32 and folds the layer scale
  into the post norms. `gemma4_12b_layer_run.py` rounds the residual to BF16
  after each layer and applies the scale on the host.
- The recipe's `model.q4nx` route does not read FLM's 12B package format. The
  12B header has 16-row signed-q4 tiles (2304 B), a 16-row q8 head (4352 B)
  and a BF16 embedding table, which `gemma4_text_source.py` treats as
  unscaled. Both packers at the PR head, `pack.q4_chunks_of` and `pools.cpp`,
  read 5120, 8704 and 4736-byte chunks only. The plan also packs the head as
  q4, and its q4nx embedding entry has no sqrt(hidden) scale. `gemma4_source.py`'s
  `pair_q4_0_16row`, `pair_q8_16row` and `q8_head_pool` repack those tiles
  losslessly; they are checked on synthetic tiles, not real payload bytes.
- E2B and E4B remain refused by the recipe. Their per-layer input and
  shared-K/V pieces exist only as the primitives and plan above.

## Remaining work for a model recipe

The three model entries need real source-weight payload checks and a scheduler
that executes the validated layer plan; for 12B that is the dense route above. The 12B local/full decoder layer now
composes the sandwich norms, projections, attention, MLP and output scale; its
zero-weight two-position check covers wiring only. E2B and E4B also need the
per-layer input and shared-K/V state routing in that scheduler. The 12B full
attention path routes K=V in the layer runner, but still needs real-weight acceptance.
All three require real-container arithmetic checks, complete
generation, and model-output acceptance. Vision and audio progress is recorded
in `gemma4-multimodal-kernels.md`. Do not change their
`missing_recipe` matrix state until those pieces form a deployable recipe.

Architecture references: the published [12B configuration](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json),
[E4B configuration](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json),
and [Gemma 4 text implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py).
