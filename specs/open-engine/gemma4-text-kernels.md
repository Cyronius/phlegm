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
| q4 full-vocabulary LM head | `open_kernels/designs/lm_head_q4/` and `open_kernels/gemma4_lm_head_check.py` | Full 262144-logit images passed at K=1536, 2560, 3840 |
| Final logit softcap | `open_kernels/designs/gemma4_logit_cap/` | Full 262144 logits passed against `30*tanh(logit/30)` |
| Composed head and softcap | `open_kernels/gemma4_logits_check.py` | All three text widths passed across the full vocabulary |
| E2B/E4B per-layer input GELU gate | `open_kernels/designs/gemma4_ple_gate/` | BF16-rounded 256-value NPU check matched exactly |
| Hidden RMSNorm and residual | `open_kernels/designs/gemma4_hidden_norm/` | E2B/E4B/12B widths passed with residual and plain norm modes |

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
(E4B), and `8.29e-6` (12B). After the NPU softcap, maximum absolute errors
were `4.01e-5`, `4.05e-5`, and `7.06e-5` respectively. These checks use the
generic 5120-byte q4_1 pool format; the actual Gemma 4 package's source
layout and any required transcode remain unverified.

## Remaining work for a model recipe

The three model entries need source-weight mapping and packing, text Q/K/V/O
and MLP projections, residual and sandwich-norm wiring, and a layer scheduler.
E2B and E4B also need per-layer input projection and shared-K/V routing. The 12B full
attention path needs its K=V alternative represented in the exporter and
scheduler. All three require real-container arithmetic checks, complete
generation, and model-output acceptance. Vision and audio routes for the
multimodal entries are still separate coverage gaps. Do not change their
`missing_recipe` matrix state until those pieces form a deployable recipe.

Architecture references: the published [12B configuration](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json),
[E4B configuration](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json),
and [Gemma 4 text implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py).
