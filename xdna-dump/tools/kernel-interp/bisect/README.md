# Bisecting the CPU replica against FLM's captures (2026-09-02)

These scripts found why the CPU reference model (`full_forward.py` /
`decode_step.py`) "diverged" from FLM at 0.57–0.68: **FLM's captured execution
of the 3-layer `[L,L,F]` test model contributes nothing from the full-attention
block.** The replica is the faithful (HF) math; the open NPU kernels
(`tools/open-kernels`) reproduce it at corr 1.00000.

All scripts run in WSL from `tools/kernel-interp` (they import
`full_forward` / `decode_step`, which load the 3LiF model from
`MODEL_Q4NX`, default `~/.flm/models/Qwen3.6-35B-A3B-NPU2/model_3LiF.q4nx`,
and `prompt_token_ids.npy`):

```
cd /mnt/c/code/phlegm/tools/kernel-interp
python op_table.py C:/caps/m0c > bisect/m0c_ops.txt     # op table used by two scripts
python bisect/<script>.py
```

| script | what it showed |
|---|---|
| `which_token.py` | the captured decode input act (`000904.bo`) is exactly `embed(248068)` with `model.norm.weight` at +4096; the boundary states are the prefill's (hash-identical). The replica's inputs are right. |
| `layer_boundaries.py` | FLM's captured layer inputs are the **normalized** activations. Replica vs capture: layer 0 input exact; after L0 0.996–0.9997/token; after L1 0.997–0.9998; final residual after L2 **0.72** → layer 2 (full attention) diverges. |
| `l2_attention_inputs.py` | inside layer 2: q/gate/v projections and the normalized+rotated k match FLM's captures / its CPU-built KV cache at 0.9994–0.9999. Pins RoPE = half-split pairs (i, i+32), rotary 64, θ=1e7; norm weights as stored (effective). |
| `l2_expert_batches.py` | FLM's expert ops take per-expert token batches; L1's batches match the replica's MoE input at 0.993–0.9995, L2's only at 0.37–0.66. Router/post-norm weights in the captured packs equal the file's. |
| `l2_attention_scale.py` | scaling the attention contribution: **at 0.0 the L2 expert batches match at 0.995/0.9995** — FLM's MoE input is the post-norm of the residual with no attention added. |
| `skip_attention.py` | end to end: replica decode logits vs FLM's captured logits 0.671 with attention → **0.998 with layer-2 attention zeroed, same argmax**; prefill final hidden 0.72 → 0.995. |
| `control_40l.py` | control on the base 40-layer (interval-4) model: replica prefill logits vs `C:/caps/pf_t11_full/008566.bo`. |

Variants that did **not** explain it (all worse than baseline): no gate, SiLU
gate, non-causal, kv-head map h%2, other scales, no/(1+w) q norm, interleaved
per-head q|gate rows, swapped q/gate, alternative RoPE conventions, other
token/position choices for the decode step.

Consistent with the earlier note that FLM "mis-executes interval-3" models
(Josh's pruned 27B has `full_attention_interval=3`): FLM's captures are not a
valid oracle for such models, and its fused kernels are not a valid engine for
them.
