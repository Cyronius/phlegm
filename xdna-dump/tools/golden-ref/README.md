# Golden reference — what interval-3 *should* compute

`reference.py` runs the real **Qwen/Qwen3.6-35B-A3B** (the Qwen3.5-MoE hybrid:
gated-DeltaNet linear layers + periodic full-attention + MoE) through the
official `transformers` forward, sliced to the **exact layer configs the FLM
engine mis-executes**. It is the arithmetic oracle: the correct logits and
per-layer activation magnitudes for the interval-3 slices the NPU engine turns
to NaN.

## Run (needs a CUDA GPU ~24 GB; built on a RunPod RTX PRO 4000 Blackwell)

```bash
pip install --break-system-packages -U transformers accelerate safetensors huggingface_hub
python reference.py          # downloads the model, prints per-variant results,
                             # writes reference_results.json
```

Slices (over original layer indices, matching `../seq-capture/slice_keep.py`):
`3LiF=[0,1,3]`, `4Li3=[0,1,3,4]`, `5Li3=[0,1,3,4,5]`. The script loads the full
model once (bf16, CPU), moves only the needed layers + embed/norm/lm_head to GPU,
reassigns `decoder.layers` per variant, and reads last-position logits.

## Result (2026-08-28) — the overflow is the engine's, not the architecture's

All three interval-3 slices are **finite and well-scaled** in the reference; the
per-layer hidden-state magnitude barely grows across the full-attention→DeltaNet
chain:

| variant | layer_types | ref per-layer hid absmax | ref logits absmax | **engine** |
|---|---|---|---|---|
| 3LiF | `[L,L,F]` | 1.26 → 2.63 → 3.92 | 8.4 (finite) | ~10.6 finite |
| 4Li3 | `[L,L,F,L]` | …3.92 → 4.03 | 8.7 (finite) | **3.4e38** (fp32 edge) |
| 5Li3 | `[L,L,F,L,L]` | …4.03 → 4.81 | 10.2 (finite) | **NaN** |

Correct math keeps activations ~O(4–5) per layer through exactly the chain where
the engine explodes by ~10³⁷ per DeltaNet layer and overflows to NaN. This is
decisive: **the interval-3 architecture is numerically fine; the closed engine's
DeltaNet-after-interval-3-full-attention scaling is broken.** These logits +
per-layer norms are the spec for a Tier-1 host-side replacement (which reuses the
working xclbin kernels). See `../seq-capture/README.md` and the plan doc.
