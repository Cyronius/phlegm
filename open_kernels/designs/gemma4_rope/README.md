# Gemma 4 Q/K RoPE

This NPU2 design rotates one decode token's BF16 query and key heads. It
accepts an FP32 table containing BF16-rounded cosine values followed by
BF16-rounded sine values for half-split head pairs. The 512-wide
full-attention table implements
proportional RoPE: 64 frequency pairs are active and 192 have cosine 1 and
sine 0. Local attention uses all 128 pairs of a 256-wide head.

Build one image per text attention geometry with `PHLEGM_HEADS` and
`PHLEGM_HEAD_DIM`:

| Text variant | Local Q+K | Full Q+K |
| --- | --- | --- |
| E2B | 9 × 256 | 9 × 512 |
| E4B | 10 × 256 | 10 × 512 |
| 12B | 24 × 256 | 17 × 512 |

From the Phlegm root:

```bash
source target/npu-dev/env.sh
PHLEGM_HEADS=9 PHLEGM_HEAD_DIM=256 \
  dibs run build --label 'Build Gemma 4 RoPE' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gemma4_rope/gemma4_rope.py \
  target/npu-dev/gemma4-rope-e2b-local
dibs run npu --label 'Check Gemma 4 RoPE' --eta 5m -- \
  python open_kernels/designs/gemma4_rope/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gemma4-rope-e2b-local \
  --out target/npu-dev/gemma4-rope-e2b-local-check-4096 \
  --geometry e2b-local --pos 4096
```

On 2026-10-27, all six geometries passed NPU checks at positions 0, 127,
4096, and 65535. The checker verifies BF16 arithmetic against its FP64
reference and checks unchanged proportional pairs. This is a primitive; a
Gemma 4 layer recipe still needs projections, Q/K normalization, attention,
feed-forward, and model packaging.
