# Gemma 4 text per-head RMSNorm

This NPU2 design normalizes BF16 head rows using epsilon `1e-6`, then
multiplies by one BF16 scale vector shared across rows. Query and key use
their learned scale vectors. Value uses a vector of ones, matching the
scale-free value norm. Output is BF16 and can feed `gemma4_rope` for query
and key or an attention value cache for value.

Set `PHLEGM_HEADS` to 1, 2, 8, or 16 and `PHLEGM_HEAD_DIM` to 256 or 512
before running `open_kernels/build_design.py`. The input and output each
contain `[heads, head_dim]` BF16 values. The scale contains `head_dim`
BF16 values. A projection's output must be rounded to BF16 before this
kernel, as Gemma 4 normalizes BF16 projected states.

```bash
source target/npu-dev/env.sh
PHLEGM_HEADS=8 PHLEGM_HEAD_DIM=256 \
  dibs run build --label 'Build Gemma 4 head norm' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gemma4_head_norm/gemma4_head_norm.py \
  target/npu-dev/gemma4-head-norm-8-256
dibs run npu --label 'Check Gemma 4 head norm' --eta 5m -- \
  python open_kernels/designs/gemma4_head_norm/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gemma4-head-norm-8-256 \
  --out target/npu-dev/gemma4-head-norm-8-256-check \
  --heads 8 --head-dim 256
```

On 2026-10-27, all eight head shapes passed NPU checks with learned and
unit scales (16 runs total). These are synthetic arithmetic checks; the
Gemma 4 layer and its source weights still need integration.
