# Gemma 4 text hidden RMSNorm and residual

This NPU2 design serves Gemma 4's hidden-width RMSNorm with epsilon `1e-6`
and the following residual addition. The first BF16 input buffer contains
`[value, residual]`, each of length `hidden`; the second contains the learned
BF16 scale of length `hidden`. The output is BF16 `[hidden]`. The kernel
rounds the norm result to BF16 before adding the residual, then rounds the
sum to BF16. A zero residual yields the plain norm needed at pre-attention
and pre-FFN boundaries.

Build `PHLEGM_HIDDEN=1536`, `2560`, or `3840` for E2B, E4B, or 12B:

```bash
source target/npu-dev/env.sh
PHLEGM_HIDDEN=1536 \
  dibs run build --label 'Build Gemma 4 hidden norm' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gemma4_hidden_norm/gemma4_hidden_norm.py \
  target/npu-dev/gemma4-hidden-norm-1536
dibs run npu --label 'Check Gemma 4 hidden norm' --eta 5m -- \
  python open_kernels/designs/gemma4_hidden_norm/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gemma4-hidden-norm-1536 \
  --out target/npu-dev/gemma4-hidden-norm-1536-check \
  --hidden 1536
```

The three images built on 2026-10-27. All six NPU checks passed: each width
with a residual and with a zero residual. The designs do not yet form a full
layer recipe.
