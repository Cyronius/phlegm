# GPT-OSS RMSNorm

This NPU source computes one 2880-value GPT-OSS RMSNorm with epsilon 1e-5,
using a FP32 residual and BF16 norm weight. Output is BF16 for the q4_1
projection kernels. It serves both input and post-attention norms; the caller
selects the appropriate weight vector.

The checker compares random, all-zero and offset residuals with the FP64
`replica_dense.rms` reference after BF16 rounding.

```bash
source target/npu-dev/env.sh
dibs run build --label 'Build GPT-OSS RMSNorm' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_norm/gptoss_norm.py target/npu-dev/gptoss-norm
for case in random zero offset; do
  dibs run npu --label 'Check GPT-OSS RMSNorm' --eta 10m -- \
    python open_kernels/designs/gptoss_norm/check.py \
    --driver target/npu-dev/harness-build/run_kernel \
    --build target/npu-dev/gptoss-norm \
    --out "target/npu-dev/gptoss-norm-$case" --case "$case"
done
```

All three cases matched the BF16-rounded reference exactly on NPU on
2026-10-27.
