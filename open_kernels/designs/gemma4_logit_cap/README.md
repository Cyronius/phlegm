# Gemma 4 final vocabulary softcap

Gemma 4 text models transform their 262144 FP32 vocabulary logits with
`30 * tanh(logit / 30)`. This NPU2 design streams 256-logit blocks through
one core. It consumes and produces a contiguous FP32 `[262144]` buffer.

```bash
source target/npu-dev/env.sh
dibs run build --label 'Build Gemma 4 logit cap' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gemma4_logit_cap/gemma4_logit_cap.py \
  target/npu-dev/gemma4-logit-cap
dibs run npu --label 'Check Gemma 4 logit cap' --eta 5m -- \
  python open_kernels/designs/gemma4_logit_cap/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gemma4-logit-cap \
  --out target/npu-dev/gemma4-logit-cap-check
```

The full-vocabulary synthetic NPU check passed on 2026-10-27 with maximum
absolute error `5.73e-6` against FP64 `tanh`, including zero and saturation
cases. A Gemma 4 LM-head exporter and model recipe still need integration.
