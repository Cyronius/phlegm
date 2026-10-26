# GPT-OSS 20B router

This standalone design computes a 2880-input × 32-expert BF16 router on NPU2,
adds the learned BF16 bias, chooses four experts with stable ties, and
normalizes their top-four weights. Weight layout is row-major `[2880,32]`.
The first input buffer is `[x bf16[2880] | bias bf16[32]]` so the tile needs
only two input DMA streams.
The 256-byte output starts with `int32 ids[4]`, then `float weights[4]` at
byte 16, and `float logits[32]` at byte 32. Expert execution and model
dispatch remain separate.

From the Phlegm root:

```bash
source target/npu-dev/env.sh
dibs run build --label 'Build GPT-OSS router' --eta 8m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_router/gptoss_router.py \
  target/npu-dev/gptoss-router-build
dibs run npu --label 'Check GPT-OSS router' --eta 4m -- \
  python open_kernels/designs/gptoss_router/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gptoss-router-build \
  --out target/npu-dev/gptoss-router-check
```

The 2026-10-26 NPU check passed four cases against an FP64 biased projection
and stable top-four softmax: three random inputs (one with bias-driven expert
selection) and an exact tie. All expert IDs matched. Maximum logit error was
5.33e-6 and maximum weight error was 6.91e-7.
