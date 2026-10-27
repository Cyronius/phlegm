# GPT-OSS output bias and residual

This NPU image computes `residual + (projection + bias)` for all 2880 hidden
values. The projection and residual are FP32; the output projection bias is
BF16. Input `pair` is `[45 blocks, 2 vectors, 64 values]`, with residual first
and projection second in each block. Bias is a separate 2880-value BF16 input.
The two input streams fit the NPU tile's DMA channel limit. Supplying zero bias
also supports the FFN residual after the top-four MoE mixer.

From the Phlegm root:

```bash
source target/npu-dev/env.sh
dibs run build --label 'Build GPT-OSS output residual' --eta 10m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_residual/gptoss_residual.py \
  target/npu-dev/gptoss-residual-build
dibs run npu --label 'Check GPT-OSS output residual' --eta 10m -- \
  python open_kernels/designs/gptoss_residual/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gptoss-residual-build \
  --out target/npu-dev/gptoss-residual-check
```

On 2026-10-27 all five NPU cases passed, including a cancellation case that
checks addition order and a zero-bias case. The maximum absolute error was
9.537e-7. The full top-four MoE arithmetic check also passed with this image
as its final FFN residual stage; maximum relative error was 1.9414e-6.
This does not integrate the kernel into a GPT-OSS layer recipe.
