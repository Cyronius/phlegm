# GPT-OSS top-four expert mixing

The router emits four expert IDs at bytes 0..15 and four FP32 weights at
bytes 16..31. The mixer accepts the selected experts' 2880-value FP32 down
projection outputs in ID order, packed as `[45 blocks, 4 experts, 64 rows]`,
and emits their weighted sum. It is a standalone NPU image.

From the Phlegm root, after sourcing `target/npu-dev/env.sh`:

```bash
dibs run build --label 'Build GPT-OSS top-four mixer' --eta 10m -- \
  python open_kernels/build_design.py open_kernels/designs/gptoss_mix/gptoss_mix.py \
  target/npu-dev/gptoss-mix-build
dibs run npu --label 'Check GPT-OSS top-four mixer' --eta 10m -- \
  python open_kernels/designs/gptoss_mix/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --router-build target/npu-dev/gptoss-router-build \
  --mixer-build target/npu-dev/gptoss-mix-build \
  --out target/npu-dev/gptoss-mix-check
```

The check runs the compiled router and mixer on four cases, including stable
ties. Its maximum relative error against a FP64 weighted sum was 1.15e-7.

`check_moe.py` runs one token through the compiled router, four expert gate/up
and down projections, full-vector activation, and mixer. It compares the final
2880-value output against a FP64 reference with the BF16 activation boundary:

```bash
dibs run npu --label 'Check GPT-OSS MoE arithmetic' --eta 20m -- \
  python open_kernels/designs/gptoss_mix/check_moe.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --router-build target/npu-dev/gptoss-router-build \
  --projection-build target/npu-dev/gptoss-expert-gemv-b90 \
  --gate-build target/npu-dev/gptoss-gate-b45 \
  --mixer-build target/npu-dev/gptoss-mix-build \
  --out target/npu-dev/gptoss-moe-check
```

Pass `--residual-build target/npu-dev/gptoss-residual-build` to also run the
FFN residual addition with zero bias after the mixer.

The 2026-10-27 NPU run selected experts 27, 23, 29 and 17. The final output's
maximum relative error was 1.9471e-6 and cosine was 0.999999999998659. This
checks one-token MoE arithmetic across separate images. Packing a model,
constructing a layer recipe, attention integration and runtime dispatch remain.
With the residual image included, the final residual output passed with
maximum relative error 1.9414e-6.
