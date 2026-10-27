# GPT-OSS 20B YaRN RoPE

This NPU source rotates the 64 query and eight key heads of one decode token.
It consumes BF16 `[72,64]` in Q-then-K order and an FP32 table of 32 cosine
values followed by 32 sine values. The table comes from
`ModelSpec.rope_inv_freq()` and `rope_scale()`, including YaRN's attention
factor. Output is BF16 `[72,64]` for the sink-attention input and K cache.
Projection with per-channel bias and cache wiring are separate steps.

From the Phlegm root, with the NPU toolchain installed:

```bash
source target/npu-dev/env.sh
dibs run build --label 'Build GPT-OSS RoPE' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_rope/gptoss_rope.py target/npu-dev/gptoss-rope
for pos in 0 127 4096 65535; do
  dibs run npu --label 'Check GPT-OSS RoPE' --eta 10m -- \
    python open_kernels/designs/gptoss_rope/check.py \
    --driver target/npu-dev/harness-build/run_kernel \
    --build target/npu-dev/gptoss-rope \
    --out "target/npu-dev/gptoss-rope-check-$pos" --pos "$pos"
done
```

On 2026-10-27, positions 0, 127, 4096 and 65535 matched the BF16-rounded
FP64 replica exactly on NPU. The first scalar BF16 cast rounded downward on
this toolchain; the kernel now explicitly applies round-to-nearest-even before
storing each BF16 result.
