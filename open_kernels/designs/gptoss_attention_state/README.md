# Bounded GPT-OSS full attention

This image processes 512 GPT-OSS K/V rows per NPU call and exports the
online-softmax accumulator, maximum and denominator for all 64 query heads.
The host carries that state into the next call. A 16-byte trailer on `q_state`
gives the valid-row count, so the final call may contain one to 512 real rows
and zero padding. The host seeds each head with its learned sink maximum,
denominator one and zero value accumulator, then divides accumulator by
denominator after the final call. The same source can specialize to 128 rows.

`q_state` holds BF16 Q `[64,64]`, then FP32 state `[64,68]` (64 value sums,
maximum, denominator, two padding floats), then four int32 metadata fields
with the valid count first. Each K/V row is `[K:8,64 | V:8,64]` BF16.

From the Phlegm root:

```bash
source target/npu-dev/env.sh
GPTOSS_ATTN_STATE_ROWS=512 dibs run build --label 'Build GPT-OSS attention state' --eta 15m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_attention_state/gptoss_attention_state.py \
  target/npu-dev/gptoss-attn-state-r512
dibs run npu --label 'Check GPT-OSS full attention' --eta 20m -- \
  python open_kernels/designs/gptoss_attention_state/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gptoss-attn-state-r512 \
  --out target/npu-dev/gptoss-attn-state-check-r4097 --total-rows 4097
```

The 2026-10-27 check passed 4097 rows across nine NPU calls, with one valid
row in the final chunk. Maximum relative error against the FP64 sink-attention
reference was 2.731e-6 and cosine was 0.9999999999988936. A prior 4096-row
check also passed across eight calls. This validates attention arithmetic and
state carry; QKV projection, YaRN RoPE, output projection, cache management,
layer packing and runtime dispatch remain separate.
