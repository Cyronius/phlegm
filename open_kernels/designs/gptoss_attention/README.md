# GPT-OSS sink attention

This one-core NPU source computes GQA for one GPT-OSS 20B decode token:
64 query heads, eight KV heads, head width 64 and one learned sink logit per
query head. The sink seeds online softmax's denominator with mass one and a
zero value accumulator. The score scale is 1/8; the sink logit is unscaled.
Q and K must already have YaRN RoPE applied. The host selects the most recent
128 K/V rows for a sliding layer or all prior rows for a full layer.

`q_sink` holds BF16 Q `[64,64]` followed by FP32 sinks `[64]`. Each `kv_rows`
record holds BF16 K `[8,64]` then V `[8,64]`. Output is FP32 `[64,64]`.
The direct design specializes to 1, 128 or 256 rows. From the Phlegm root:

```bash
source target/npu-dev/env.sh
GPTOSS_ATTN_ROWS=128 dibs run build --label 'Build GPT-OSS sliding attention' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_attention/gptoss_attention.py \
  target/npu-dev/gptoss-attn-r128
dibs run npu --label 'Check GPT-OSS sliding attention' --eta 12m -- \
  python open_kernels/designs/gptoss_attention/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gptoss-attn-r128 \
  --out target/npu-dev/gptoss-attn-check-r128 --rows 128
```

On 2026-10-27, the 1-row, sliding-128 and full-256 images passed on NPU
against the checked FP64 GPT-OSS replica. Their maximum relative errors were
7.63e-8, 5.57e-7 and 6.87e-7. A direct 4096-row image compiled but its NPU
run failed after about two seconds; longer contexts use the bounded state
kernel in `../gptoss_attention_state/`.
