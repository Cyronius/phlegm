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

`check_chain.py` composes the separate QKV q4_1 projection, YaRN RoPE and
128-row sink-attention images with one shared BF16 activation. It checks all
ten Q/K/V projection calls and the attention output against quantized-weight
decode and the FP64 replica. For example:

```bash
dibs run npu --label 'Check GPT-OSS attention chain' --eta 15m -- \
  python open_kernels/designs/gptoss_attention/check_chain.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --qkv-build target/npu-dev/gptoss-qkv-b16 \
  --rope-build target/npu-dev/gptoss-rope-v3 \
  --attn-build target/npu-dev/gptoss-attn-r128-v2 \
  --out target/npu-dev/gptoss-chain-p127 --pos 127
```

The 2026-10-27 runs passed at positions 127 and 65535. Their attention
maximum relative errors were 8.396e-4 and 2.011e-7 respectively, with
cosines above 0.99999998. The fixture uses synthetic quantized weights in
the real container layout and an artificial 128-row cache; it is a composed
arithmetic check, not model acceptance.

`check_sublayer.py` extends the chain through six output projection calls,
BF16 o_proj bias and FP32 residual addition. The composed NPU check passed
at positions 127 and 65535 with sublayer maximum relative errors 3.331e-4
and 2.265e-6, and cosines above 0.99999997. It still uses synthetic weights
and a pre-normalized activation; a packaged GPT-OSS layer needs its norms,
MoE composition, cache and runtime wiring.

`check_layer.py` adds the input and post-attention RMSNorm and feeds the
attention residual into the existing top-four MXFP4 expert chain. The full
synthetic one-token layer passed on NPU at positions 127 and 65535. Maximum
relative errors were 2.783e-3 and 2.658e-7, with cosines 0.99999748 and
0.99999999999998. It checks arithmetic across separate compiled images;
packing a real layer, managing persistent KV state and dispatching a model
remain open.
