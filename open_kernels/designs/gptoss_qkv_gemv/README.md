# GPT-OSS Q/K/V projection

This one-core NPU source projects 2880 BF16 input values through GPT-OSS
20B's 2560-byte q4_1 chunks and adds the per-output BF16 bias. It computes
one or 16 bands of 32 output rows per call. The container stores four-band
supertile order; `pack.projection` selects the requested contiguous bands.
The input is padded to 2944 BF16 values, followed by one 32-value BF16 bias
per output band. Output is FP32. Eight 16-band calls cover Q's 4096 outputs;
one each covers K and V's 512 outputs.

The projected Q/K values must be BF16-rounded and rotated by `gptoss_rope`
before `gptoss_attention` or `gptoss_attention_state`. The layer recipe still
needs to connect these calls, cache rows, and output projection.

From the Phlegm root:

```bash
source target/npu-dev/env.sh
GPTOSS_QKV_BANDS=16 dibs run build --label 'Build GPT-OSS QKV' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_qkv_gemv/gptoss_qkv_gemv.py \
  target/npu-dev/gptoss-qkv-b16
dibs run npu --label 'Check GPT-OSS K projection' --eta 10m -- \
  python open_kernels/designs/gptoss_qkv_gemv/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gptoss-qkv-b16 \
  --out target/npu-dev/gptoss-qkv-check-k --role k --bands 16
```

On 2026-10-27, all eight Q slices and complete K/V projections passed on
NPU against the q4nx decoder, including bias. Worst maximum relative error
among these ten calls was 1.612e-6; minimum cosine exceeded
0.9999999999997. The fixture uses synthetic quantized chunks in the real
container layout; it has not accepted a packaged model.
