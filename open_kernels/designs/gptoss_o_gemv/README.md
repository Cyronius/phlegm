# GPT-OSS attention output projection

This NPU source applies the GPT-OSS 20B `o_proj` q4_1 matrix to one BF16
`[4096]` attention result. The converter stores 96 bands of 32 rows in
four-band supertiles: 3072 rows total, of which the first 2880 are the real
hidden output. Six 16-band calls produce the full padded output. The per-row
BF16 output bias and FP32 residual addition are handled by `gptoss_residual`.

`pack.projection` selects the correct 32 x 128 half-width chunks from the
container. The checker compares each NPU result with the q4nx decoder's
FP64 matrix product on synthetic quantized chunks in the real layout.

```bash
source target/npu-dev/env.sh
dibs run build --label 'Build GPT-OSS output projection' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_o_gemv/gptoss_o_gemv.py target/npu-dev/gptoss-o-b16
for band in 0 16 32 48 64 80; do
  dibs run npu --label 'Check GPT-OSS output projection' --eta 10m -- \
    python open_kernels/designs/gptoss_o_gemv/check.py \
    --driver target/npu-dev/harness-build/run_kernel \
    --build target/npu-dev/gptoss-o-b16 \
    --out "target/npu-dev/gptoss-o-check-$band" --start-band "$band"
done
```

All six slices passed on 2026-10-27. Worst maximum relative error was
1.764e-6 and the minimum cosine exceeded 0.9999999999996.
