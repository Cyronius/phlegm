# GPT-OSS MXFP4 chunk decode

This standalone NPU2 kernel reads one 2560-byte GPT-OSS expert chunk and
produces 32 × 128 row-major fp32 weights. It follows
`open_kernels/model/mxfp4.py`: E8M0 scales, E2M1 nibble levels, and the
container's row-pair/column-pair nibble order. Biases embedded in padding
bytes 128–191 are separate and are not added here. This is a decode primitive,
not a complete expert GEMV or GPT-OSS runtime route.

From the Phlegm root:

```bash
source target/npu-dev/env.sh
dibs run build --label 'Build GPT-OSS MXFP4 decoder' --eta 8m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_mxfp4/gptoss_mxfp4.py \
  target/npu-dev/gptoss-mxfp4-build
dibs run npu --label 'Check GPT-OSS MXFP4 decoder' --eta 4m -- \
  python open_kernels/designs/gptoss_mxfp4/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gptoss-mxfp4-build \
  --out target/npu-dev/gptoss-mxfp4-check
```

On 2026-10-26, the NPU check matched the Python container decoder bit for
bit across three random chunks and an edge chunk covering every nibble code,
both row parities, subnormal E8M0 scales, and the highest finite test scale.
The report is `target/npu-dev/gptoss-mxfp4-check/report.json`.
