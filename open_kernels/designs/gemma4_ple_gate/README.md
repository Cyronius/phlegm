# Gemma 4 E2B/E4B per-layer input gate

The E2B and E4B text layers apply `gelu_pytorch_tanh` to a 256-value
projection, multiply it by the 256-value per-layer input, and project the
result back to the hidden width. This NPU2 design implements the activation
and product. Both inputs and the output are BF16 `[256]`; it rounds to BF16
after GELU and again after multiplication, matching the model's BF16 tensor
boundaries. The two linear projections and the following norm/residual still
need recipe wiring.

```bash
source target/npu-dev/env.sh
dibs run build --label 'Build Gemma 4 PLE gate' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gemma4_ple_gate/gemma4_ple_gate.py \
  target/npu-dev/gemma4-ple-gate
dibs run npu --label 'Check Gemma 4 PLE gate' --eta 5m -- \
  python open_kernels/designs/gemma4_ple_gate/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gemma4-ple-gate \
  --out target/npu-dev/gemma4-ple-gate-check
```

The 2026-10-27 synthetic NPU check matched the BF16-rounded FP64 reference
exactly across 256 values, including near-zero and saturation cases.
