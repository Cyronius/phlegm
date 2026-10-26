# GPT-OSS expert gate

`gptoss_gate.py` is a standalone NPU design for 64 expert channels. The inputs
are fp32 gate/up projection results **including** their biases; the output is
bf16. Its arithmetic matches `model/replica_gptoss.py`:

`(clip(up, -7, 7) + 1) * min(gate, 7) * sigmoid(1.702 * min(gate, 7))`.

The gate has no lower clamp. The design uses the shared `vecmath.h` fp32
emulation and converts once to bf16 at the output. The 64-channel stripe is a
component, not a GPT-OSS model route: GPT-OSS still needs 2880-wide expert
packing and projection, 32-expert routing, biases, attention sink, a manifest
and runtime integration.

Build and check on a host with IRON, XRT and adequate memlock:

```sh
source target/npu-dev/env.sh
dibs run build --label 'Build GPT-OSS gate' --eta 8m -- \
  python open_kernels/build_design.py \
    open_kernels/designs/gptoss_gate/gptoss_gate.py target/npu-dev/gptoss-gate-build
dibs run npu --label 'Check GPT-OSS gate' --eta 4m -- \
  python open_kernels/designs/gptoss_gate/check.py \
    --driver target/npu-dev/harness-build/run_kernel \
    --build target/npu-dev/gptoss-gate-build \
    --out target/npu-dev/gptoss-gate-check
```

The numerical check includes random inputs and the asymmetric clamp edges.
It compares the device result with fp64 arithmetic rounded to bf16, using a
maximum absolute error scaled by `max(abs(reference), 1)` of 0.012. This
threshold permits the AIE exponential approximation and bf16 output rounding;
it is not a whole-model quality gate.
