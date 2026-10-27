# GPT-OSS MXFP4 expert projection

This source computes one GPT-OSS 20B expert gate, up or down projection from
2880 BF16 inputs to 2880 FP32 outputs. It streams 90 bands of 32 output rows,
each across 23 MXFP4 input chunks. The input has 64 zero padded lanes, giving
2944 columns. The column-block-zero chunk holds the BF16 bias; the kernel adds
it after the dot product.

`pack.py` extracts one role from the fused container's expert bytes. It uses
the checked `recipes.pack.expert_chunks` mapping, including alternating gate
and up slabs and the four-row-block supertile raster. The kernel consumes
`[band, input_chunk, 2560-byte chunk]` order. The output is row-major FP32.

From the Phlegm root:

```bash
source target/npu-dev/env.sh
GEG_BANDS=90 dibs run build --label 'Build GPT-OSS expert projection' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gptoss_expert_gemv/gptoss_expert_gemv.py \
  target/npu-dev/gptoss-expert-gemv-b90
dibs run npu --label 'Check GPT-OSS expert projection' --eta 12m -- \
  python open_kernels/designs/gptoss_expert_gemv/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gptoss-expert-gemv-b90 \
  --out target/npu-dev/gptoss-expert-gemv-check-b90 --bands 90
```

The same design can build with `GEG_BANDS=1` for a short diagnostic. This
single-core source prioritizes arithmetic coverage. Expert selection, gate
activation, layer composition and model dispatch remain separate.

The activation design also has a full-vector build with
`GPTOSS_GATE_BLOCKS=45`. With both full images built, the one-expert check runs
gate projection, up projection, clamped activation and down projection in
sequence on the NPU:

```bash
GPTOSS_GATE_BLOCKS=45 dibs run build --label 'Build GPT-OSS expert activation' --eta 10m -- \
  python open_kernels/build_design.py open_kernels/designs/gptoss_gate/gptoss_gate.py \
  target/npu-dev/gptoss-gate-b45
dibs run npu --label 'Check one GPT-OSS expert FFN' --eta 12m -- \
  python open_kernels/designs/gptoss_expert_gemv/check_expert.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --projection-build target/npu-dev/gptoss-expert-gemv-b90 \
  --gate-build target/npu-dev/gptoss-gate-b45 \
  --out target/npu-dev/gptoss-expert-ffn-check
```

On 2026-10-27 the full 90-band NPU image passed against FP64 products of
independently decoded chunks plus BF16 bias. Eight chunks were copied from a
checked-in GPT-OSS container fixture into the five-megabyte test stream. The
maximum absolute error was 0.0001373291, maximum relative error was
6.4429e-7, and cosine was 0.9999999999999815. The packing tests passed 13
cases including the existing container slab-order fixture. The compiler's
core ELF `.text` section was 4384 bytes. Results and generated images remain
under ignored `target/npu-dev/gptoss-expert-gemv-*`.

The composed one-expert NPU check also passed. Relative to the FP64 decoded
MXFP4 reference with a BF16 activation boundary, the stage maximum relative
errors were 6.45e-7 (gate projection), 3.59e-7 (up projection), 6.98e-5
(activation), and 3.25e-6 (down projection). It does not select or mix the four
routed experts and does not establish GPT-OSS model support.
