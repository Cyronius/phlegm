# Gemma 4 text decode attention

This NPU2 design computes causal decode attention for one query head against
one selected K/V head. The caller supplies only allowed cache rows: the
newest 512 or 1024 rows for local attention, or consecutive chunks of the
full prefix for full attention. It applies Gemma 4's score scale of 1.0 and
uses online softmax so a bounded image can carry state through a long prefix.
Group-query head selection, cache management, and the final projection are
caller work.

Build with `PHLEGM_HEAD_DIM` 256 or 512 and `PHLEGM_ATTN_ROWS` 1, 128,
512, or 1024. Inputs are:

- `query_state`: one BF16 query head followed by a FP32 carry buffer.
- `kv_rows`: `[rows, 2, head_dim]` BF16, with K then V in each row.

The carry buffer has `2 * head_dim + 16` FP32 elements. Its first `head_dim`
elements are the weighted value accumulator. Element `head_dim` is the
online maximum and `head_dim + 1` is its denominator. The final `head_dim`
elements, starting at `head_dim + 16`, are the normalized attention output.
Initialize the entire carry buffer to zero for the first chunk. For each
later chunk, concatenate the same query head with the previous output buffer.
After the final chunk, the normalized output can be rounded to BF16 for the
output projection.

```bash
source target/npu-dev/env.sh
PHLEGM_HEAD_DIM=512 PHLEGM_ATTN_ROWS=1024 \
  dibs run build --label 'Build Gemma 4 attention' --eta 12m -- \
  python open_kernels/build_design.py \
  open_kernels/designs/gemma4_attention/gemma4_attention.py \
  target/npu-dev/gemma4-attention-512-1024
dibs run npu --label 'Check Gemma 4 attention' --eta 10m -- \
  python open_kernels/designs/gemma4_attention/check.py \
  --driver target/npu-dev/harness-build/run_kernel \
  --build target/npu-dev/gemma4-attention-512-1024 \
  --out target/npu-dev/gemma4-attention-512-1024-check \
  --head-dim 512 --rows 1024 --chunks 4
```

On 2026-10-27, all eight images passed synthetic FP64 checks. The 512-wide
image carried NPU state through four 1024-row chunks; its final output had
maximum absolute error `3.54e-7` at 4096 rows. A Gemma 4 model recipe and
real source-weight validation remain to be implemented.

`check_chain.py` checks all six E2B, E4B and 12B local/full text head layouts
through the separate NPU head-norm, RoPE and attention images. It verifies
Q/K/V norms, rotation, grouped K/V selection and 128-row attention. All six
passed on 2026-10-27. Conditioned on the actual NPU norm and RoPE outputs,
the worst attention error was `5.67e-6`. The worst end-to-end difference from
the BF16-rounded FP64 oracle was `0.00505`: one E4B local query norm value
rounded to an adjacent BF16 value and changed the scale-1 attention scores.
The chain records both diagnostics in each `report.json`.
