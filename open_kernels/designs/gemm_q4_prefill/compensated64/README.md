# Experimental compensated64 GGUF prefill

This design maps output features to eight physical columns and tokens to four
core rows. Each core computes64 features x64 tokens with FP32 C. It is a
standalone experiment; it does not change the production recipe or its defaults.

The supported block size is T256. N must be a positive multiple of512, K a
positive multiple of256. The Llama1B projection family is:

| Role | N | K |
|---|---:|---:|
| QKV |3072|2048|
| output |2048|2048|
| gate/up |8192|2048|
| down |2048|8192|

## Data contract

Weights are the GGUF-direct Q4 pool with6144-byte chunks, each covering32
feature rows x256 K values. Its scale/offset planes are losslessly widened
float32 values, narrowed with RNE on the core. Two chunks form a64-feature
band. This is not the compact5120-byte pool and not a weight requantization.

The dequant producer writes BFP high/residual panels. The MAC accumulates
HH, HL, LH in that order in FP32 and omits LL. This is a precision change,
requiring independent source/operand and full-model gates. It retains two row
accumulators and a rolled K loop. The source is not ArtLite implementation code.

Input X has `T*K*9/4` bytes. Its global order is
`[K64][T/8][K8][hi,lo][row8][9bytes]`. Existing `bfp16_hilo` host packing works:

```
[K64][T/32][4 token8][K8][hi,lo] == [K64][T/64][8 token8][K8][hi,lo]
```

Both flatten to the same global token8 order and preserve the8-K exponent
groups. Only DMA grouping changes. Native BF16 `tile_x` does not have this
property and cannot be used here. Host packing must raise the shared exponent
when a positive mantissa rounds to128; saturation changes input values.

Output Y is ordinary FP32 `[N,T]`. A physical feature column joins four token
rows, not eight feature lanes. The joined order is `[tokenrow4][feature64][token64]`;
its DDR drain sizes are `[4,64,64]`, strides `[64,T,1]`.

## Build and reproduce

From the Phlegm root, after loading its local toolchain:

```bash
source target/npu-dev/env.sh
python open_kernels/designs/gemm_q4_prefill/compensated64/build.py \
  --out target/npu-dev/my-compensated64-up --n 8192 --k 2048
```

Use a fresh output directory under ignored `target/`. `build.py` reads the
parent canonical dequant header and applies checked substitutions to generate
only its packed-output adaptation and four entry points under `out/source/`.
It uses the existing parent `gemm_bfp_compensated_emit.h`; no generated header,
compiled kernel or log belongs in this source folder. Build metadata, logs,
source hashes and all artifacts stay under `out/`.

To compare an image with a previously validated build, append
`--compare-cdo <previous-build-directory>`. A differing ELF/init/enable CDO is
recorded and rejected. For a full model, build all four projection streams
from these same sources and require all three CDO payloads to match before
sharing one hardware context. Do not mix streams from different images.

## Verification

CPU-only index simulation covers input panel/tap order, equivalence of existing
host packing, C deblocking/join/drain, exactly-once Y coverage and weight ranges:

```bash
python open_kernels/designs/gemm_q4_prefill/compensated64/check_layout.py \
  --n 8192 --k 2048 --tokens 256
```

The existing numerical checker accepts this global input format:

```bash
target/npu-dev/artlite-env/bin/python utilities/gguf-prefill/check_gemm.py \
 --kernel-dir target/npu-dev/my-compensated64-up/build \
 --driver target/npu-dev/harness-build/run_kernel \
 --out target/npu-dev/my-compensated64-up/check \
 --n 8192 --k 2048 --tokens 256 --activation-format bfp-hilo
```

Compare input and output bytes with the previous fused probe on the same
problem/seed: geometry preserves arithmetic order, so a tolerance pass alone
is insufficient. Then run complete-model300/8192-position and boundary gates,
CLI/bridge comparison, and alternating engine trials. Retain the known artificial
zero-history failure and all invalid timing samples. Compilation, other NPU
clients and memory-heavy CPU jobs must not overlap performance trials.

The tested ancestor has60,488B L1 use including the4KiB stack,5136B programs,
and64B largest observed individual frame. Its64x64 C FIFO has depth1, packed
B depth2, and weight FIFO depth1. Four-way C joins plus W/B use up to six
memtile input channels; W8+B4 shim inputs fit the sixteen available. These
checks must be repeated if the source, shape or toolchain changes. The ancestor
kernel passed numerical and byte-identity checks; its initial speed comparison
was under unrelated training load and is not idle performance acceptance.

Measured results and remaining gates belong in
[`specs/open-engine/gguf-prefill-linux.md`](../../../../specs/open-engine/gguf-prefill-linux.md).
