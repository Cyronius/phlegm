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

## Experimental turbo HH only

`--turbo-hh` explicitly selects one BFP product per accumulator, HH only. It
ignores both low limbs and is **not compensated arithmetic**. The default
remains HH + HL + LH. Both modes retain FP32 C and exactly the same weight,
activation and output layouts. Turbo currently keeps the high/residual producer
and host packing unchanged; this experiment saves MAC work only. Hardware
compilation does not establish model quality or speed.

Build a fresh complete family through Dibs; all three CDO payloads must agree:

```bash
source target/npu-dev/env.sh
dibs run build --label "experimental HH-only T256 family" --eta 10m -- \
  python open_kernels/designs/gemm_q4_prefill/compensated64/build_family.py \
  --out target/npu-dev/turbo64-family --turbo-hh
```

The same family helper without `--turbo-hh` builds compensated streams. Each
stream's `build.json` records `input=bfp16_hilo` and its separate arithmetic
precision: `bfp16-hh` with `HH only; low limbs ignored` for turbo, or
`bfp16-hilo` for the default. Never infer compensation from the host ABI name.
Turbo needs independent numerical and model checks; preregistered experimental
model bounds are cosine >=0.9990, at most five percentage points argmax loss,
and at most 1% NLL increase. These are experiment acceptance bounds, not the
production compensated route's gates or a claim that turbo passes.

### Compact local high-only producer

`--turbo-high-only` is a separate experiment from `--turbo-hh`. It converts each
weight vector to its BFP high panel without computing a residual or writing a
low panel. Local A becomes 4,608B in `[row-pair][K8][row][high]` order. Its MAC
preserves the prior turbo HH accumulation order. Activation DMA and local B
remain high/low, and the MAC advances over each unused B low panel. The
external `bfp16_hilo` activation ABI, 6,144B weight chunks, FP32 output, and
previous modes are unchanged. This isolates producer savings; it does not yet
remove activation low-limb transfers or host packing work.

Use `build_family.py --turbo-high-only --out target/npu-dev/turbo64-high-only-family`
inside the same Dibs build command above, with a fresh output directory.
`check_layout.py --turbo-high-only` adds an independent compact-A panel walk.
Metadata identifies `producer=high-only` and `local_weight_panel_bytes=4608`.
Expected L1 including stack and RTP is 55,880B; compiled resources must be
checked. Compare output bytes to the prior HH-only variant before reusing its
quality evidence. Its known boundary NLL failure remains a limitation if the
outputs match; compact packing does not change the predeclared quality bounds.

### High-only activation DMA gather

`--turbo-high-ab` retains compact local A and gathers only B high panels from
unchanged host `bfp16_hilo` storage. Local B and its depth-two FIFOs become
4,608B per element. The MAC reads contiguous high panels and keeps the same
HH operation order. The default and `--turbo-high-only` retain their previous
DMA and local B layouts.

For K2048, each activation fill uses byte sizes `[32,64,72]`, strides
`[36864,144,1]`; K8192 uses `[2,64,64,72]`, strides
`[2359296,36864,144,1]`. Factoring the long-K dimension avoids an iteration
count of128 exceeding the six-bit limit. Maximum stride is589824 words,
inner width18 words, and all panel bases are word-aligned with positive
strides. A complete fill moves147456 or589824 bytes, not an individual72B
shim transaction. The small whole-transfer failure in xdna2-notes F-0221
therefore does not establish a blocker; hardware correctness remains required.

Build with `build_family.py --turbo-high-ab --weight-depth 1` under Dibs, into
fresh `target/npu-dev/turbo64-high-ab-family`. Run
`check_layout.py --turbo-high-ab` for the full semantic panel/byte walk,
including the long-K rank split. Expected L1 is46,664B including stack/RTP.
`--weight-depth 2` is an optional separate experiment, requiring compact AB
and an actual compiler placement check; its58,952B sum alone is not acceptance.
The default weight FIFO depth remains1. ABI metadata separately records
`activation_dma=gather-high-only`, both local panel sizes, and weight depth.
