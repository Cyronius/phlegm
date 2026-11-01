# trit_embed: a ternary sentence embedder in one NPU dispatch

This is the NPU encoder for **trit-embed-3l**: a 17.2M-parameter English sentence embedder with
ternary weights {−1, 0, +1} and int8 activations. Its shape is 3 post-norm BERT layers, width 384,
12 heads, FFN 1536, tanh GELU, mean pooling, 384-d unit-norm output and a 128-token cap. All
three layers run on the 32 cores (8 columns × 4 rows) of the XDNA2 NPU in **one dispatch per
128-token slot**. Short texts share a slot.

| File | What |
|---|---|
| `trit_design.py` | IRON design: four token rows per core; weights broadcast as 4096-byte packets |
| `fused2.h`, `f2_*.cc` | AIE kernels: mask, activation quantization, ternary projections, attention, LayerNorm, output; they use `open_kernels/include/vecmath.h` |
| `build.py` | Compile to `build/` (`final.xclbin`, `insts.bin`, `build.json` with program size and hashes) |
| `host/td_npu2.cpp`, `build_host.py` | Persistent XRT C ABI with two bound runs (`tdn_bind` / `tdn_start` / `tdn_wait`) |
| `trit_npu.py` | Host: weight packing, text packing, `TritNpu(model_dir).encode(texts)` |
| `check.py` | NPU vs CPU-reference cosine, and timing |

## Build and run (Linux)

```sh
source target/npu-dev/env.sh                       # mlir-aie 1.4.2, Peano, XRT
python open_kernels/designs/trit_embed/build.py    # ~1 min; 15952 of 16384 program bytes per core
python open_kernels/designs/trit_embed/build_host.py
# Runtime needs torch, transformers and safetensors, plus the model folder
# (config.json, model.safetensors, tokenizer, trit_embed.py):
python open_kernels/designs/trit_embed/check.py path/to/trit-embed-3l
```

```python
from trit_npu import TritNpu
npu = TritNpu("path/to/trit-embed-3l")
vectors = npu.encode(["how do index funds work", "Index funds track a market index."])
```

`TritNpu` checks the build manifest before loading the graph: the model must be 3 layers wide and
384 wide, the sources must be unchanged since the build, and no profiling stub is allowed. The model
folder (format `trit-packed-v1`) stores each encoder linear as 2-bit ternary codes plus one bf16 scale
per output row. The host relays those codes and scales into 4096-byte packets unchanged. The host side
runs through the model folder's own reference module: the int8 embedding lookup and input LayerNorm,
pooling, the output projection and normalization.

When running from a Claude Code session, raise `ulimit -l` first: the user systemd manager caps
locked memory at 8 MiB. On a shared machine, hold the NPU (`dibs run npu -- ...`).

## Arithmetic

- The NPU does all Q/K/V/O and FFN projections (packed 2-bit ternary weights × int8
  activations), activation quantization, 12-head attention with segment masks, online softmax,
  tanh GELU, residuals and both post-residual LayerNorms.
- Q/K/V, attention probabilities and GELU outputs round to bf16.
- FFN-down keeps six independent 256-channel activation scales.
- The CPU does tokenization, the embedding lookup and input LayerNorm, mean pooling, the fp32
  384×384 output projection and L2 normalization. These are included in every timing below.

Memory: the weight stream is 2.09 MB (3 × 170 packets of 4096 bytes). The host keeps the int8 word
and position embeddings with their fp32 row scales, plus the fp32 token-type embedding, LayerNorm and
384×384 output projection: 12.5 MB.

## Measured

Strix Halo (Ryzen AI Max+ 395), Fedora 44, Linux 7.2.11, mlir-aie 1.4.2, llvm-aie
21.0.0.2026080301+c9c5ecb7, XRT from `target/npu-dev`. Text to vector includes tokenization, with two
host threads.

**Quality, 2026-10-31.** Real NPU vectors from the release files (`trit-packed-v1`) were scored
against the CPU reference on the same files: nDCG@10, full corpora, 128-token cap.

| | SciFact | NFCorpus | FiQA | SCIDOCS | Mean | STSB |
|---|---|---|---|---|---|---|
| NPU | 60.27 | 30.71 | 26.38 | 17.84 | 33.80 | 82.83 |
| CPU reference | 60.46 | 30.77 | 26.32 | 17.85 | 33.85 | 82.87 |

The paired query-bootstrap 95% CI for NPU minus CPU on the four-task mean is [−0.19, +0.08]. The
minimum vector cosine to CPU is 0.9977 on the STSB sentences.

**Speed, 2026-10-31,** on a quiet machine (`dibs run --measure`: CLEAN), release files:

| Workload | NPU texts/s | Dispatches |
|---|---|---|
| 2,758 STSB test sentences (packed) | 9,928 | 307 |
| 5,183 SciFact abstracts (one per dispatch) | 1,199 (0.83 ms each) | 5,183 |
| One sentence, batch 1 | 1.11 ms per text | 1 |

`TritNpu.dispatch_ms` times each run from its start to its wait. With two runs in flight that
includes the time a run queues behind the other, so its median (1.38 ms) is not the per-dispatch
cost. Throughput gives the cost: 4.32 s for 5,183 dispatches is 0.83 ms each.

**Speed, 2026-10-28,** on a quiet machine (`dibs` CLEAN), with the predecessor 3-layer checkpoint
of the same shape. These are the optimization steps on the King James Bible (31,102 verses,
1.07M tokens):

| Path | Dispatch | 31,102 verses | 9,360 passages |
|---|---:|---:|---:|
| Original fused, cap 32/128 buckets | 1.62 ms | 36.5 s | 18.6 s |
| + segment masks and packing | 1.09 ms | 12.8 s | 12.8 s |
| + host overlap (two bound runs) | 1.09 ms | 9.6 s | 9.9 s |
| + double-buffered weight broadcast | 0.975 ms | 8.0 s | 8.1 s |
| + cheaper exp and reciprocal | 0.83 ms | **7.5 s** (4,163/s) | **7.5 s** |
| CPU reference, 16 threads | — | 19.3 s | 27.6 s |

Dispatch times drift by about 10% between sessions, so compare rows measured together.

**Where the 0.84 ms goes.** Each part was stubbed in turn, using `build.py --stub ...` and
`--defs F2_PROF_*`:

- data movement and sync: 0.45 ms (54%);
- projections: 0.27 ms, of which GELU is 0.15 ms;
- quantization: 0.065 ms;
- attention: 0.04 ms;
- LayerNorm: 0.02 ms.

Weight blocks fill only 56–81% of the fixed 4096-byte packets. Weights stream from DDR on every
dispatch; they are not memtile-resident.

These were tried without gain: separate even/odd MAC accumulators, vectorized quantizer scales,
aligning texts to 8-token groups (−9%), and per-kernel `-Os`/unroll pragmas. IRON links kernels as
bitcode before one whole-program code generation, so those flags do not change core code size.

**Open:**

- resident or replicated weights;
- fuller packets;
- a `phlegm serve` embeddings route;
- Windows runs (untested here).

## Provenance

This was moved on 2026-10-31 from the ternary-embedder experiment's `native-stage/`, where it was
`fused2_design.py`, `fused2_encoder.py` and `host/td_npu2.cpp`. That directory was not under version
control. The design builds the identical program: all 32 cores' `.text`/`.data` and `insts.bin`
match the measured `build_f2_L3_cap128_d2` byte for byte. Only the xclbin metadata differs. On the
NPU, `TritNpu` vectors are bit-identical to the old encoder's on all 2,758 STSB sentences. The
kernels now include the shared `open_kernels/include/vecmath.h` instead of their own older copy. The
shared file adds only an opt-in `VECMATH_COMPACT` path, so the cheaper `fexp`/`frecip` stay in
`fused2.h`.
