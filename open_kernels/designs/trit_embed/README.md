# trit_embed: a ternary sentence embedder in one NPU dispatch

This is the NPU encoder for **trit-embed-3l**: a 17.2M-parameter English sentence embedder with
ternary weights {−1, 0, +1} and int8 activations. Its shape is 3 post-norm BERT layers, width 384,
12 heads, FFN 1536, tanh GELU, mean pooling, 384-d unit-norm output and a 128-token cap. All
three layers run on the 32 cores (8 columns × 4 rows) of the XDNA2 NPU in **one dispatch per
128-token slot**. Short texts share a slot.

| File | What |
|---|---|
| `trit_design.py` | IRON design: four token rows per core; weights broadcast as 8192-byte elements |
| `fused2.h`, `f2_*.cc` | AIE kernels: mask, activation quantization, ternary projections, attention, LayerNorm, output; they use `open_kernels/include/vecmath.h` |
| `gelu_lut.py`, `gelu_lut.h` | The GELU table and its generator (`--check` verifies the header) |
| `build.py` | Compile to `build/` (`final.xclbin`, `insts.bin`, `build.json` with program size and hashes) |
| `host/td_npu2.cpp`, `build_host.py` | Persistent XRT C ABI with two bound runs (`tdn_bind` / `tdn_start` / `tdn_wait`) |
| `trit_npu.py` | Host: weight packing, text packing, `TritNpu(model_dir).encode(texts)` |
| `check.py` | NPU vs CPU-reference cosine, and timing |
| `bench.py` | Builds against each other: alternating rounds, ms per dispatch, output differences, cycle profiles |

## Build and run (Linux)

```sh
source target/npu-dev/env.sh                       # mlir-aie 1.4.2, Peano, XRT
python open_kernels/designs/trit_embed/build.py    # ~1 min; 15536 of 16384 program bytes per core
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

`TritNpu` checks the build manifest before loading the graph: the model must be 3 layers deep and
384 wide, the sources must be unchanged since the build, and no profiling stub is allowed. The model
folder (format `trit-packed-v1`) stores each encoder linear as 2-bit ternary codes plus one bf16 scale
per output row. At load the host repacks them into the NPU's packets (packet format 2, below). The
host side runs through the model folder's own reference module: the int8 embedding lookup and input
LayerNorm, pooling, the output projection and normalization.

When running from a Claude Code session, raise `ulimit -l` first: the user systemd manager caps
locked memory at 8 MiB. On a shared machine, hold the NPU (`dibs run npu -- ...`).

## Arithmetic

- The NPU does all Q/K/V/O and FFN projections (packed 2-bit ternary weights × int8
  activations), activation quantization, 12-head attention with segment masks, online softmax,
  tanh GELU, residuals and both post-residual LayerNorms.
- Q/K/V, attention probabilities and GELU outputs round to bf16.
- GELU comes from a table: the input rounds to bf16, then one of 256 linear pieces over [−8, 8)
  (the hardware LUT's linear approximation), worst error 2.2e-4 against the exact GELU of that input.
- Softmax works in base 2: the 1/√32 score scale and log2(e) are one constant, and 2^x uses a
  cubic (1.4e-4 relative, against the 2e-3 of the bf16 the probabilities round to). The online
  softmax takes 64 keys a step.
- FFN-down keeps six independent 256-channel activation scales.
- The CPU does tokenization, the embedding lookup and input LayerNorm, mean pooling, the fp32
  384×384 output projection and L2 normalization. These are included in every timing below.

**Packet format 2.** A 4096-byte packet holds 32 output channels. In each byte the fields of channels
0–15 store code + 1 and those of channels 16–31 store the code. Masking with 0x33 or 0xcc then gives
int4 operands without a shift. The low fields' extra sum of activations is removed with the per-token
code sums that the quantizer leaves after its scales. From byte 3072 come the bias (fp32) and weight
scale (bf16), already repeated for the four token rows. All of this is exact: format 2 gives the same
bits as the original packing.

Memory: the weight stream is 2.11 MB (3 × 86 elements of 8192 bytes; each LayerNorm packet has an
element of its own). The host keeps the int8 word and position embeddings with their fp32 row scales,
plus the fp32 token-type embedding, LayerNorm and 384×384 output projection: 12.5 MB.

## Measured

Strix Halo (Ryzen AI Max+ 395), Fedora 44, Linux 7.2.11, mlir-aie 1.4.2, llvm-aie
21.0.0.2026080301+c9c5ecb7, XRT from `target/npu-dev`. Text to vector includes tokenization, with two
host threads.

**Quality, 2026-11-01.** Real NPU vectors from the release files (`trit-packed-v1`) were scored
against the CPU reference on the same files: nDCG@10, full corpora, 128-token cap.

| | SciFact | NFCorpus | FiQA | SCIDOCS | Mean | STSB |
|---|---|---|---|---|---|---|
| NPU | 60.36 | 30.68 | 26.27 | 17.92 | 33.81 | 82.84 |
| CPU reference | 60.46 | 30.77 | 26.32 | 17.85 | 33.85 | 82.87 |

The paired query-bootstrap 95% CI for NPU minus CPU on the four-task mean is [−0.17, +0.08]. The
minimum vector cosine to CPU is 0.9978 on the STSB sentences. The 2026-10-31 build scored 33.80 and
82.83 the same way.

**Speed, 2026-11-01,** on a quiet machine (`dibs run --measure`: CLEAN before and after), release files:

| Workload | NPU texts/s | 2026-10-31 build | Dispatches |
|---|---|---|---|
| 2,758 STSB test sentences (packed) | 14,926 | 9,928 | 307 |
| 5,183 SciFact abstracts (one per dispatch) | 1,888 (0.53 ms each) | 1,199 (0.83 ms) | 5,183 |
| One sentence, batch 1 | 0.79 ms per text | 1.11 ms | 1 |

Back to back with fixed inputs (`bench.py`), a dispatch takes 0.48 ms against 0.80 ms for the
2026-10-31 build in the same sitting. The rest of the 0.53 ms per abstract is host work that the
two runs in flight do not hide.

`TritNpu.dispatch_ms` times each run from its start to its wait. With two runs in flight that
includes the time a run queues behind the other, so it is not the per-dispatch cost; throughput is.

**Optimization steps, 2026-11-01.** Each ratio is a step's time per dispatch over the previous
step's, both measured by `bench.py` in one sitting (alternating rounds). "Exact" means the final-layer
states are bit-identical to the previous step's.

| Step | Exact | × time |
|---|---|---|
| Await a layer's trace drain at the next K/V exchange; fetch K/V back with one 4-D shim task | yes | 0.98 |
| GELU from the 256-piece table instead of the fp32 sigmoid form | no | 0.86 |
| Quantizer \|x\| as a sign-bit mask (`aie::abs` on fp32 lanes compiles to 32 scalar extracts and inserts) | yes | 0.95 |
| Softmax in base 2 with the cubic 2^x | no | 0.96 |
| Trip-count hints, so the MAC loop and the quantizer loops are software-pipelined | yes | 0.96¹, 0.99 |
| Packet format 2 and a straight-line epilogue | yes | 0.94 |
| The softmax exponentials as one loop | yes | 0.99 |
| 8192-byte broadcast elements: two packets a projection call, 64 keys an attention call | no | 0.92 |
| All, against the 2026-10-31 build | no | **0.605** |

¹ Measured together with the cubic 2^x (0.926 for both); 0.96 is that over the cubic's own 0.963.

The arithmetic changes (the GELU table, the cubic 2^x and 64-key softmax steps) were checked against
the CPU reference on the full four-task evaluation above.

**Speed, 2026-10-28,** on a quiet machine (`dibs` CLEAN), with the predecessor 3-layer checkpoint
of the same shape. These are the earlier optimization steps on the King James Bible (31,102 verses,
1.07M tokens):

| Path | Dispatch | 31,102 verses | 9,360 passages |
|---|---:|---:|---:|
| Original fused, cap 32/128 buckets | 1.62 ms | 36.5 s | 18.6 s |
| + segment masks and packing | 1.09 ms | 12.8 s | 12.8 s |
| + host overlap (two bound runs) | 1.09 ms | 9.6 s | 9.9 s |
| + double-buffered weight broadcast | 0.975 ms | 8.0 s | 8.1 s |
| + cheaper exp and reciprocal | 0.83 ms | **7.5 s** (4,163/s) | **7.5 s** |
| CPU reference, 16 threads | — | 19.3 s | 27.6 s |

Dispatch times drift by about 10% between sessions, and sit on discrete levels within one, so compare
rows measured together.

**Where the time goes now.** The cores are busy: with every kernel stubbed (`build.py --stub all`)
a dispatch takes 0.26 ms, against 0.48 ms with them. Per core and dispatch, in core timer ticks
(`--defs F2_PROF_CYCLES=<mask>`, read by `bench.py`), out of about 805,000 from the mask to the last
output:

- projections: 364,000. Q, K and V take 67,000, O and FFN-down 159,000, and FFN-up with GELU 138,000.
  A packet takes about 600 ticks: the MAC loop about a quarter, the epilogue for its 128 outputs
  (scale, bias, rounding, then GELU, residual or K/V layout) most of the rest.
- attention: 251,000. With 32-key steps its exponentials took about 40% of it and its bf16 products
  about a third. The products convert both operands to bfp16 every time, so q is converted again for
  every key group.
- quantization: 65,000;
- LayerNorm: 30,000;
- calls, FIFO handshakes and waits: about 95,000.

The timer reads go over the processor bus, which profiling builds enable from the runtime sequence.
Each mask adds code, and only some fit the 16 KiB at once; `--stub attend` makes room for mask 1.

These were tried without gain:

- a memtile ring in front of the broadcast (8 deep: no change; 16 exhausts the memtile's 24 BDs per
  channel);
- broadcast depth 3 or 4;
- fewer K/V shim tasks on their own;
- before 2026-10-31: separate even/odd MAC accumulators, vectorized quantizer scales, aligning texts
  to 8-token groups (−9%) and per-kernel `-Os`/unroll pragmas.

Quantizer templates specialized for fp32 and bf16 input do not fit in program memory. IRON links kernels as
bitcode before one whole-program code generation, so per-kernel flags do not change core code size.

**Open:**

- the projection epilogue (most of a packet's 600 ticks);
- bfp16 operands converted once per head instead of once per product;
- host work per dispatch (0.53 against 0.48 ms);
- two 128-token slots per dispatch;
- a `phlegm serve` embeddings route;
- Windows runs (untested here).

## Provenance

This was moved on 2026-10-31 from the ternary-embedder experiment's `native-stage/`, where it was
`fused2_design.py`, `fused2_encoder.py` and `host/td_npu2.cpp`. That directory was not under version
control. The moved design built the identical program: all 32 cores' `.text`/`.data` and `insts.bin`
matched the measured `build_f2_L3_cap128_d2` byte for byte, and `TritNpu` vectors were bit-identical
to the old encoder's on all 2,758 STSB sentences. The kernels include the shared
`open_kernels/include/vecmath.h` instead of their own older copy. The 2026-11-01 changes are in the
table above.
