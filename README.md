# phlegm

**An open host for running Qwen3.6-MoE on the AMD Ryzen AI (XDNA2) NPU** —
including the pruned *interval-3* hybrid-attention variants that FastFlowLM's
closed engine mis-executes (it overflows to `NaN`; phlegm produces finite,
correct logits).

phlegm reuses AMD's closed NPU compute kernels but replaces the closed host-side
orchestration with an **open, correct, and extensible** one. It also reads FLM's
model container in **both** the older 1.0.2 (`q4_1`) and newer 1.0.3 (`Q4_K`)
formats.

> The name derives from **FLM** — and it was written while its author had a cold.

> **Status: proof-of-concept / research.** The CPU reference path below runs
> anywhere and demonstrates the interval-3 fix. The NPU path needs supported
> hardware and a FastFlowLM install (for the kernels). See `docs/` for the full
> technical write-up.

---

## What it is (and isn't)

- **Is:** an MIT-licensed open host — a Rust engine + Python tooling — that
  drives AMD's NPU kernels to execute the model correctly, plus a byte-exact
  CPU reference forward, a GGUF→q4nx converter, and an OpenAI-compatible server.
- **Isn't:** a redistribution of AMD's kernels. **The `.xclbin` NPU kernels are
  NOT in this repo** — they are proprietary/patent-pending and ship with
  FastFlowLM. To run on the NPU you install FLM (which also provides the model
  files); phlegm supplies only the open orchestration. See
  [`NOTICE.md`](NOTICE.md) for the terms.

### Why it exists
FLM's closed host **NaN-collapses on `full_attention_interval=3`** models at the
full-attention → gated-DeltaNet boundary. The kernels are fine; the bug is in the
schedule, which is host-side. phlegm's open host runs the exact same kernels and
stays finite — proven on hardware. See [`docs/npu-open-engine.md`](docs/npu-open-engine.md).

---

## Requirements

**CPU reference path (works anywhere):**
- Python 3.11+ with `numpy` (and for the converter: `torch`, `gguf`, `safetensors`,
  `einops`, `tokenizers`).
- Rust (stable) for the Rust engine.
- A `model.q4nx` file (from an FLM install, or produced by the converter below).

**NPU path (additional):**
- AMD Ryzen AI with an **XDNA2** NPU (Strix / Strix Halo / Kraken / Gorgon Point)
  and the AMD NPU driver.
- **XRT** runtime, and MSVC (Windows) to build the C++/XRT shim and drivers.
- **FastFlowLM installed** — provides the `.xclbin` kernels and the model files.

---

## Layout

```
npu-engine/        Rust host: q4nx reader, CPU forward, stateful decode, NPU driver
  src/             q4nx.rs (format-aware reader), forward.rs, decode.rs, xrt.rs, main.rs
  m0/              C++ NPU decode drivers (runlist / ping-pong context driver)
  xrt-shim/        C-ABI shim over XRT's C++ API
  deps/XRT/        vendored XRT headers (Apache-2.0)
tools/
  kernel-interp/   Python CPU reference forward, NPU pool builders, generate loop,
                   tokenizer/sampler, and the 1.0.3 (Q4_K) reader (q4nx_v103.py)
  q4nx-convert/    GGUF -> q4nx converter + 1.0.3 read-support validation
  server/          OpenAI-compatible HTTP server
  seq-capture/     the XRT-proxy reverse-engineering toolkit (how FLM was decoded)
  golden-ref/      HF golden-reference forward (the interval-3 oracle)
docs/              design docs incl. the byte-exact 1.0.3 format spec
```

---

## Quickstart — CPU reference (no NPU): prove interval-3 runs finite

This needs no NPU. It reads a real q4nx model and runs the full forward on the
CPU, showing finite logits on the exact config FLM's closed engine NaN-collapses
on. Point it at any q4nx model (e.g. an FLM-installed one).

**Rust engine:**
```sh
cd npu-engine
cargo build --release
# prefill an interval-3 model and report logit health (finite = the fix works):
cargo run --release -- run "<path>/model_5Li3.q4nx" 248045,846,198,3710,369,279,6511,314,9338,30,248046
# prints: model format: q4_1|q4k ; logits: finite=true absmax ~8-10
```

**Python reference forward** (same math, more diagnostics):
```sh
cd tools/kernel-interp
MODEL_Q4NX="<path>/model_3LiF.q4nx" python full_forward.py
```

Both auto-detect the file format: **1.0.2 (`q4_1`, 5120 B chunks)** or
**1.0.3 (`Q4_K`, 4736 B chunks)** — same downstream code either way.

---

## Convert a model (GGUF → q4nx)

phlegm reads FLM's q4nx files directly. To make your own from a GGUF, use the
converter, which emits FLM's exact verified byte layout.

```sh
# one-time: vendor the reference converter (used for the newer 1.0.3 path + validation)
cd tools/q4nx-convert
git clone https://github.com/FastFlowLM/FLM_Q4NX_Converter reference

# convert a Q4_K GGUF to the newer 1.0.3 (Q4_K) q4nx format:
cd reference && python convert.py -i /path/to/model-Q4_K.gguf -o /path/to/out
#   -> /path/to/out/model.q4nx
```

Validate the 1.0.3 reader end-to-end (drives the reference packer + checks a real
converted file against `gguf.dequantize`):
```sh
cd tools/q4nx-convert
python validate_v103.py                                   # unit: vs reference _pack_q4k
python validate_real_08b.py <converted>/model.q4nx <src>.gguf   # vs gguf.dequantize
```

See [`docs/qwen36-1.0.3-format-support.md`](docs/qwen36-1.0.3-format-support.md)
for the byte-exact 1.0.2 vs 1.0.3 format spec.

---

## NPU path (supported hardware + FastFlowLM install)

This runs the model on the actual NPU by driving FLM's kernels from phlegm's open
host. High level (details and exact buffer layouts in
[`docs/npu-open-engine.md`](docs/npu-open-engine.md) and `npu-engine/README.md`):

1. **Build the C++/XRT drivers + shim** (Windows + MSVC + XRT):
   ```sh
   cd npu-engine/xrt-shim && ./build_shim.cmd
   cd ../m0 && ./build_nobarrier.cmd     # ping-pong context decode driver
   cd ..    && cargo build --release --features npu
   ```
2. **Build the per-layer NPU weight pools** from the q4nx file + serialize the
   CPU-computed prefill state:
   ```sh
   cd tools/kernel-interp
   python run_5li3_npu.py     # interval-3 slice, or:
   python l30_build.py        # full 30-layer model
   ```
3. **Generate** (autoregressive, NPU runs the layers, host samples):
   ```sh
   python generate_npu.py --prompt "why is the sky blue?"
   ```
   The decode driver ping-pongs across two `layer.xclbin` contexts in runs of ≤3
   to stay under the ERT queue cap — no wasted compute. Interval-3 stays finite
   where FLM NaNs.

### Serve (OpenAI-compatible HTTP)
```sh
python -m tools.server         # listens on :52625, /v1/chat/completions
```
See [`tools/server/README.md`](tools/server/README.md).

---

## Reverse-engineering toolkit

`tools/seq-capture/` is the XRT-proxy capture/replay toolkit used to decode FLM's
kernels and buffer formats (how all of the above was derived). `tools/golden-ref/`
runs the real HF model as the interval-3 correctness oracle. Both are research
artifacts, not needed to run phlegm.

---

## License

phlegm's code is **MIT** — see [`LICENSE`](LICENSE). It derives from FastFlowLM
(also MIT). The AMD NPU **kernels are not included** and are governed by
FastFlowLM's proprietary terms — see [`NOTICE.md`](NOTICE.md). Everything you need
to run on the NPU comes from your own FastFlowLM install.
