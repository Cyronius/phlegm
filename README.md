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
npu-engine/        Rust host: q4nx reader, CPU forward, pool builders, three NPU
                    generate backends (li3/l30/l40), OpenAI server, CLI
  src/             q4nx.rs, forward.rs, pools.rs, decode.rs, xrt.rs,
                   generate_5li3.rs / generate_l30.rs / generate_l40.rs,
                   server.rs, main.rs
  m0/              C++ NPU decode drivers (runlist / ping-pong context driver)
  xrt-shim/        C-ABI shim over XRT's C++ API
  deps/XRT/        vendored XRT headers (Apache-2.0)
tools/
  kernel-interp/   Python originals the Rust backends above were ported from —
                   still used for the L40 one-time pool pre-build, and for
                   CPU-reference diagnostics (full_forward.py). See below.
  q4nx-convert/    GGUF -> q4nx converter + 1.0.3 read-support validation (Python only)
  server/          the pre-port Python OpenAI-compatible server (superseded by
                   `open-qwen-npu serve`; kept for reference)
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

1. **Fetch the NPU kernels.** The `.xclbin` kernels are proprietary
   (patent-pending, free to use under FLM's revenue-capped
   [TERMS](https://github.com/ROCm/FastFlowLM/blob/main/TERMS.md)) and are
   **not in this repo** — but they sit in FastFlowLM's public repo, so fetch
   them from there:
   ```pwsh
   pwsh -File tools/get-kernels.ps1          # layer + lm_head (the decode/serve path)
   pwsh -File tools/get-kernels.ps1 -All     # + the op-level prefill kernels
   $env:FLM_XCLBIN_DIR = "$PWD\kernels\Qwen3.6-35B-A3B-NPU2"
   ```
   (Equivalent by hand: `wget https://raw.githubusercontent.com/ROCm/FastFlowLM/v1.0.2/src/xclbins/Qwen3.6-35B-A3B-NPU2/layer.xclbin`
   etc. A FastFlowLM checkout or install also works — the engine's default
   path points at one. The script pins FLM **v1.0.2**: the captured kernel
   ELFs pair with that release's kernels, and upstream `main` already ships
   different bytes.) The per-op kernel *ELFs* (`elf_*.bin`) are a separate
   matter: FLM's engine generates them at runtime, so they must be captured
   locally with `tools/seq-capture` — they are neither downloadable nor
   redistributable.
2. **Build the C++/XRT drivers + shim** (Windows + MSVC + XRT), then the Rust
   engine with the `npu` feature — this builds `open-qwen-npu`, which now
   covers pool-building, generation, and serving (the old per-step Python
   scripts below are superseded except where noted):
   ```sh
   cd npu-engine/xrt-shim && ./build_shim.cmd
   cd ../m0 && ./build_nobarrier.cmd     # ping-pong context decode driver
   cd ..    && cargo build --release --features npu
   ```
3. **Generate.** There are three backends, one per model variant. `li3` and
   `l30` build their NPU weight pools in-process on first use (a Rust port of
   `build_pools.py`/`l30_build.py` — no separate build step); `l40` still
   needs a one-time Python pool pre-build (see below). All three ping-pong
   across two `layer.xclbin` contexts in runs of ≤3 to stay under the ERT
   queue cap, and stay finite on interval-3 where FLM NaNs.

   > **Paths are currently hardcoded per-backend** in
   > [`generate_5li3.rs`](npu-engine/src/generate_5li3.rs),
   > [`generate_l30.rs`](npu-engine/src/generate_l30.rs), and
   > [`generate_l40.rs`](npu-engine/src/generate_l40.rs) (`model_path`,
   > `xclbin_dir`, `kernel_dir`/`elf_dir`, `output_dir`/`buf_dir`) — they
   > default to the author's machine. Only `l40`'s `xclbin_dir` honors an env
   > override (`FLM_XCLBIN_DIR`); everywhere else, edit the `Default` impl for
   > your own paths until these are wired to config/env. This is
   > proof-of-concept status, not yet a clean multi-machine setup.

   **`li3-run` — interval-3, 5-layer slice** (the NaN-repro/debug config):
   ```sh
   cargo run --release --features npu -- li3-run "why is the sky blue?" 64
   ```
   Needs `model_5Li3.q4nx` (an FLM install has one, or slice one with the
   converter's `--layers` flag).

   **`l30-run` — the full pruned 30-layer model** (interval-3 throughout —
   this is the model phlegm exists for). Needs `model_30L.q4nx`, converted
   from Josh's pruned checkpoint on HuggingFace:
   [Cyronius/Qwen3.6-27B-A2.8B](https://huggingface.co/Cyronius/Qwen3.6-27B-A2.8B)
   (30 layers, 26.2B total / 2.83B active, pruned from Qwen3.6-35B-A3B — see
   [`docs/pruned-qwen36-support.md`](docs/pruned-qwen36-support.md)):
   ```sh
   cd tools/q4nx-convert
   python convert.py -i qwen36-27b-a2.8b-mtp-Q4KM.gguf -o out_30L
   # -> out_30L/model.q4nx  — point generate_l30.rs's DEFAULT_MODEL at this
   cd ../../npu-engine
   cargo run --release --features npu -- l30-run "why is the sky blue?" 64
   ```
   Pools stream from disk per-token here (30 × 512MB doesn't fit resident),
   so it's legitimately slower per token than `li3`/`l40` — inherent to the
   algorithm, not a bug.

   **`l40-run` — the full base 40-layer model** (interval-4, NPU-only
   prefill, no CPU forward math on the request path). Needs a one-time Python
   pool pre-build before the Rust backend can run (pools then stay resident
   across requests):
   ```sh
   cd tools/kernel-interp
   python pools_only_l40.py <path>/model.q4nx C:/code/FastFlowLM/npu-engine/m3out/l40
   cd ../../npu-engine
   cargo run --release --features npu -- l40-run "why is the sky blue?" 64
   ```

### Serve (OpenAI-compatible HTTP)
```sh
cargo run --release --features npu -- serve --backend li3   # or l30 / l40
# listens on :52625, /v1/chat/completions — same pool-prereqs as the *-run commands above
```
Configured via env vars (`FLM_HOST`, `FLM_PORT`, `FLM_MODEL_ID`,
`FLM_MAX_TOKENS`, `FLM_MAX_TOKENS_CAP`, `FLM_MODEL_DIR` for the tokenizer,
`FLM_BACKEND`); without `--features npu` only `--backend mock` is available.
This replaces the old `python -m tools.server` — see
[`server.rs`](npu-engine/src/server.rs), ported from
[`tools/server/`](tools/server/README.md).

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
