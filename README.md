# phlegm

**An open runtime and open NPU kernels for running Qwen3.6-MoE models on
AMD Ryzen AI (XDNA2) — started to fix a bug in the official one, now running
the whole model on kernels we wrote ourselves.**

Cyrus pruned Qwen3.6 down to a smaller model (fewer layers, still strong) to
run on AMD's Ryzen AI NPU. The official app for that NPU, FastFlowLM, fails
on it silently: it skips the full-attention block on this model's layout and
outputs `NaN` or garbage. The model isn't broken. phlegm began as a
different host driver for FastFlowLM's closed kernels that schedules them
correctly, and has since replaced the kernels themselves with open ones.

> The name comes from **FLM** (FastFlowLM) — written while its author had a cold.

> **Status (2026-09-03):** two working paths on real hardware.
> 1. **Open kernels** (`tools/open-kernels/`): a full decode step for the
>    30-layer pruned Qwen3.6-27B-A2.8B runs entirely on kernels in this repo,
>    no FastFlowLM binary involved: **~155 ms/token (about 6 tok/s)**, 62
>    dispatches, greedy output matching the CPU reference at every position.
>    Driven by the config harness today; the resident backend that serves it
>    over HTTP is the next item.
> 2. **Closed-kernel host** (`npu-engine/` backends `l30`/`l40`/`li3`): runs
>    FastFlowLM's own kernels correctly on the model FastFlowLM mis-executes.
>    This is what the HTTP server uses right now.
>
> Both have rough edges (hardcoded paths, WSL build for the kernels). A
> slower CPU-only mode works on any machine.

---

## Open kernels: the whole decode step, no closed binaries

`tools/open-kernels/` holds our own AIE kernels for XDNA2, written with the
open IRON / mlir-aie toolchain (75 core kernels, ~8.7k lines, MIT). Together
they cover every op in a Qwen3.6-MoE decode step:

- q4 GEMV with dequant fused into the integer matrix unit (weights stay
  4-bit in memory; nothing is unpacked on the host or in DDR)
- gated DeltaNet state update, conv1d / SiLU / q-k norm glue, post-norm
- full attention with a dynamic KV length (RoPE, online softmax, gated output)
- MoE router with on-device top-8, and the 8 routed experts plus the shared
  expert and combine as **one dispatch**, reading experts straight from the
  resident weight pool by runtime index (no host round trip, no host slicing)
- RMSNorm + residual, silu*mul, q8 lm_head

Measured on Cyrus's pruned 27B (30 layers, full-attention interval 3), one
xclbin context for the whole layer:

| milestone | ms / token | notes |
|---|---|---|
| phase 1, one dispatch per op | 1239 | 1622 dispatches, correct logits |
| routed experts as one dispatch | 460 | |
| MoE block (routed + shared + combine) fused | 348 | |
| linear-attention layer as 3 dispatches | 202 | |
| full-attention layer as 1 dispatch | 313 | 132 dispatches; measured under heavier box load |
| q4 GEMV moved to the integer matrix unit | 208 | same load as the row above; bandwidth-bound now |
| whole layer in one xclbin context | 165 | 62 dispatches |
| dynamic KV, 3-token greedy decode | **157 / 154 / 155** | argmax matches the reference at every position |

Every kernel has a `make_test.py` / `compare.py` pair against a byte-exact
CPU reference, and the README there records the XDNA2 traps met along the
way (a context switch resets the cores, `.bss` isn't zeroed, buffer args
≥ 5 carry a hidden offset bit, and so on). Status, per-kernel results, and
the build recipe: [`tools/open-kernels/README.md`](tools/open-kernels/README.md).
Plans and numbers per step: `docs/open-kernels-*.md`.

The GEMV tile arithmetic started from
[vegah/LLMNpuTest](https://github.com/vegah/LLMNpuTest) (Apache-2.0).

---

## What it is (and isn't)

- **Is:** an MIT-licensed program — mostly Rust, some Python, plus C++ AIE
  kernels — that runs Cyrus's pruned Qwen3.6 model correctly on the NPU. It
  includes our own open NPU kernels for the whole decode step, a converter
  (turns a model file into the format phlegm reads), and a small web server
  (so it behaves like a normal chat API).
- **Isn't:** a copy of AMD's NPU code. FastFlowLM's closed kernels
  (`.xclbin` files and the `q4_npu_eXpress` runtime) are **not included in
  this repo** and never were. The closed-kernel backends (`l30`/`l40`/`li3`)
  still need a FastFlowLM install to get them; the open-kernel path does not.
  See [`NOTICE.md`](NOTICE.md) for the license terms.
- **Scope today:** one model family (Qwen3.6-MoE), decode on the open
  kernels, prefill as sequential decode. Not a multi-model product shell.

### The bug, in plain terms
Qwen3.6 models mix two kinds of attention layers in a repeating pattern —
most layers use a cheaper "linear attention," and every Nth layer uses full
attention. Every Qwen3.6 model anyone else publishes uses N=4, and
FastFlowLM handles that fine. Cyrus's pruned model is the odd one out: it
uses N=3, and right at that boundary FastFlowLM's math overflows to `NaN` —
garbage output, every time. This isn't just "N=3 doesn't make sense" — an
independent reference implementation of the same architecture runs N=3 fine,
with sane, finite output. So it's a genuine defect in FastFlowLM's engine,
just one that only this unusual a configuration would ever trigger; nobody
running a standard Qwen3.6 model would hit it. The NPU's compute kernels
themselves are fine — phlegm's host runs the exact same kernels, schedules
them correctly for N=3, and stays correct. Full write-up:
[`docs/npu-open-engine.md`](docs/npu-open-engine.md).

### Other FastFlowLM issues found along the way
Not the main event, but worth flagging — found while working on the above,
unrelated to the `NaN` bug:
- **FastFlowLM's newer release, 1.0.3, is about 3x slower than 1.0.2** on the
  same hardware — decode dropped from ~6.8 to ~2.3 tok/s, prefill from ~3.1
  to ~1.0 tok/s, measured with FastFlowLM's own reported numbers. 1.0.3
  re-quantizes weights into a different on-device format, and the kernels
  that unpack it on the NPU grew accordingly. Because of this, phlegm still
  targets 1.0.2's kernels — porting to 1.0.3 would mean inheriting that
  slowdown for no correctness benefit, so it isn't worth it right now.
- **FastFlowLM's prefill (processing your prompt before it starts
  generating) is much slower than its own decode** — roughly 1-2 tok/s vs
  ~6.5 tok/s decode — because it regenerates NPU work per prompt token
  instead of batching. phlegm's NPU path avoids this (see
  [`docs/npu-prefill.md`](docs/npu-prefill.md)), which is part of why it's
  faster on short prompts.

---

## Requirements

**To try it on CPU** (works on any machine, no special hardware — slow, but
proves the fix):
- Python 3.11+ with `numpy`
- Rust (stable)
- A `model.q4nx` model file (from an FLM install, or made with the converter below)

**To run it for real, on the NPU:**
- An AMD Ryzen AI **XDNA2** NPU (Strix / Strix Halo / Kraken / Gorgon Point) and its driver
- The **XRT** runtime, and MSVC (Windows) to build the driver code
- **FastFlowLM installed** — only for the closed-kernel backends; that is
  where their kernel files come from. The open-kernel path needs the
  mlir-aie toolchain instead (see `tools/open-kernels/README.md`)

---

## Layout

```
npu-engine/        The Rust program: reads model files, builds NPU work,
                    runs the model (three modes — see "Running on the NPU"),
                    and serves an API. This is what you actually run.
  src/             q4nx.rs (model file reader), forward.rs (CPU math),
                   pools.rs (NPU work builder), generate_5li3.rs /
                   generate_l30.rs / generate_l40.rs (the three run modes),
                   server.rs (the API), main.rs (command line)
  m0/              C++ NPU drivers the Rust program calls into
  xrt-shim/        glue between Rust and AMD's XRT runtime (C++)
  deps/XRT/        vendored XRT headers (Apache-2.0)
tools/
  open-kernels/    OUR OWN NPU kernels (IRON / mlir-aie): designs/<op>/ with
                   the C++ core kernel, the IRON dataflow, a make_test.py that
                   slices real inputs, and a compare.py against a CPU oracle.
                   designs/decode_chain/make_27b.py runs the whole 27B step.
  kernel-interp/   Python originals the Rust program above was ported from —
                   still used for one setup step (see "Running on the NPU"),
                   and for CPU-only diagnostics
  q4nx-convert/    converts a GGUF model file into phlegm's format (Python only)
  server/          the old Python API server, replaced by the Rust one above
  seq-capture/     the toolkit used to reverse-engineer FastFlowLM's kernels
  golden-ref/      an independent reference forward pass, used to check correctness
docs/              design docs and the full technical write-ups
```

---

## Quickstart: prove the fix works (no NPU needed)

This runs the model's math on your CPU and shows you real, finite output on
the exact setup that makes FastFlowLM's NPU engine output `NaN`. Point it at
any `model.q4nx` file (e.g. one from an FLM install).

**Rust:**
```sh
cd npu-engine
cargo build --release
cargo run --release -- run "<path>/model_5Li3.q4nx" 248045,846,198,3710,369,279,6511,314,9338,30,248046
# prints: model format: q4_1|q4k ; logits: finite=true absmax ~8-10
```

**Python** (same math, more diagnostics):
```sh
cd tools/kernel-interp
MODEL_Q4NX="<path>/model_3LiF.q4nx" python full_forward.py
```

Either one auto-detects which of FastFlowLM's two file formats it's reading,
so it works on model files from either the older or newer version of FLM.

---

## Get a model file (GGUF → phlegm's format)

phlegm reads FastFlowLM's own model file format (`q4nx`) directly — if you
already have one, skip this. To make one from a regular GGUF file:

```sh
cd tools/q4nx-convert
git clone https://github.com/FastFlowLM/FLM_Q4NX_Converter reference
cd reference && python convert.py -i /path/to/model-Q4_K.gguf -o /path/to/out
#   -> /path/to/out/model.q4nx
```

To double-check the converted file is correct:
```sh
cd tools/q4nx-convert
python validate_v103.py
python validate_real_08b.py <converted>/model.q4nx <src>.gguf
```

Full byte-format details: [`docs/qwen36-1.0.3-format-support.md`](docs/qwen36-1.0.3-format-support.md).

---

## Running on the NPU with FastFlowLM's closed kernels (needs a FastFlowLM install)

This is the closed-kernel host: phlegm's own driver running FastFlowLM's
kernels correctly. It is the path the HTTP server uses today. (To run the
**open** kernels instead, see
[`tools/open-kernels/README.md`](tools/open-kernels/README.md):
`designs/decode_chain/make_27b.py --whole-layer --tokens N` builds the
config and `open-qwen-npu npu <cfg>` runs it. No FastFlowLM files needed.)

phlegm can run three versions of the model. Pick the one that matches what
you're trying to do:

| What you want | Run this | Needs this model file |
|---|---|---|
| **Run Cyrus's pruned model** — the reason phlegm exists | `l30-run` | `model_30L.q4nx` |
| Run the original, full-size, unpruned model | `l40-run` | `model.q4nx` |
| A tiny slice for quickly reproducing/debugging the `NaN` bug | `li3-run` | `model_5Li3.q4nx` |

(The names come from layer counts: `l30` = 30 layers, `l40` = 40 layers,
`li3` = a 5-layer slice around the interval-3 boundary where the bug hits.)

### 1. Get FastFlowLM's kernel files
These are AMD's proprietary compute kernels — not included here (only the
open-kernel path is self-contained), but public in FastFlowLM's repo, so
fetch them from there:
```pwsh
pwsh -File tools/get-kernels.ps1          # what you need for running/serving
pwsh -File tools/get-kernels.ps1 -All     # + extra kernels only needed for some setup steps
$env:FLM_XCLBIN_DIR = "$PWD\kernels\Qwen3.6-35B-A3B-NPU2"
```
This pins FastFlowLM version v1.0.2 — later versions ship different kernel
bytes that won't match phlegm's current setup. A FastFlowLM install/checkout
also works instead; the program looks there by default.

There's a second, separate set of files (`elf_*.bin`) that FastFlowLM
generates itself at runtime rather than shipping — those can't be downloaded,
only captured locally with `tools/seq-capture`.

### 2. Build phlegm
```sh
cd npu-engine/xrt-shim && ./build_shim.cmd
cd ../m0 && ./build_nobarrier.cmd
cd ..    && cargo build --release --features npu
```
This produces one program, `open-qwen-npu`, that does everything below.

> **Heads up:** the file paths phlegm uses (where your model file, kernel
> files, and working directory live) are currently hardcoded for the
> author's machine, in `generate_5li3.rs` / `generate_l30.rs` /
> `generate_l40.rs`. Only the kernel-file location can be overridden with the
> `FLM_XCLBIN_DIR` environment variable above — everywhere else, you'll need
> to edit those files with your own paths. This is proof-of-concept
> software, not yet a clean install.

### 3. Run it

**Cyrus's pruned model (recommended):**
```sh
cargo run --release --features npu -- l30-run "why is the sky blue?" 64
```
Get the model file from Cyrus's pruned checkpoint on HuggingFace —
[Cyronius/Qwen3.6-27B-A2.8B](https://huggingface.co/Cyronius/Qwen3.6-27B-A2.8B)
(30 layers, 26.2B total / 2.83B active, pruned from the original Qwen3.6-35B-A3B —
see [`docs/pruned-qwen36-support.md`](docs/pruned-qwen36-support.md)), then
convert it:
```sh
cd tools/q4nx-convert
python convert.py -i qwen36-27b-a2.8b-mtp-Q4KM.gguf -o out_30L
# -> out_30L/model.q4nx — point generate_l30.rs's DEFAULT_MODEL at this
```
This mode reloads its NPU work from disk for every token (it doesn't fit in
memory all at once), so it's noticeably slower per token than the other two
— that's expected, not a bug.

**Original full-size model:**
```sh
cd tools/kernel-interp
python pools_only_l40.py <path>/model.q4nx C:/code/FastFlowLM/npu-engine/m3out/l40
cd ../../npu-engine
cargo run --release --features npu -- l40-run "why is the sky blue?" 64
```
This one needs a one-time Python setup step first (builds and saves the NPU
work up front); after that it's the fastest of the three since everything
stays loaded.

**Debug slice:**
```sh
cargo run --release --features npu -- li3-run "why is the sky blue?" 64
```
Needs `model_5Li3.q4nx` (an FLM install has one, or make a small slice with
the converter's `--layers` flag).

### 4. Or run it as an API server
```sh
cargo run --release --features npu -- serve --backend l30   # or l40 / li3
# listens on :52625, /v1/chat/completions — same setup as the run commands above
```
Settings are environment variables: `FLM_HOST`, `FLM_PORT`, `FLM_MODEL_ID`,
`FLM_MAX_TOKENS`, `FLM_MAX_TOKENS_CAP`, `FLM_MODEL_DIR`, `FLM_BACKEND`.
Without `--features npu`, only a `--backend mock` (no real model) is available.

---

## Reverse-engineering toolkit

`tools/seq-capture/` is the toolkit used to figure out FastFlowLM's kernel
and file formats in the first place (how everything above was worked out).
`tools/golden-ref/` runs the real model independently, as a way to check
phlegm's output is correct. Neither is needed just to run phlegm.

---

## License

phlegm's code is **MIT** — see [`LICENSE`](LICENSE), including the open
kernels under `tools/open-kernels/`. The host derives from FastFlowLM (also
MIT). FastFlowLM's own NPU **kernels are not included** and are governed by
its terms — see [`NOTICE.md`](NOTICE.md); only the closed-kernel backends
need them. One kernel's tile arithmetic derives from vegah/LLMNpuTest
(Apache-2.0, notice kept alongside).
