# phlegm

**An open replacement for the software AMD's NPU chip needs to run Qwen3.6-MoE
models — built to fix a bug in the official one.**

Cyrus pruned Qwen3.6 down to a smaller model (fewer layers, still strong) to
run on AMD's Ryzen AI NPU. The official app for that NPU, FastFlowLM, crashes
on it — not with an error, but silently: it outputs `NaN` (not-a-number)
instead of real predictions, so the model looks broken. It isn't. The bug is
in how FastFlowLM schedules work for the NPU chip, not in the chip or its
compute kernels. phlegm is a different driver for those same kernels that
schedules them correctly and gets real output.

> The name comes from **FLM** (FastFlowLM) — written while its author had a cold.

> **Status: proof-of-concept.** It works — proven on real hardware — but the
> NPU setup has rough edges (see below), and you need specific AMD hardware to
> use it. A slower CPU-only mode works on any machine and is the easiest way
> to see the fix for yourself.

---

## What it is (and isn't)

- **Is:** an MIT-licensed program — mostly Rust, some Python — that runs
  Cyrus's pruned Qwen3.6 model correctly on the NPU. It also includes a
  converter (turns a model file into the format phlegm reads) and a small web
  server (so it behaves like a normal chat API).
- **Isn't:** a copy of AMD's NPU code. The actual compute kernels (`.xclbin`
  files) that run on the chip are proprietary and **not included in this
  repo** — they ship with FastFlowLM. You still need FastFlowLM installed to
  get them; phlegm only replaces the part of FastFlowLM that decides how and
  when to run them. See [`NOTICE.md`](NOTICE.md) for the license terms on that.

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
- **FastFlowLM installed** — this is where the NPU kernel files come from

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

## Running on the NPU (supported hardware + FastFlowLM install)

phlegm can run three versions of the model. Pick the one that matches what
you're trying to do:

| What you want | Run this | Needs this model file |
|---|---|---|
| **Run Cyrus's pruned model** — the reason phlegm exists | `l30-run` | `model_30L.q4nx` |
| Run the original, full-size, unpruned model | `l40-run` | `model.q4nx` |
| A tiny slice for quickly reproducing/debugging the `NaN` bug | `li3-run` | `model_5Li3.q4nx` |

(The names come from layer counts: `l30` = 30 layers, `l40` = 40 layers,
`li3` = a 5-layer slice around the interval-3 boundary where the bug hits.)

### 1. Get the NPU kernel files
These are AMD's proprietary compute kernels — not included here, but public
in FastFlowLM's repo, so fetch them from there:
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

phlegm's code is **MIT** — see [`LICENSE`](LICENSE). It derives from
FastFlowLM (also MIT). The AMD NPU **kernels are not included** and are
governed by FastFlowLM's own terms — see [`NOTICE.md`](NOTICE.md). Everything
you need to run on the NPU comes from your own FastFlowLM install.
