# phlegm

**Open kernels and an open runtime for running LLMs on the AMD Ryzen AI (XDNA2) NPU.**

phlegm started as a fix for FastFlowLM, which silently skips the full-attention block of a pruned Qwen3.6 model and
outputs garbage. It went on to replace FastFlowLM's closed NPU kernels with open ones, written with the IRON /
mlir-aie toolchain. Those kernels now run whole models on the NPU, through the openflowlm-next engine.

> **Status (2026-10-26):** C++ remains the inference engine; the native Rust engine port comes last.
> CLI/server generation is connected to the resident C++ bridge. Local Llama 3.2 1B NPU generation and experimental kernels have hardware evidence; other models and HTTP performance still need acceptance. Tray model flows remain pending.

## Layout

| dir | what | state |
|---|---|---|
| [`phlegm-core/`](phlegm-core/) | Rust runtime support: tokenizer, sampler, OpenAI server, XRT bindings and resident C++ engine adapter | C++ bridge connected to CLI and server; native Rust engine port deferred |
| [`phlegm-cli/`](phlegm-cli/) | The `phlegm` command: `run`, `serve`, `bench` | real C++ inference path for run/serve/bench; model onboarding pending |
| [`phlegm-tray/`](phlegm-tray/) | Windows tray app (Tauri v2): model library, downloads, chat, a local OpenAI endpoint | tray icon and window only |
| [`open_kernels/`](open_kernels/) | The open AIE kernels, their recipes per model family, the build harness and the fp64 replicas | working; see its README |
| [`viz/`](viz/) | Animated NPU decode flow, with topology and timing explanations | Python exporter and HTML viewer; CLI integration pending |
| [`npu_offload/`](npu_offload/) | Embedding GEMM sources and block-attention build dependencies | source imported; embedding runtime port pending |
| [`specs/open-engine/`](specs/open-engine/) | Requirements and tests for the kernels and recipes (`OPEN-*`) | imported with the kernels |
| [`website/`](website/) | The project site (Astro + Starlight), for GitHub Pages | skeleton |
| [`xdna-dump/`](xdna-dump/) | What we know about the XDNA2 NPU, its toolchain and FastFlowLM, plus the tools that found it | raw sources; synthesis pending |

`open_kernels/` and `specs/open-engine/` are also in openflowlm-next (https://github.com/Atomic-Germ/openflowlm-next),
at the same paths. phlegm is their canonical home; `open_kernels/UPSTREAM.md` records what moves between the two.

## Building

Rust (stable) and MSVC on Windows:

```sh
cargo build                          # phlegm-core + phlegm-cli
cargo run -p phlegm-cli -- serve --mock  # hardware-free demo on :52625
```

On a Windows NPU machine, build the [resident engine bridge](benchmarks/reference/README.md),
then run a model with its matching open-kernel set:

```powershell
phlegm run --model-dir C:/models/my-model --kernels C:/kernels/my-model `
  --bridge target/reference-bench/phlegm-reference-bench.exe --prompt "Hello"
phlegm serve --model-dir C:/models/my-model --kernels C:/kernels/my-model `
  --bridge target/reference-bench/phlegm-reference-bench.exe
```

`phlegm serve --mock` explicitly selects the hardware-free demo. The model needs
`tokenizer.json`, `tokenizer_config.json`, and a chat template (external
`chat_template.jinja` or inline in the config). The bridge and its `.build.json`
sidecar must be built from this phlegm revision and a compatible OpenFlowLM
checkout. The CLI validates the sidecar before using the NPU.

The [Phlegm Bench protocol](specs/bench/spec.md) measures prefill and decode at
512, 1024, 2048, and 4096 input tokens, with JSON reports restricted to clean,
optimized builds from matching release tags. Machine metadata collection and
report validation work now (`phlegm bench --help`). A [temporary C++ reference
adapter](benchmarks/reference/README.md) runs the suite against an external
OpenFlowLM checkout, explicitly labeling its results as reference-engine runs.

The native Rust XRT bindings build with the `npu` feature. On Windows, first
create the XRT import library:

```sh
phlegm-core/xrt-shim/build_import_lib.cmd
cargo build --features phlegm-cli/npu
```

On Linux, set `XILINX_XRT` to an XRT install containing `include/xrt` and
`lib/libxrt_coreutil.so`, then build with a C++17 compiler. Set `XRT_INCLUDE_DIR`
or `XRT_LIB_DIR` if those directories are elsewhere. The XRT library directory
must also be on the runtime loader path. This compiles the Rust device bindings;
the model inference path currently still uses the reference bridge.

```sh
export XILINX_XRT=/path/to/xrt
export LD_LIBRARY_PATH="$XILINX_XRT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo build --features phlegm-cli/npu
```

The tray and the website use pnpm:

```sh
pnpm install
pnpm tray      # phlegm-tray in dev mode
pnpm site      # website on http://localhost:4321/phlegm
```

Kernels build with mlir-aie 1.4.2 + Peano, natively on Windows or in WSL: see
[`open_kernels/README.md`](open_kernels/README.md).

The [flow visualization](specs/viz/spec.md) models one decode step from an exported
kernel set. The kernel exporter writes `viz.json` automatically when the build
metadata is available. From the repository root, generate or render it with:

```sh
python -m open_kernels.viz <kernel-set-dir>
python -m open_kernels.viz.page <kernel-set-dir> -o decode.html
python -m pytest specs/viz/tests
```

Open `decode.html` in a browser to play the step and inspect its tiles and streams.

## License

MIT — see [`LICENSE`](LICENSE). Third-party parts and FastFlowLM's terms: [`NOTICE.md`](NOTICE.md).
