# phlegm

**Open kernels and an open runtime for running LLMs on the AMD Ryzen AI (XDNA2) NPU.**

phlegm started as a fix for FastFlowLM, which silently skips the full-attention block of a pruned Qwen3.6 model and
outputs garbage. It went on to replace FastFlowLM's closed NPU kernels with open ones, written with the IRON /
mlir-aie toolchain. Those kernels now run whole models on the NPU, through the openflowlm-next engine.

> **Status (2026-10-19):** this repo is being restructured as a monorepo, and the engine is being rebuilt in Rust.
> The kernels work today; the phlegm engine, CLI and tray app do not run models yet.

## Layout

| dir | what | state |
|---|---|---|
| [`phlegm-core/`](phlegm-core/) | The Rust engine: GGUF reader, kernel manifest, weight packing, XRT, decode, tokenizer, sampler, OpenAI server | server, tokenizer, sampler and XRT bindings salvaged; the engine port is next |
| [`phlegm-cli/`](phlegm-cli/) | The `phlegm` command: `run`, `serve`, `pull`, `bench` | `serve` with a mock backend |
| [`phlegm-tray/`](phlegm-tray/) | Windows tray app (Tauri v2): model library, downloads, chat, a local OpenAI endpoint | tray icon and window only |
| [`open_kernels/`](open_kernels/) | The open AIE kernels, their recipes per model family, the build harness and the fp64 replicas | working; see its README |
| [`specs/open-engine/`](specs/open-engine/) | Requirements and tests for the kernels and recipes (`OPEN-*`) | imported with the kernels |
| [`website/`](website/) | The project site (Astro + Starlight), for GitHub Pages | skeleton |
| [`xdna-dump/`](xdna-dump/) | What we know about the XDNA2 NPU, its toolchain and FastFlowLM, plus the tools that found it | raw sources; synthesis pending |

`open_kernels/` and `specs/open-engine/` are also in openflowlm-next (https://github.com/Atomic-Germ/openflowlm-next),
at the same paths. phlegm is their canonical home; `open_kernels/UPSTREAM.md` records what moves between the two.

## Building

Rust (stable) and MSVC on Windows:

```sh
cargo build                          # phlegm-core + phlegm-cli
cargo run -p phlegm-cli -- serve     # mock OpenAI server on :52625
```

NPU support needs the XRT import lib once, then the `npu` feature:

```sh
phlegm-core/xrt-shim/build_import_lib.cmd
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

## License

MIT — see [`LICENSE`](LICENSE). Third-party parts and FastFlowLM's terms: [`NOTICE.md`](NOTICE.md).
