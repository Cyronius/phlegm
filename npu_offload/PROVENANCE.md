# Source provenance

Imported on 2026-10-23 from Atomic-Germ/OpenFlowLM-Next `main` at
`478dbea3f00f868ac58205803cc537b402c6cad5`:

- `gemm_rtp/`: embedding GEMM designs, exporter, layout and build checks. These
  are also dependencies of the LLM block-attention designs.
- `m5-eltwise/kernels/`: GELU and narrowing kernels used by those designs.
- `matmul/` and `section3/`: source-only experiments and synthetic fixture
  generators retained from the upstream kernel tree.

Phlegm's `matmul/export_gemma.py`, `gemma_shapes.py`, and `check_gemma.py` are
local additions that build and verify the twelve Embedding-Gemma 300M matmul
shapes from the imported whole-array design. Generated xclbins and fixtures
stay under ignored `target/`; no upstream binary was imported.

Compiled executables, binary fixtures and model weights were not imported.
`utilities/kernel-sync-sources.json` lists every imported source file and hash.

Keep the original file headers. The NpuEmbeddings additions identify MIT
licensing; `gemm_pretiled.py` also identifies its derivation from AMD's
mlir-aie whole-array example under Apache-2.0 WITH LLVM-exception. The import
does not replace those notices with Phlegm's project license.

These are kernel sources and build tools. Phlegm's embedding runtime is not
implemented by this import. See `gemm_rtp/README.md` for the upstream build flow.
