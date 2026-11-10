# Attribution and third-party terms

phlegm is MIT (see `LICENSE`). The parts below come from other projects and keep their own terms.

## open_kernels/

The open NPU kernels, MIT. They were written in phlegm and developed further in openflowlm-next
(https://github.com/Atomic-Germ/openflowlm-next) by Cyrus Attoun, Vegard Berget (vegah), Atomic-Germ, IvGeniy04,
PRIHLOP and D-revv. `open_kernels/PROVENANCE.md` has the full history and the per-file third-party notes:

- **vegah/LLMNpuTest** (https://github.com/vegah/LLMNpuTest), Apache-2.0: the GEMV tile arithmetic and the `rot13`
  smoke test. Its license is at `open_kernels/designs/rot13/LICENSE.LLMNpuTest`.
- **Xilinx/mlir-air**, MIT, and **Xilinx/mlir-aie**, Apache-2.0 WITH LLVM-exception: parts of
  `open_kernels/designs/whisper_fa/`, attributed at the point of use.

## npu_offload/ and kernel validation tools

Source imported from OpenFlowLM-Next, with original per-file notices retained.
The NpuEmbeddings GEMM and elementwise additions identify MIT licensing;
`npu_offload/gemm_rtp/gemm_pretiled.py` also credits AMD's mlir-aie whole-array
example under Apache-2.0 WITH LLVM-exception. See `npu_offload/PROVENANCE.md`
and `utilities/kernel-sync-sources.json` for source revisions and file hashes.

## phlegm-core/deps/XRT/

Headers from **Xilinx/XRT** (https://github.com/Xilinx/XRT), Apache-2.0. phlegm-core uses them to build its C-ABI
shim over XRT.

## C++ engine and JSON

Phlegm's permanent C++ engine in `src/open_qwen36/` derives from OpenFlowLM-Next
trunk `38ae3d2643` (base `beccd46` plus nine Gemma 4 patches). Copyright (c)
OpenFlowLM Community, MIT; the full notice ships in `LICENSE_OPEN_RUNTIME.md`.
The import preserves the upstream paths and source bytes. The engine was written as
OpenFlowLM's open engine (OFLM PR 6, Cyrus Attoun) and developed there by several
contributors; `specs/open-engine/engine-provenance.md` records the origin, author and
upstream status of every engine change. It does not include FastFlowLM's host headers
or any FastFlowLM binary. `src/open_qwen36/bfp16.hpp` credits ggml-xdna's MIT BFP16
packing (Cyrus Attoun).

`src/include/nlohmann/` is **nlohmann/json 3.12.0**, MIT, copyright
2013–2025 Niels Lohmann. Its SPDX notices remain in each header.

The GGUF reader and quantization layouts in `src/open_qwen36/gguf_file.*`
follow **llama.cpp / ggml** (MIT). The Saluki IQ tables are not part of this
trunk import. The third-party MIT terms below cover JSON and llama.cpp-derived code.

## FastFlowLM

phlegm began as an open host for FastFlowLM's (https://github.com/FastFlowLM/FastFlowLM) kernels, and its findings
about FastFlowLM's file formats and kernels are in `xdna-dump/`. FastFlowLM's source is MIT (Copyright © 2026
Advanced Micro Devices, Inc.).

phlegm does **not** contain or redistribute FastFlowLM's closed NPU kernels (`.xclbin` files) or its closed runtime
libraries. Those are proprietary and governed by FastFlowLM's own terms. The capture tools in
`xdna-dump/tools/seq-capture/` run against a FastFlowLM install that you provide.

(The name is a joke: it *derives from FLM*, and its author had a cold at the time.)

## Third-party MIT notices

MIT License

Copyright (c) 2013–2025 Niels Lohmann
Copyright (c) 2023–2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
