# Attribution and third-party terms

phlegm is MIT (see `LICENSE`). The parts below come from other projects and keep their own terms.

## open-kernels/ (git submodule)

The open NPU kernels, the C++ engine (`src/open_qwen36/`, derived from OpenFlowLM-Next, MIT), `npu_offload/` and
the viz come from https://github.com/Cyronius/open-kernels. Its `NOTICE.md`, `LICENSE` and `LICENSE_OPEN_RUNTIME.md`
give their attribution and third-party terms. Phlegm's NPU builds compile that engine, including
**nlohmann/json 3.12.0** (MIT) and code that follows **llama.cpp / ggml** (MIT); their notices are below.

## phlegm-core/deps/XRT/

Headers from **Xilinx/XRT** (https://github.com/Xilinx/XRT), Apache-2.0. phlegm-core uses them to build its C-ABI
shim over XRT.

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
