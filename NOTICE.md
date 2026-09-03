# Attribution & third-party terms

## phlegm derives from FastFlowLM
phlegm is an independent, open **host** for running Qwen3.6-MoE models on the AMD
Ryzen AI (XDNA2) NPU. It was reverse-engineered from, and reuses concepts and
file formats of, **FastFlowLM (FLM)** — https://github.com/FastFlowLM/FastFlowLM.
FLM's source is MIT-licensed (Copyright © 2026 Advanced Micro Devices, Inc.); the
MIT-licensed parts of this repository that derive from it retain that notice.

(The name is a joke: it *derives from FLM*, and it was built while its author had
a cold.)

## Two kinds of kernels: ours (MIT, in tree) and FastFlowLM's (closed, not in tree)
`tools/open-kernels/` contains phlegm's **own** NPU kernels, written from
scratch with the open IRON / mlir-aie toolchain. They are MIT like the rest of
the repo. The `rot13` smoke test and the q4 GEMV tile arithmetic started from
**vegah/LLMNpuTest** (https://github.com/vegah/LLMNpuTest), Apache-2.0; its
license is kept at `tools/open-kernels/designs/rot13/LICENSE.LLMNpuTest`.

phlegm does **not** contain, and does **not** redistribute, AMD's NPU kernel
binaries (FastFlowLM's `.xclbin` compute kernels and `q4_npu_eXpress` /
`qwen3_6_moe_npu` runtime). Only the closed-kernel backends (`l30`/`l40`/`li3`)
use them; the open-kernel path does not. Those binaries are **proprietary and patent-pending** and are governed
by FastFlowLM's own terms (paraphrased):

> Free for non-commercial, academic, or personal use, and for commercial use by
> companies with annual revenue at or below USD 10M. Above that threshold a
> commercial license from FastFlowLM is required. Unauthorized commercial use may
> violate patent law.

See FastFlowLM's `TERMS.md` for the authoritative text, and contact
info@fastflowlm.com for commercial licensing.

**To run the closed-kernel backends you must have FastFlowLM installed** (it
provides those kernels and, conveniently, the model files). The open-kernel
path has no such dependency. Everything in this repo is MIT. What you do with the AMD kernels
is between you and FastFlowLM's terms above.

## Other bundled third-party code
- `npu-engine/deps/XRT/` — headers vendored from **Xilinx/XRT**
  (https://github.com/Xilinx/XRT), Apache-2.0. Used to build the C++/XRT shim.
- `tools/q4nx-convert/reference/` (not committed; you clone it) —
  **FastFlowLM/FLM_Q4NX_Converter**, MIT. Used only for converter validation.
