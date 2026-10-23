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

## FastFlowLM

`benchmarks/reference/bridge.cpp` can be built against the caller's external
OpenFlowLM open engine (MIT, OpenFlowLM Community contributors). The engine is
not vendored here; its sources retain the license and notices in that checkout.

phlegm began as an open host for FastFlowLM's (https://github.com/FastFlowLM/FastFlowLM) kernels, and its findings
about FastFlowLM's file formats and kernels are in `xdna-dump/`. FastFlowLM's source is MIT (Copyright © 2026
Advanced Micro Devices, Inc.).

phlegm does **not** contain or redistribute FastFlowLM's closed NPU kernels (`.xclbin` files) or its closed runtime
libraries. Those are proprietary and governed by FastFlowLM's own terms. The capture tools in
`xdna-dump/tools/seq-capture/` run against a FastFlowLM install that you provide.

(The name is a joke: it *derives from FLM*, and its author had a cold at the time.)
