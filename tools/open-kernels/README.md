# open-kernels — our own NPU kernels (IRON / mlir-aie), driven by phlegm

Phase 0a of `.claude/plans/open-kernels-feasibility.md` (2026-09-01): prove we
can build an AIE kernel with the open toolchain and run it on the XDNA2 NPU
through phlegm's own XRT shim, the same way FLM's closed kernels are run.
**Done — ROT13 round-trips byte-exact via both loading paths.**

## Toolchain (WSL builds, Windows runs)

- WSL Ubuntu-24.04: `~/ironenv142` venv with `mlir_aie==1.4.2` (release wheel,
  `-f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.2`),
  `llvm-aie` (Peano 21.0.0) from `utils/peano-requirements.txt`, source tree
  `~/mlir-aie` checked out at `v1.4.2` (for `utils/env_setup.sh`).
- The wheel bundles `aiecc`, `aie-opt`, `aie-translate`, `bootgen` but **not**
  `xclbinutil` (needed for the xclbin) nor `aiebu-asm` (needed for the
  `insts.elf` wrap). Both are XRT tools; without sudo in WSL they were built in
  a throwaway `ubuntu:24.04` Docker container from XRT master (2.26.0) and
  installed to `~/xrt-tools/{bin,lib}` (boost .so's alongside). Recipe:
  `scratchpad/build_xclbinutil.sh` of the 2026-09-01 session — `apt` deps,
  `xrtdeps.sh -docker`, `cmake -DXRT_NATIVE_BUILD=yes`, `ninja xclbinutil
  aiebu-asm`.
- Native Windows wheels for `mlir_aie` 1.4.2 and Peano exist (Python 3.11);
  not used yet — same xclbinutil/aiebu gap, and WSL builds are 5 s.

Build shell:
```
source ~/ironenv142/bin/activate
source ~/mlir-aie/utils/env_setup.sh
export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib
cd /mnt/c/code/phlegm/tools/open-kernels
python build_design.py designs/rot13/rot13.py          # -> designs/rot13/build/{final.xclbin,insts.bin,insts.elf}
```
`build_design.py` pins the device to `npu2` (without it IRON silently targets
NPU1) and calls `DESIGN.specialize(**SPECIALIZE).compile(xclbin_path,
inst_path, elf_path)`; a design module just exposes `DESIGN` and `SPECIALIZE`.

## Running through phlegm

`open-qwen-npu npu <config>` (the decode driver's config language) gained two
directives for open designs:

- `kernelx <name> <xclbin-ctx> <insts.bin>` — classic mlir-aie flow:
  `xrt::kernel(ctx, "MLIR_AIE")` + a cacheable instruction BO bound at args
  1/2 of every run (word count at arg 2).
- `run <kernel> <buf>...` — generic immediate submit, buffers at args 3+.

FLM's flow (`kernel <name> <ctx> <insts.elf>`: `xrt::elf → module →
ext::kernel`) also loads IRON's `insts.elf` unchanged. Example
`designs/rot13/run.cfg`:
```
device
xclbin R .../rot13/build/final.xclbin
kernelx k R .../rot13/build/insts.bin      # or: kernel k R .../build/insts.elf
buf in 1024 .../rot13/in.bin
buf out 1024
run k in out
dump out .../rot13/out1.bin 1024
```
Result 2026-09-01: `run k [2 bufs] -> state 4 (0.602 ms)` classic, `0.447 ms`
ELF; output == ROT13(input) byte-exact for both.

## Designs

- `designs/rot13/` — smoke test. Kernel + dataflow from
  [vegah/LLMNpuTest](https://github.com/vegah/LLMNpuTest) (Apache-2.0,
  `LICENSE.LLMNpuTest`), dispatch half replaced by phlegm's driver. That repo
  is also the trap catalogue to read before writing any kernel here
  (device pin, floor rounding, no fp32 vector multiply on AIE2P, 2-in/2-out
  DMA streams per core, 128-byte shim transfers deliver zeros, ...).

Next: phase 0b, the routed-expert fetch spike (see the plan).
