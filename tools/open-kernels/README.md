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

- `designs/expert_fetch/` — phase 0b spike (PASSED 2026-09-02): a shim DMA
  descriptor into a DDR pool retargeted to a runtime-chosen slab by control
  packets bounced through DDR, no host round-trip. `ddr_bounce_fetch.mlir` is
  the proof; the rest is the bisection ladder. See the plan's 0b section.
- `designs/gemv_q4/` — **phase 1**: q4_1 GEMV with in-kernel dequant, consuming
  chunks in the LAYER-POOL order FLM's kernel uses (`pools.rs std_perm`: 64-row
  bands of `K/128` chunks, half = c%2, k-tile = c/2). Tile arithmetic ported from
  vegah's `granite_gemv.h` (same chunk layout). 8 cores, x broadcast, 4 chunks
  per DMA element. Shape via env `GEMV_N/GEMV_K/GEMV_CORES`; `make_test.py
  --region qkv|z|share_up|share_gate|share_down` slices the region out of the
  captured L0 pool, writes an fp64 reference from the same bytes and a
  `run_<region>.cfg`; `compare.py <region>` checks. Results (random bf16 x):

  | region | shape | PASS | steady-state |
  |---|---|---|---|
  | qkv | 8192×2048 (10.5 MB) | cos 1.0, maxrel 1.2e-5 | 1.09 ms |
  | share_down | 2048×512 | cos 1.0, maxrel 7.6e-6 | 0.24 ms |
  | share_up | 512×2048 | cos 1.0, maxrel 1.1e-5 | 0.24 ms |

- `designs/lm_head_q8/` — **phase 1**: q8 lm_head GEMV from the captured
  lm_head pool (`C:/caps/m0d/000127.bo`, its own 128-row supertile order:
  32-chunk bands, quarter = c%4, k-tile = c/4). One entry point with a runtime
  `group` argument; 1940 bands split 243/242 over 8 cores with hand-built taps.
  `make_test.py [--bands B]` + `compare.py <tag>`. Full 248320 logits: **PASS
  cos 1.0, maxrel 2.9e-6, 21.4 ms** (540 MB, ~25 GB/s; FLM's closed lm_head:
  15.4 ms). 80-band subset: 0.67 ms at 33 GB/s.

- `designs/deltanet/` — **phase 1**: gated DeltaNet decode step, 32 v-heads,
  S[32,128,128] fp32 in/out + per-head (k, q, v, decay, beta) in, o[32,128]
  out. S does not fit L1, so it streams through each core (one per column,
  4 heads each) in 16-row slices, twice per head: pass 1 forms S^T k, pass 2
  writes S' = decay·S + k⊗delta and forms o = S'^T q/√128. Every fp32 product
  is a bf16 hi/lo split (AIE2P has no fp32 vector multiply). `make_test.py`
  takes S from a real captured boundary state (`C:/caps/pf_t11_full`) and
  random k/q/v/decay/beta; `compare.py` checks S_out and o. **PASS: S_out
  maxrel 6.2e-6, o maxrel 6.3e-6, 0.44 ms** (2 MB state read twice + written
  once).

  Two hardware facts learned here, both silent hangs (ERT state 8) otherwise:
  - **A shim DMA channel's start queue holds 4 BDs.** Queue more fills on one
    channel without awaiting and the extra ones are dropped; the core waits
    forever. Fix: `fill(..., wait=True, group=tg)` per head and `tg.finish()`
    before issuing more than 2 heads (4 BDs) ahead — the sequence in
    `dn_step.py`. Drains are issued first so cores never block on output.
  - **A column shim has 16 BDs total** and IRON packs 4 workers per column by
    default; designs with several fills per core need `Worker(..., tile=Tile(c, 2))`
    to spread one core per column (the verifier catches this one).

- `designs/dn_glue/` — **phase 1**: the linear-attention layer glue around the
  DeltaNet step, one core: alpha/beta projections (2048×32 bf16) → decay =
  exp(A·softplus(·+dt_bias)), beta = sigmoid; depthwise conv1d k=4 over
  [state rows, qkv] + SiLU; per-head L2 norm of q/k; emits the fp32[512]
  per-head records `designs/deltanet` consumes and the shifted conv state.
  Inputs from the captured L0 side pool (`C:/caps/m0d/000119.bo`, repacked
  into our 4 KB-element side blob by `make_test.py`) and the captured decode
  conv state (`C:/caps/m0c/000898.bo`); xn/qkv random. **PASS: state
  bit-exact, k/q/v maxrel ≤ 1.1e-5, decay/beta ≤ 1.4e-7, 0.71 ms.** Uses
  `ironutil.Pipeline` (throttled fills/drains) and fp32 vector exp/reciprocal
  helpers in `dn_glue.h` (`vexp32`, `vrecip32`, `vsigmoid32`).

  Traps met here (each cost a build-run cycle; all silent):
  - **`release(n)` frees the n OLDEST acquired elements.** You cannot hold one
    element (x) across later acquire/release pairs on the same fifo; copy it
    to a `Buffer` first (`glue_copy.cc`). Symptom: garbage that depends on x.
  - **Python-unrolled loops overflow the 16 KB program memory** —
    `XAie_LoadElf failed with XAIE_INVALID_ELF`. Use `range_` and pass indices
    as runtime ints.
  - **The objectfifo lowering allocates depth+1 buffers** when a side acquires
    `depth` at once; the aie-opt `MemoryMap` in the error is the truth.
  - **`aie::tanh` / `aie::invsqrt` / `aie::inv` are LUT approximations
    (~1e-2).** A sigmoid via `(tanh+1)/2` gave 2.6 % error on normalised q/k.
    Build fp32 transcendentals from bf16 MACs: `vexp32` (bit trick + degree-6
    poly, 1e-7) and Newton-refine the hardware `inv`/`invsqrt` seeds.
  - **Stack overflow hangs the core** (state 8): the fp32 exp path spills
    many 128-B vectors; `stack_size=0xD00` was too small, `0x1800` works. The
    stack sits at 0x0 with the fifo buffers right after it.
  - **The build caches too much.** `iron.jit`'s design cache is keyed on the
    Python source + CompileTime args, and aiecc keeps kernel sources/objects
    in `final.prj` and skips recompiling them — header edits produced
    bit-identical results three builds in a row. `build_design.py` now wipes
    `final.prj` before every build.

- `designs/ln/` — layer RMSNorm with fused residual add: y = x + add,
  xn = bf16(rms(y)·w). PASS, xn bit-identical to the fp64→bf16 reference.
- `designs/dn_post/` — DeltaNet post step: og = bf16(rms128(o)·ssm_norm ·
  silu(z)). PASS (4 of 4096 one-ulp bf16 differences).
- `include/vecmath.h` — the fp32-on-bf16-MAC toolkit every kernel above uses:
  `splitN`, `fmulN/faddN/fsubN`, `vexpN` (1e-7), `vrecipN`/`srsqrt`
  (Newton-refined hardware seeds), `vsigmoidN`/`vsiluN`.
- `designs/layer_chain/` — **MILESTONE (2026-09-02): a whole linear-attention
  layer runs on open kernels and matches the CPU replica.** Layer 0 of the
  captured 3LiF decode block (token 248068 at position 11, states from
  `C:/caps/m0c/000898.bo`, weights from the captured L0 pool/pack/side) as a
  host-driven chain of seven dispatches: ln → gemv(qkv) → gemv(z) → glue →
  dn_step → post → gemv(out) → ln(+residual, post-attn norm). Reference:
  `tools/kernel-interp/decode_step.py linear_decode` in fp64.
  `make_chain.py` (WSL, needs the 3LiF model) writes buffers + `run.cfg`;
  `compare_chain.py` checks:

  | output | cos | maxrel | note |
  |---|---|---|---|
  | xn (normed input) | 0.9999986 | 2.4e-3 | bf16 |
  | residual after attention | 0.9999996 | 9.4e-4 | fp32; error = bf16 xn/og rounding, as FLM |
  | MoE input xm | 0.9999975 | 2.7e-3 | bf16 |
  | DeltaNet state S | 1.0000000 | 2.8e-4 | fp32 |
  | conv state | 0.9999996 | 1.9e-3 | bf16 |

  ~15 ms for the chain as run (7 xclbin contexts, cold; the fused single
  dispatch is phase 2). Multiple `xclbin`/`kernelx` contexts in one driver
  config work; the glue's xn slot is filled by `dump xn` + `load side` (host
  round trip, test only).

- `designs/router/` — MoE router: bf16 GEMV 2048×256, fp32 softmax, top-8 on
  the core (positive floats compare as uint32 on the scalar unit), renormalised
  weights. On the layer chain's real MoE input: same 8 experts, same order as
  the fp64 reference; p maxrel 1.6e-5, w 2.9e-6.
- `designs/silu_mul/` — h = bf16(silu(g)·u): bit-exact.
- `designs/moe_combine/` — two designs: `moe_axpy` (acc += w[e]·y_e, one run
  per expert, slot from a small buffer, accumulator ping-pongs between two BOs)
  and `moe_fin` (out = xres + acc + sigmoid(xm·sgw)·shared). maxrel 6.9e-6.
  **A run with 14 buffer arguments is rejected by the firmware (ERT state 6,
  abort, and the driver stops)** — hence two designs; keep runs ≤ ~8 buffers.
- `designs/moe_chain/` — **the MoE block of layer 0 on open kernels, 48
  dispatches over 9 xclbin contexts**: ln → router → per expert (gemv up, gemv
  gate, silu_mul, gemv down, moe_axpy) → shared expert (gemv up/gate, silu_mul,
  gemv down) → moe_fin. Reference: fp64 from the replica's dequantised expert
  weights with bf16 xm and bf16 h exactly as the kernels round. **PASS: router
  idx identical, block output cos 1.0000000 maxrel 2.8e-5** (1.4e-4 vs the
  replica's fp32-activation version). Expert weights are sliced by the host
  from the reference's indices; the on-device fetch (phase 0b) replaces that
  slicing in the fused kernel.

Together with `layer_chain`, **all of layer 0 (attention + MoE) of the
captured decode step now reproduces on open kernels.**

- `designs/attn/` — full-attention decode step, one core: head RMSNorm ×
  effective q/k norm weights, partial RoPE (rotary 64 of 256, half-split,
  cos/sin for the position supplied by the host in a 2 KB meta record),
  online-softmax attention over the cached K/V rows (streamed 1 KB per row,
  two fills per position) plus the new position, sigmoid gate; emits the new
  bf16 cache rows. The number of cached rows is a CompileTime parameter
  (`ATTN_POS`): the runtime sequence is a static instruction stream.
- `designs/attn_chain/` — **layer 2 (full attention) of the captured decode
  step on open kernels**: ln → gemv q/gate/k/v → attn → gemv o → ln. PASS:
  residual cos 0.9999999 (maxrel 4.0e-5), new cache rows and gated output
  within bf16 rounding, 0.95 ms for the attention kernel at 11 positions.

  Two more traps: **a run with 9 buffer arguments aborts (ERT state 6); 6
  works** — so the GEMV gained `GEMV_YOFF`/`GEMV_YTOT` (write y at an offset
  of a shared output BO: `build_z_hi` puts gate after q, `build_512_hi` puts v
  after k). And **shared kernel bodies must be `inline` (vague linkage) +
  `noinline`**: `static` copies per translation unit overflowed the 16 KB
  program memory (`_XAie_LoadProgMemSection: Overflow of program memory`).

**All three layer types (linear attention, MoE, full attention) of the decode
step now run on open kernels and match the CPU replica.**

- `designs/decode_chain/` — **the whole decode step as one driver config**:
  L0 linear+MoE, L1 linear+MoE, L2 attention+MoE, final norm, lm_head — 164
  dispatches over 19 xclbin contexts, ~all in ~0.3 s of kernel time
  (lm_head 38 ms cold). `make_decode.py` slices every weight from the captured
  pools/packs/sides, predicts each layer's routing with the mirrored math (and
  adopts the NPU's own selection from a previous run if it differs);
  `compare_decode.py` compares logits with FLM's capture (odd vocab rows) and
  the replica. Result (2026-09-02):

  | | corr | top token |
  |---|---|---|
  | open kernels vs CPU replica | **1.00000** (residuals 0.999999 per layer) | same |
  | open kernels vs FLM capture | 0.671 | differs |
  | CPU replica vs FLM capture | 0.671 | differs |

  So the open kernels reproduce the replica's math exactly, and inherit its
  **pre-existing divergence from FLM** (the repo already deposed the CPU model
  as oracle for this reason: 0.57–0.68 vs both NPU paths, which agree at
  0.9996). The divergence is a semantics difference in some op, not
  accumulation; it must be found before the open engine is "correct". The
  decode block in `C:/caps/m0c` is FLM's older many-ops-per-layer flow with
  every op's buffers captured, so it can be bisected op by op against this
  modular chain. (Process exit after 19 contexts segfaults in XRT teardown,
  after all work and dumps are done — harmless, noted.)

Phase-1 status and what's next: `.claude/plans/open-kernels-feasibility.md`,
"Phase 1 progress".
