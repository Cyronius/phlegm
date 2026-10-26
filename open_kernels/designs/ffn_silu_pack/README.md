# Standalone SiLU + HILO pack prototype

Experimental, turbo-only arithmetic study. No C++ runtime integration, inference
context sharing, hardware acceptance, model-quality acceptance or speedup claim.
The scalar FP32 implementation is a functional baseline with soft-float helpers;
it is not intended as the final high-throughput glue kernel.

Inputs: native F32 `[F,T]` gate and up, T=256. Default F=8192, four cores;
eight cores is a build knob, not yet compiled/tested. Per worker, a contiguous
partition of K64 blocks is walked over T32 subblocks. Each input FIFO object is
`[64 features][32 tokens]` (8192 B). One output is 4608 B, in exact current
`bfp16_hilo` order `[K64][T32][token8][K8][hi72,lo72]`. Each 72-byte plane is
8 token rows, each one exponent followed by 8 signed mantissas. There are no
zero DMA strides. The fixture checks global native-input and output indexing.

Caller preflight: every input finite, |gate| and |up| <=1e8; all computed output
finite and <=1e20. This deliberately bounded prototype does not embed an error
channel or implement general nonfinite handling. For |gate|>=80, negative SiLU
is zero and positive SiLU tends to gate; negative underflow is deliberate.
Within the active range, a fifth-degree exp polynomial on [0,ln2], integer
exponent restoration and FP32 division implement sigmoid. CPU checks report
error against independent FP32 and double-exp SiLU separately.

Packing first rounds the computed F32 value to BF16 with integer RNE. Each
8-feature row uses exponent max(7,max exponent). Mantissas use RNE; a positive
128 rollover increases the shared exponent and retries. Negative -128 remains
valid. Residual is RNE BF16(value-decoded high), then independently encoded.
`check_cpu.py` uses a separate float64 scaling/rint encoder as its byte oracle;
`probe.py` independently vectorizes that oracle for the complete shape.
Packing the approximation is exact in CPU tests; it does not generally equal
packing exact SiLU. The output decoded relative error bar is 0.02, preregistered
for this operator only. Future inference changes still require the chosen
whole-model 300/8192-token and real boundary gates; this bar cannot replace them.

## Reproduce without NPU

From the repository root:

```sh
source target/npu-dev/env.sh
dibs run build --label 'SiLU pack CPU oracle' --eta 3m -- python open_kernels/designs/ffn_silu_pack/check_cpu.py --out target/npu-dev/parallel-pipeline/silu-pack-cpu-v1
dibs run build --label 'Standalone SiLU pack image' --eta 5m -- python open_kernels/designs/ffn_silu_pack/build.py --out target/npu-dev/parallel-pipeline/silu-pack-aie-v1
dibs run cpu --label 'Prepare SiLU pack fixture' --eta 3m -- python open_kernels/designs/ffn_silu_pack/probe.py prepare --out target/npu-dev/parallel-pipeline/silu-pack-fixture-v1 --build target/npu-dev/parallel-pipeline/silu-pack-aie-v1 --cpu-probe target/npu-dev/parallel-pipeline/silu-pack-cpu-v1
```

All destinations must be new directories. Generated files stay under ignored
`target/`. The builder records canonical source and artifact hashes. Fixture
preparation verifies source identities and saves immutable input/reference
hashes. No command above opens the NPU or performs performance measurements.

## Standalone device correctness, for the owning agent

The prototype author has NOT run this step. Serialize it with all other NPU
work under Dibs; source the same environment first. The larger timeout permits
a functional scalar probe and is not a performance target:

```sh
dibs run npu --label 'Standalone SiLU pack correctness' --eta 3m -- env HARNESS_TIMEOUT_MS=120000 target/npu-dev/harness-build/run_kernel target/npu-dev/parallel-pipeline/silu-pack-fixture-v1/run.cfg
python open_kernels/designs/ffn_silu_pack/probe.py check --out target/npu-dev/parallel-pipeline/silu-pack-fixture-v1
```

The checker requires byte equality to CPU candidate packing, then independently
checks decoded output error <=0.02 against double and FP32 SiLU. Byte failures
remain failures even if the loose operator error passes. `device-check.json`
records the device output hash. No automatic runtime integration follows.

## Recorded preparation result, 2026-10-26

CPU: 394 encoder fixtures including BF16 ties, positive rollover, negative -128,
zeros and tiny/large finite values passed byte comparison. Three small seeds
212–214 gave prepack SiLU relative error 1.23–1.30e-6 vs double, independently
below0.02 vs FP32. Full F8192/T256 seed212 gave 1.21423e-6 vs double,1.21384e-6
vs FP32, and decoded packed error0.00166343 vs double. CPU packing matched the
independent full-shape encoder exactly. At preparation time device behavior
was untested; the local NPU result below records subsequent validation.

Four-core build compiled. All four ELF .text sizes9392B (<16384). Buffers41984B
plus reserved stack4096B end at46080 (<65536). Disassembly: kernel frame896B,
main128B, observed helper frames64B; the stack-sizes section contains only
main_init64B, so that section alone is incomplete. Each column uses2 MM2S and1
S2MM shim channels. Outer repeat32<=64, positive strides; all local objects are
64-byte multiples. Raw compiler-budget evidence and disassembly:
`target/npu-dev/parallel-pipeline/silu-pack-aie-v1/{budgets.json,disassembly.txt}`.

This separate image would add a context/configuration transition when connected
to GEMM. Removing host transfers is not a net gain until complete-chain timing
includes this cost. Sharing the inference context requires a later explicit
image/layout design and rebuilding all streams.


## Local NPU result, 2026-10-26

Original full-shape fixture ran: numerical error .00166343 passes .02, but
CPU-candidate byte equality fails9 low-plane bytes. Keep this failure. The
separate extra-output diagnostic ran and its actual device F32 values encode
byte-exactly with the independent host oracle (zero mismatches); packed output
also equals the original kernel output. Device SiLU error is1.2143e-6 versus
double and1.2139e-6 versus independentFP32. See DIAGNOSTIC.md. This establishes
a standalone arithmetic/encoder split for the fixture, not model acceptance
or speed. No inference runtime integration or default promotion.
