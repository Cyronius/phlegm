# Original strict failure and decomposition diagnostic

2026-10-26: root ran the original four-core image on the NPU. Strict
CPU-candidate byte gate FAILED:9 bytes differ among4718592. Decoded relative
errors double0.001663429978956 / FP320.001663429800884 were below0.02; this
does not turn the original byte failure into a pass. Original output SHA256:
`80cd03be8af2a02b911d3661375a26a5519d6f294be89458b68a7aa402bd916c`.

All nine bytes are residual-plane mantissas; exponents and all high planes
match. Every affected CPU prepack F32 value is at most3 float ULPs from a BF16
midpoint. Perturbing only that value by1–4 ULPs across the midpoint reproduces
the changed row's complete device encoding. This suggests scalar arithmetic
rounding drift, but is not proof of device encoder correctness. Coordinates,
source values and bit patterns are saved in the ignored
`target/npu-dev/parallel-pipeline/original-nine-byte-diagnostic.json`.

A NEW diagnostic preserves the original kernel math/packing and adds a fourth
BO containing the actual computed prepack F32 values. Original kernel/design,
build and failed fixture remain unchanged. Diagnostic files:
`diagnostic.cc`, `diagnostic.py`, `build_diagnostic.py`,
`probe_diagnostic.py`. The added values FIFO has depth1; all four core .text
sizes9504B, frames896/128/64B, stackreserved4096B, total placedL1 end54272B.
Two MM2S and two S2MM channels/column fit. Compiler flags remain
`-ffp-contract=off`; the image is separate from all inference contexts.
All-core disassembly/flags/budget evidence is saved alongside the build.

Prepared artifacts:

- Build: `target/npu-dev/parallel-pipeline/silu-pack-diagnostic-aie-v2`.
- Fixture: `target/npu-dev/parallel-pipeline/silu-pack-fixture-v2`.
- The earlier values-only `silu-values-aie-v1` is unused.

Owning agent may run the following after all competing NPU work completes:

```sh
source target/npu-dev/env.sh
dibs run npu --label 'SiLU actual-values packing diagnostic' --eta 3m -- env HARNESS_TIMEOUT_MS=120000 target/npu-dev/harness-build/run_kernel target/npu-dev/parallel-pipeline/silu-pack-fixture-v2/run.cfg
python open_kernels/designs/ffn_silu_pack/probe_diagnostic.py check --out target/npu-dev/parallel-pipeline/silu-pack-fixture-v2
```

This checker requires independent RNE encoding of actual device F32 to match
all device packed bytes exactly. It separately checks actual device F32 error
<=0.02 versus independent double-exp and FP32 SiLU. It explicitly records the
original strict byte failure and checks its immutable output hash. This is a
different, decomposed correctness experiment, not a relaxed original gate.
No NPU job was executed by the prototype author. No timing, model-quality or
runtime-integration claim follows even if this diagnostic passes.


## Local diagnostic result, 2026-10-26

The extra-output diagnostic ran on the NPU. Independent encoding of actual
device FP32 values matched every packed byte, and diagnostic packed output
matched the original device output. Actual FP32 relative errors were
1.214261629e-6 against double and1.213874056e-6 against independent FP32,
within the declared0.02 bound. The original nine-byte CPU-candidate failure
remains recorded. Evidence: ignored
`target/npu-dev/parallel-pipeline/silu-pack-fixture-v2/diagnostic-check.json`.
This is standalone operator evidence; model integration and speed are untested.
