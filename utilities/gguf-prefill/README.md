# GGUF prefill experiments

These tools support the diagnostic experiments in
[the owning report](../../specs/open-engine/gguf-prefill-linux.md). They do not
promote a kernel or establish ArtLite parity. Run one NPU experiment at a time;
keep builds and memory-heavy CPU work outside timing windows.

## Compensated BFP probe

`build_bfp_stream_probe.py` generates an isolated design from the current
canonical GEMM using checked substitutions. It leaves the production design's
assertions and defaults intact. Its C++ kernel and emitter live in
`open_kernels/designs/gemm_q4_prefill/gemm_bfp_compensated_{probe.cc,emit.h}`.
It retains Q4 weights and uses BFP high/residual operand panels with three
products HH+HL+LH. The fused form emits panels directly from dequantization.

From the repository root, with the local toolchain environment loaded:

```bash
source target/npu-dev/env.sh
python utilities/gguf-prefill/build_bfp_stream_probe.py --fused \
  --out target/npu-dev/my-fused-probe --n 8192 --k 2048 --tokens 256
```

Use a fresh output directory. Build every stream. The fused form's activation
layout is `bfp-hilo`; the unfused form's is `bfp-hilo-pairs`:

```bash
target/npu-dev/artlite-env/bin/python utilities/gguf-prefill/check_gemm.py \
  --kernel-dir target/npu-dev/my-fused-probe/build \
  --driver target/npu-dev/harness-build/run_kernel \
  --out target/npu-dev/my-fused-probe/check \
  --n 8192 --k 2048 --tokens 256 --activation-format bfp-hilo
```

Packed-input checks import ArtLite's host encoder from the separate ArtLite
environment. They compare against independent dense source and BF16-operand
products, with the existing relative-error limits 0.005 and 0.0001. A passing
kernel check is not a model-quality gate. Full-model logits, window boundaries,
and engine timing remain prerequisites to integration. The CPU feasibility tool
`study_bfp_compensation.py` is not a hardware test.

`compare_gemm_trials.py` alternates already checked harness configurations,
discards the first four of sixty dispatches per process, brackets each pair with
reference controls, rejects drift above 5%, and records raw samples, hashes,
platform, host load and other NPU clients. It does not include host packing,
full-model scheduling or HTTP. All invalid pairs remain in its output.

`--ablate-matmul` / `--ablate-dequant` on the builder generate wrong outputs for
timing attribution only. Use `--timing-only-ablation` when comparing them; never
report an ablated route as a faster implementation.

`experiments/pack_bfp_hilo.cpp` is an AVX2 host-packing feasibility probe,
linked against the separately built reference runtime's `block_host.o` for the
native packing baseline. It accepts T, K, float32 input, native-output and
packed-output paths. Its finite input range is intentionally bounded; it has no
runtime hook. Byte checks against the host encoder and edge fixtures belong in
the owning experiment report.
