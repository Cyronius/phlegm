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

## Explicit turbo prefill

`phlegm run ... --turbo` and `phlegm serve ... --turbo` select a complete
`--kernels DIR` companion bundle at `DIR/turbo/`. The first implementation
uses only the HH BFP product for batched projections, instead of HH+HL+LH.
It retains the same high/residual input packing and Q4 source weights; ignored
low limbs are an intentional precision tradeoff. Decode and paired-probability
attention retain their existing arithmetic. This is experimental Llama1B
prefill support, not a clock/power mode or an ArtLite speed guarantee.

Build **all four** streams with
`open_kernels/designs/gemm_q4_prefill/compensated64/build.py --turbo-hh`.
Then `prepare_bfp_hilo_image.py --turbo --base DIR --out DIR/turbo` with
`--qkv`, `--o`, `--up`, `--down` pointing to their fresh `build/` directories.
The assembler verifies HH-only metadata, binary hashes and identical shared
CDO payloads. Base can use native BF16 or compensated packed-input projections;
existing packed buffer sizes must not be enlarged twice. Generated bundles stay
under ignored `target/`; they are not checked into git.

Turbo requires a matching `turbo.json` bound to both manifest hashes. Missing,
changed or unsupported bundles fail explicitly; `--mock` and sequential-only
prefill cannot be combined with turbo. Without the flag, the supplied base
bundle is selected as before. Turbo preserves auto scheduling (layer-major when available) or explicit block
prefill; the bridge must confirm a batched route.

Use `compare_model.py --profile turbo` for the separately declared quality
bounds: median cosine >=.9990, argmax agreement no more than5 percentage points
below the sequential control, next-token loss <=1.01 times control. `strict`
remains the default (.9999,1pp,1.005). Both profiles record forward KL from CPU
to route. Preserve strict failures and report model/context-specific quality
and speed; accepting a wider bound does not establish correctness or general
model quality. Require full long-prefix and real-boundary checks before timing. The independent
kernel turbo profile allows relative Frobenius error <=.02 against each source
and BF16-operand reference; strict .005/.0001 limits remain unchanged.

Initial local Llama1B evidence: turbo300 and full8192 pass the declared turbo
profile, but the real47-token4096 boundary slice fails the1% relative NLL bound
(+3.55%, absolute+.000504 nats/token;47/47 argmax unchanged). This is a known
experimental limitation, not an accepted all-context quality profile. Strict300
cosine also fails as expected. Live CLI turbo generation succeeds. See the
active plan for complete metrics and artifact hashes; no turbo engine speed
is accepted until clean paired timing is available.

2026-10-26 clean paired engine result: initial HH-only turbo is slower
(1111 versus matched compensated1136 prefill tokens/s,5/5 valid1K pairs,Dibs
CLEAN before/after). Lower precision is implemented but a speed benefit has
not been demonstrated. Keep the explicit quality failures alongside this
negative timing result; next investigation concerns producer/dataflow costs.
