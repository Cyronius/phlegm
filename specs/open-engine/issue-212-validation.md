> Imported on 2026-10-23 from the uncommitted openflowlm-issue212 worktree at 478dbea3. This is a historical OFLM validation record, not a new Phlegm hardware run. Its src/, registry and .agents/ paths refer to OFLM. Phlegm imports the kernel/spec changes and comparison utilities; companion OFLM runtime/catalog changes are preserved in open_kernels/upstream-patches/issue212-runtime.patch. No OFLM converter is imported.

# Issue 212: evaluation and fixes

This branch starts at `478dbea3` (2026-10-22 main). The report in [issue 212](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/212) used `99fe7e4`, Linux, Strix Halo MAX+ 395, XRT 2.27 and turbo. These results use Windows, Ryzen AI 9 HX PRO 370 / Strix, XRT 2.21, driver 32.0.20102.3930, firmware 1.1.2.64 and MSVC 19.44 Release. They establish a local before/after result; they do not reproduce the survey's absolute speeds or its complete OFLM/FLM/Artlite comparison.

## Findings and disposition

| Finding | Current evidence and change |
|---|---|
| Llama prefill attends token by token | Confirmed on main. Add Llama to the full-rotary `prep=rope` batched NPU route. Use the manifest's Llama-scaled inverse frequencies. Projection batching remains unchanged. |
| Batched attention accuracy | A simple recipe toggle was insufficient: one BF16 softmax-probability pass failed the longer-prompt comparison. Preserve high and low probability parts for Llama and evaluate both products, including cross-window rescaling. Keep Qwen and K2 probability paths unchanged. |
| Three small-model kernel sets absent | Add `llama32-1b`, `llama32-3b` and `qwen3-0.6b` to default export discovery, using catalog-compatible destination names. Native complete exports succeeded, including GGUF twins, decode, norm, head, projection GEMMs and every attention tier. Existing Llama 8B also exported completely. |
| Silent closed fallback | Already fixed on main: automatic fallback names the closed engine; forced-open selection refuses missing sets. No duplicate selection patch is needed. |
| Misleading attention startup message | Already fixed on main: dense sets without block attention say one dxB dispatch per token. |
| Stale Llama 8B tokenizer pin | Confirmed. Refresh the 185-byte tokenizer config and **also download the external chat template**. Updating only the tokenizer hash would lose the chat template moved out of that file. Public file metadata is recorded in `tests/fixtures/artlite_catalog_headers.json`. |
| Qwen catalog incorrectly advertises Q4_K | Confirmed from all six public container headers: 5120-byte Q4_1 and 8704-byte Q8 tiles. Correct labels to `Q4_1 + Q8`; no weight conversion is involved. |
| Upstream Qwen 27B Q8 embedding cannot load | Confirmed from the upstream header: `I8 [7760,20,8704]`, 248320x5120 values. Add a native Q8 row decoder and an independent Python reader path, with no requantization or full-table allocation. Route every Core embedding entry point through `embed_row`. Native synthetic tests cover every row of a multi-tile table, padding, boundaries and invalid requests. Full upstream 27B model inference was not run. |
| Unexplained extra memory | Confirmed duplicated projection weights: prefill copied immutable bytes already held for decode. Replace contiguous same-layout copies with XRT sub-buffer views; separately packed layouts retain owned storage. Log weights, extra/shared prefill bytes, state, constants, scratch, position tables and context capacity. |
| Native-context KV allocation | Confirmed fixed-capacity allocation, also acknowledged for FLM in the survey. This branch does not implement paged/growing KV. Select the same explicit context capacity on both sides of a comparison. |
| Decode context slowdown / weight bandwidth | Remains. Prefill batching and weight views do not change decode attention arithmetic or its KV walk. Follow-up optimization needs isolated decode dispatch/window profiles and its own fidelity checks. No blanket claim of FLM parity is justified. |

## Numerical verification

The exact sharing check uses the original compiled engine and the modified engine with the **same** kernel set. Llama 1B's six-position France prompt is bit-identical across every logit and all 16 generated tokens. The allocation ledger reports 608174080 shared bytes (580 MiB) and zero additional prefill weight bytes, removing 580 MiB of duplicated weight payload (592 MiB of the old 1-MiB-aligned XRT allocations). Llama 3B's ledger reports 1761607680 shared bytes (1680 MiB). These are engine buffer allocations, not measured total process RSS; instruction buffers, XRT internals and host scratch are outside this ledger.

`utilities/compare-open-logits.py` rejects missing positions, shape differences and nonfinite values. Its strict pairwise argmax gate exposes near ties; it is not evidence that the old NPU route is ground truth. `utilities/prefill-ref-q4nx.py`, ported from the repository's existing experimental branch with failure exit codes corrected, independently computes float64 residuals/attention and float32 projections from the same container values.

The CPU-reference gate is the pre-existing branch criterion: median per-position cosine >=0.9999, argmax agreement no more than one percentage point below the baseline, and next-token loss no more than 0.5% worse. Here the baseline is the old **block-projection, per-token-attention** route, not a one-token projection run.

| Llama 1B sample | Old route | Fixed batched route | Result |
|---|---|---|---|
| 300 positions, cosine median | 0.999992 | 0.999992 | Pass |
| Same sample, argmax vs CPU | 99.0% | 99.3% | Pass |
| Same sample, next-token loss | 2.7606 | 2.7599 | Pass (-0.02%) |
| 47-position arithmetic prompt | 93.6% argmax, loss 2.4308 | 95.7%, loss 2.4336 | Pass (+0.11% loss) |
| 37-position coding prompt | 97.3% argmax, loss 2.1039 | 94.6%, loss 2.1024 | Strict argmax gate fails by one extra near-tie flip; loss improves 0.07% |

Do not omit the short coding failure when reporting the broader pass. Its extra CPU-reference flip has a 0.00389-logit top-two margin. Both 32-token coding and arithmetic continuations matched the old 1B route. A 47-position 3B prompt has all 47 argmax choices equal between the NPU routes, but the later free-running continuations diverged; that is not a bit-identical generation claim. The 3B 300-position CPU-reference gate passes: both median cosines are 0.999992; argmax agreement improves from 99.7% to 100.0%, and loss changes from 2.6768 to 2.6777 (+0.03%).

Qwen3 0.6B sharing is byte-identical across 20 prompt positions plus 16 decode dumps. Llama's sharing check is byte-identical across 6 prompt positions plus 16 decode dumps. The exact comparator also distinguishes signed zero.

Boundary checks pass on the final engine: a real 257-token prompt spans a partial block, with final-logit cosine 0.999993 and the same 16-token continuation. Starting at position 4090 with a zero-filled prior cache and 47 real prompt tokens exercises the >4096 softmax merge: cosine 0.999995 and the same 16-token continuation. That second check validates the window/merge contract, **not long-context retrieval or needle recall**. Compact results, including the short coding failure, are saved in `tests/fixtures/issue212/quality-results.json`.

The fixed, self-authored passage, tokenizer IDs and repeated-prose timing input are under `tests/fixtures/issue212`. The timing input is a dense-engine microbenchmark, not a recall evaluation. Synthetic/random token timing cannot establish useful generation, and MoE timing can depend on routing as well as token count.

## Timing protocol and results

`utilities/benchmark-open-cli.py` interleaves the original and fixed Release executables on identical token IDs, 8192-token capacity, 32 output tokens and performance power mode. Each size gets a discarded warmup per variant and three measured repetitions. `OMP_WAIT_POLICY=PASSIVE` is set before launch. No other NPU jobs overlap the sweep. The first 1K warmup overlapped the tail of a compile-only export; it is excluded. CLI timings exclude model load, transport and tokenization, so they are **engine prefill**, not HTTP TTFT.

Raw artifacts in this worktree: `build-issue212/bench-llama1b/{metadata.json,results.json,*.stdout,*.stderr}`. Metadata contains executable, model, input and manifest hashes plus kernel build provenance. The compact raw samples and hashes are checked in as `tests/fixtures/issue212/llama1b-benchmark.json`.

| Prompt tokens | Old prefill tok/s, median [range] | Fixed prefill tok/s, median [range] | Speedup | Decode tok/s, old -> fixed median |
|---|---|---|---|---|
| 1024 | 125.5 [123.0-125.8] | 324.6 [297.3-327.5] | 2.59x | 36.07 -> 36.19 |
| 2048 | 83.1 [82.9-83.3] | 262.8 [253.6-263.6] | 3.16x | 27.96 -> 28.10 |
| 4096 | 49.6 [49.0-49.6] | 185.7 [184.1-190.6] | 3.75x | 19.32 -> 19.32 |

The 1K candidate has a roughly 10% range; three repetitions do not establish a precise small effect. The multi-fold prefill improvement is well above the observed spread. Decode did not improve, and fixed prefill still slows with context. These results are not evidence of FLM parity.

## Test fairness and remaining comparison work

The survey identifies real bugs; dismissing it as an environment problem is unwarranted. Its closed-FLM 8B comparator is a different model of the same shape, its Artlite comparisons use different quantizations, and its 27B FLM run uses a different container and MTP. Those are useful system-level comparisons, but they cannot isolate kernel arithmetic, quality, or speculative-decoding overhead. Re-run identical containers, token IDs/templates, context capacity, power mode, output count and speculation settings before claiming a kernel-only ratio.

A faithful performance run must also check actual prompt processing (not only requested length), cache reuse, output count, finish reason, finite logits, window/chunk boundaries and useful outputs. Needle recall is a separate quality check; these dense-engine measurements do not replace it. The survey's five-minute stop rule is reasonable for a bounded sweep, but its missing long-context points are censored results, not proof of unsupported context.

This machine has FLM 1.0.7 installed, whereas the original report used a different platform/runtime. No matched current FLM server sweep or Artlite binary comparison has been claimed. The 8B kernel export is compile-validated; no full 8B model inference was run. The complete Linux release preset/package and upstream 27B integration remain unrun. With less than 5 GiB free after the small test containers, downloading the 27B model is not feasible here.

## Reproduce

See [the build skill](../../.agents/skills/open-artlite-regressions/SKILL.md) for native toolchain and complete export commands. Example local timing run (paths to the model and reference build are installation-specific):

```powershell
$env:OMP_WAIT_POLICY='PASSIVE'
python utilities/benchmark-open-cli.py --reference-cli build-issue212/open_qwen36_cli-baseline.exe --candidate-cli build-issue212/Release/open_qwen36_cli.exe --reference-kernels build-issue212/kernels-llama1b-baseline --candidate-kernels src/xclbins/Llama-3.2-1B-NPU2/open_kernels --model <model-directory> --ids-file specs/open-engine/tests/fixtures/issue212/llama-context-8192.ids --sizes 1024,2048,4096 --max-ctx 8192 --max-tokens 32 --reps 3 --warmups 1 --out <new-results-directory>
```

Generate prompt logits with `--gemm-block --prefill-logits --dump-logits PREFIX`, once with `OFLM_OPEN_ATTN_BLOCK=0`, once with `=1`, using the same IDs. Then run:

```powershell
python utilities/prefill-ref-q4nx.py --model-dir <model-directory> --ids specs/open-engine/tests/fixtures/issue212/llama-quality-300.ids --dump baseline=<old-prefix> --dump batched=<new-prefix> --baseline baseline
python -m pytest specs/open-engine/tests -q
```

The final Python suite passes 859 tests, with 21 skipped. Native Q4NX embedding and manifest tests pass (2/2), as do all host-stage fixture checks. Generated-registry output matches in UTF-8 mode. The follow-up exact-gate suite passes 7 tests, including signed-zero identity and benchmark rejection cases. Native checks and build/skill validation commands are described in the build skill. The first full regression run found a stale Qwen3 fixture (M-specific attention build directories) and a pre-existing native Manifest test attempting copies of a move-only object; both are corrected without changing production manifest copy semantics.
