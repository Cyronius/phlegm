# Running Cyronius/Qwen3.6-27B-A2.8B on FastFlowLM

Goal: build FastFlowLM from source and get the pruned Qwen3.6 MoE
(30 layers, 26.2B total / 2.83B active, pruned from Qwen3.6-35B-A3B) running on
the Ryzen AI NPU. Then decide what (if anything) to do with the IQ3 "speed demon" variant.

## What the research found

**Hardware on this machine is suitable.** Ryzen AI 9 HX PRO 370 (Strix Point,
XDNA2), NPU device present and OK, 47.6 GB RAM. FLM's official 35B package has a
24.3 GB footprint; the pruned model will be ~25% smaller. Driver must be
≥ 32.0.203.311 (verify in Device Manager before starting).

**The repo splits open/closed like this:**

| Open source (buildable/modifiable) | Closed (shipped prebuilt in-repo) |
|---|---|
| CLI, server, model registry ([model_list.json](src/model_list.json)) | NPU engines: `src/lib/{xrt,hrx}/qwen3_6_moe_npu.{dll,lib}` + `.so` |
| AutoModel orchestration incl. [modeling_qwen3_6_moe.cpp](src/common/AutoModel/modeling_qwen3_6_moe.cpp) | Quantizer/loader `q4_npu_eXpress` (header is public: [q4_npu_eXpress.hpp](src/include/tensor_utils/q4_npu_eXpress.hpp)) |
| Tokenizer, sampler, chat templates, download logic | NPU kernels: `src/xclbins/Qwen3.6-35B-A3B-NPU2/*.xclbin` |

**The closed engine is config-driven, not hardcoded to 40 layers.** Strings in
`libqwen3_6_moe_npu.so` show it reads `num_hidden_layers`,
`full_attention_interval`, `num_experts`, `num_experts_per_tok`,
`linear_num_key_heads`, etc. from config.json. It does NOT read a per-layer
`layer_types` array — only the periodic interval.

**The pruning maps onto that periodic scheme — if it was uniform.** Base model:
40 layers, full attention every 4th layer (10 full + 30 DeltaNet). The model card
says "one of every three gated-DeltaNet layers was structurally removed" → 30
layers, 10 full + 20 DeltaNet → pattern `[linear, linear, full] × 10` →
`num_hidden_layers: 30`, `full_attention_interval: 3`.
**⚠ Must confirm: the removed layers have to be one per 4-layer block, uniformly.
If the pruning kept an irregular layout, the closed engine cannot express it and
we'd be blocked (or need AMD to add `layer_types` support).**

**Model packaging is understood.** `model.q4nx` is a safetensors-format container
(JSON header + data). I pulled the official 35B header via HTTP range request:
733 tensors, names like `model.layer.N.linear_attn.ssm_*` /
`model.layer.N.mlp.{gate,up,down}_exps_proj.weight` (experts stacked into single
I8-packed tensors), separate q/k/v/o + gate for the 10 full-attention layers,
BF16 norms/router, F32 ssm_a/dt. **No MTP tensors — FLM ignores the draft head
entirely** (no speculative decoding on NPU). Vision lives in a separate
`vision_weight.q4nx`. Header dump script + output: scratchpad `q4nx_header.json`.

**An official GGUF→q4nx converter exists:**
[FLM_Q4NX_Converter](https://github.com/FastFlowLM/FLM_Q4NX_Converter) (Python).
It already handles the DeltaNet/hybrid tensor set (qwen3.5 configs) and MoE
expert stacking (gpt-oss module), and auto-dequantizes/requantizes non-Q4_0/Q4_1
inputs (e.g. Q4_K_M) with a small extra quantization error. **It does not yet
have a qwen3.6-moe architecture** — that's the main new code to write:
`configs/qwen3.6_moe.json` + `q4nx/models/qwen36_moe.py`, combining the qwen3.5
hybrid-attention mapping with gpt-oss-style expert stacking, targeting the exact
tensor names/shapes/dtypes observed in the official 35B header (note: this family
uses `model.layer.N.*`, not `model.layers.N.*`).

**Sideloading is easy.** `pull_model` skips downloading when all files listed in
the model_list entry already exist locally — so a custom entry in
`model_list.json` plus files dropped in the models dir is all the registration
needed. We build flm.exe from source anyway (`windows-default` VS2022 preset;
prebuilt engine libs are linked, xclbins ship in-repo).

## Phase 1 results (2026-08-26) — structure test DONE, verdict mixed

Method: extracted portable flm.exe 1.0.2 from the release MSI to `C:\flm-test`
(system install permission-blocked), downloaded the official 35B package to
`~\.flm\models\Qwen3.6-35B-A3B-NPU2`, then swapped in sliced `model.q4nx` +
edited `config.json` variants (originals kept as `.orig`; slicing scripts in the
session scratchpad, byte-verified against the original).

| Variant | Layers | Interval | Load | Output |
|---|---|---|---|---|
| Official baseline | 40 | 4 | OK | correct, ~6.6 tok/s |
| Whole-block drop | 24 | 4 | OK | coherent-ish, ~10.5 tok/s |
| First-8-layers | 8 | 4 | OK | diverse word salad (expected truncation damage) |
| Per-block linear drop (4k+1) | 30 | 3 | segfault 1st try, OK on retry | degenerate `/////` repetition |
| Per-block linear drop (4k+2) | 30 | 3 | segfault 1st try, OK on retry | identical degenerate `/////` |
| Tiny interval-3 | 6 | 3 | OK | degenerate `/////` |

**Conclusions:**
1. `num_hidden_layers` is genuinely config-driven — 24-layer interval-4 model
   loads AND generates plausibly. Layer-count flexibility confirmed.
2. **`full_attention_interval: 3` is accepted but mis-executed.** Every
   interval-3 variant (two structurally different 30-layer drops + a 6-layer
   one) produces the same degenerate single-token output, while even an 8-layer
   interval-4 stump produces distributionally plausible text. The engine reads
   the key (string present in the .so) but something in the closed engine —
   likely NPU sequence generation — appears to assume the 3-linear+1-full
   block layout.
3. Separate flaky load-time segfault: both 17GB interval-3 models crashed on
   first load and succeeded on immediate retry (same files). Worth reporting
   regardless.

**Implication for the pruned model:** as published (30 layers, interval 3) it
cannot run correctly on today's closed engine. Paths forward:
- **(a) File a public issue on ROCm/FastFlowLM** (likely a near-miss bug —
  the config key is already read; engine source is AMD-internal, so an issue +
  repro is the only channel). Repro artifacts kept in the model dir:
  `model_30L.q4nx` + `config_30L.json` (`model_30Lb.q4nx` is the second drop
  pattern; `model_6Li3.q4nx` is a small fast-loading repro).
- **(b) Re-prune by whole blocks** (drop N whole [L,L,L,F] blocks, keeping
  interval 4 → 36/32/28/24 layers), then LoRA-heal. Proven to execute
  correctly by the 24L test. Changes the model, but keeps everything else in
  the pipeline (converter work etc.) unchanged.
- (c) Wait/patch: no user-side workaround can express [L,L,F] on an engine
  that only executes [L,L,L,F].

~64 GB of test variants remain in the model dir for cleanup or repro use.
Windows source build still needs the C:\dev dependency setup (boost b2, XRT
headers, xrt_coreutil.lib via gendef, vcpkg-layout curl/ffmpeg/fftw); first
build attempt failed on missing headers, not yet retried.

## Phased plan (original)

### Phase 0 — Build & baseline (hours)
1. `git submodule update --init --recursive`; build with the `windows-default`
   preset per [src/README.md](src/README.md).
2. Verify NPU driver version ≥ 32.0.203.311.
3. `flm run qwen3.6-moe:35b-a3b` with the from-source binary. Confirms
   toolchain, NPU stack, and gives a performance baseline.
   (24.3 GB footprint — close other memory hogs on the 48 GB machine.)

### Phase 1 — Structure test with known-good weights (~1 day)
Answer "does the closed engine accept 30 layers / interval 3?" **before**
touching quantization:
1. Python script: read official 35B `model.q4nx`, drop one DeltaNet layer per
   4-layer block (same indices the pruning removed), renumber layers, write a
   30-layer q4nx. Pure tensor slicing — no repacking.
2. Edit config.json copy: `num_hidden_layers: 30`, `full_attention_interval: 3`.
3. Add `qwen3.6-moe:27b-test` entry to model_list.json pointing at the local dir
   (copy tokenizer/template/vision files from the official package).
4. Run it. Degraded-but-fluent output = engine is layer-count flexible → proceed.
   Crash/assert = closed-engine limitation → file a public issue on
   ROCm/FastFlowLM before investing in conversion.

### Phase 2 — Extend the converter (2–4 days; the main work)
1. Add qwen3.6-moe support to FLM_Q4NX_Converter as described above.
2. Gold-standard validation: if a GGUF of the *base* Qwen3.6-35B-A3B exists (or
   can be made with llama.cpp from the base checkpoint), convert it and diff
   tensor-by-tensor against FLM's official `model.q4nx`. Small numeric deltas
   from requantization are fine; names/shapes/dtypes/packing layout must match
   exactly.
3. Convert `qwen36-27b-a2.8b-mtp-Q4KM.gguf` (MTP tensors simply unmapped/dropped).
   Prefer the healed BF16 safetensors → f16 GGUF → q4nx route if available, to
   avoid double quantization (Q4_K_M → Q4_1).

### Phase 3 — Package & run the real model (~1 day)
1. config.json: base 35B config with `num_hidden_layers: 30`,
   `full_attention_interval: 3`; tokenizer/chat template from the base package;
   reuse official `vision_weight.q4nx` (vision encoder is untouched by pruning)
   or test whether the engine tolerates its absence.
2. model_list.json entry `qwen3.6-moe:27b-a2.8b`; run, sanity-check quality
   (chat, tool-calling — the FLM parser expects Qwen's `<function=...>` format),
   benchmark tokens/s and TTFT vs the 35B baseline. Expect prefill/decode gains
   roughly proportional to the removed layers (~25% of DeltaNet compute), not to
   the active-param ratio.

### Phase 4 — The IQ3 "speed demon": recommend skipping
- q4nx is a 4-bit (Q4_0/Q4_1) NPU format; there is no 3-bit path. Converting
  IQ3_S experts means dequant → requant to 4-bit → byte-identical economics to
  converting the Q4_K_M file, but with worse accuracy (triple quantization).
- Its second selling point, MTP speculative decoding, is unused by FLM.
- On the NPU, decode speed is set by active params and kernel schedule, not file
  size — the variant's advantages don't transfer. Only revisit if Phase 3 shows
  a memory-bound bottleneck.

## Risks / open questions

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pruning not uniform per block → inexpressible via `full_attention_interval` | Unknown — **ask/check first** | Inspect GGUF metadata or pruning script; if irregular, blocked on closed engine |
| Closed engine asserts/derives 40 layers somewhere | Low (config-driven strings) | Phase 1 detects cheaply |
| q4nx I8 packing subtleties (row/col blocking, parallel_size) differ for this family | Medium | Phase 2 gold-standard diff against official q4nx |
| Engine requires vision weights present | Medium | Reuse official vision_weight.q4nx |
| Double-quantization quality loss (Q4_K_M→Q4_1) | Medium | Use healed BF16 safetensors as source if available |

## Inputs needed from Cyrus
1. Was the layer removal uniform (exactly one DeltaNet layer per `[L,L,L,F]`
   block)? Which indices were dropped?
2. Do the healed BF16 safetensors still exist locally? (Better conversion source
   than the Q4_K_M GGUF.)
