# New-model feasibility, 2026-10-14 (rev 7)

**Weights rule (Cyrus, 2026-10-15): GGUF and safetensors both load, for any model.** Do not touch q4nx:
no `q4nx-build`, no `model.q4nx` reader work, no new `-NPU2` repos; the existing q4nx paths are legacy,
kept as they are. The file type picks the path:

- **GGUF:** the engine keeps the quant the file ships where its pools support it (Q4_0 and Q4_1 pack
  exactly) and requantizes to q4_1 where they do not (Q8_0, Q4_K, Q6_K). A Q4_1 file is never turned
  into Q4_0 (feedback-respect-gguf-quant-format).
- **Safetensors, small non-LLM engines** (embedding, ASR, TTS, drafters): loaded in 16-bit, as
  `open_embedding` and `open_whisper` do today.
- **Safetensors, every other model** (LLMs, and large engines): the 16-bit weights are quantized at load
  to the engine's native block format (q4_1, with the q8 roles where a family has them), because 16-bit
  LLM weights break the memory rule below. This needs one shared quantizer in C++ plus a local cache of
  the packed result (derived data on the user's disk, keyed by source hash and layout hash, never
  published). It is new work; today only `recipes/pack.py` and `q4nx-build` quantize.

Where a row below says "safetensors" or "container" for an LLM, read it as "GGUF, or safetensors once
the quantizer exists". Part 3 lists what the rule changes.

Three lists in this file. Part 1 (below): Hugging Face targets outside the FLM catalogue. Part 2 (at the
end): gaps in the FastFlowLM catalogue, ranked separately.

# Part 1. Hugging Face targets

Hugging Face targets (the table below) plus one drafter, checked against `upstream/main` at 1fcb7e3d.
Local `main` is 5 commits behind and lacks GGUF-direct (#180); rev 1 was written from it.

**Constraint (Cyrus):** weights load straight from the file the model ships in, GGUF where one
exists. Nothing is converted into an `*-NPU2` repo and nothing goes through `q4nx-build`. Kernel
sets still ship per family with the release; that is not a weight conversion.

**Memory rule (Cyrus, 2026-10-13):** the NPU is limited in size and in speed, so smaller is
better everywhere. **Never f32 in device memory** (pools, weights, scales). **bf16 is already
too big for weights**: keep the shipped 4/8-bit blocks resident and widen to bf16 only inside a
kernel or a transient buffer, if at all. Host-side transients (one dequantized embedding row,
scratch during a pack) and in-register accumulation are not device residency and are fine.

**Spec impact:** none yet. Each row becomes its own plan under `specs/open-engine/plans/`.

## Summary

Difficulty order, easiest first. "Track" says which shared piece of work the model rides on.

| # | Model | Track | Out of the box? | Main missing piece |
|---|---|---|---|---|
| 0 | Qwen3-Embedding-0.6B / 4B / 8B, Qwen3-Reranker-0.6B / 4B / 8B (detail: `qwen3-embedding-reranker.md`) | L: LLM as embedder/reranker | Shape yes, job no | dense qwen3 GGUF-direct already works; needs a pooled-output mode and a rerank route |
| 1 | MiniCPM-V-4.7-1B (text) | A: qwen35 GGUF-direct | No | qwen35 family has no GGUF route |
| 1b | Qwen3.5-Embedding-0.8B (Rebine), Prism-Qwen3.5-Reranker-0.8B (infgrad) | A + L | No | track A, then the same job layer as row 0 |
| 2 | Qwen3.8-3.6-27B-blend | A | No | same, plus a GGUF must be made |
| 3 | EmbeddingGemma-2 (text) | E: embedding engine | No | new text tower in the CPU engine; no GGUF exists |
| 3b | Moonshine (v2 streaming tiny / small / medium, v1 tiny / base) | T: speech | No | an ASR engine that is not Whisper; raw-audio frontend |
| 4 | MiniCPM-V-4.7-35B-A3B (text) | B: MoE GGUF-direct | No | expert ops, q8 head; kernels already exist |
| 4b | Qwen3-Coder-30B-A3B-Instruct (also unlocks Qwen3-30B-A3B and its 2507 variants) | B: MoE GGUF-direct + MoE without shared expert | No | a `qwen3moe` recipe and three new MoE kernel points; dims otherwise aligned |
| 4c | Qwen3-Coder-Next (80B-A3B) | B, much harder | No | 512 experts overflow the per-layer pool; ~50 GB resident |
| 4d | Parakeet (tdt-0.6b-v3 / v2, ctc, rnnt) | T: speech | No | FastConformer encoder, transducer decode, new GEMM shapes |
| 4e | Muse Glimmer-30B (text) | G: dense GGUF-direct, new family | No | a `muse_glimmer` spec and GGUF names; three dense-recipe features composed for the first time; one new kernel point |
| 5 | Z-Image-Turbo (+ base) | D: diffusion | No | new DiT plan, load-time packer |
| 5b | Kokoro-82M (TTS) | T: speech, new endpoint | No | no TTS in the tree: `/v1/audio/speech`, a phonemizer, LSTM and iSTFT-conv ops |
| 6 | Mellum2.1-12B-A2.5B-Thinking | B + new family | No | all of 4b, plus sliding-window layers, YaRN and a tile-misaligned expert ff |
| 7 | Iris-3B Turbo | D | No | novel pixel-space net, NPU-hostile head |
| 8 | Qwen-Image-2.1 + uncensored GGUF | D | No | new DiT, 8B VL encoder, new VAE |
| 9 | SDXL-Lightning | UNet | No | a UNet engine |
| 4f | ALoDLM-8B (Amazon, looped diffusion LM on Qwen3-8B; detail: track R) | R: looped dLLM | Shape yes, schedule no | layer-range recurrence with per-depth KV, a gate, an entropy commit policy; weights are Qwen3-8B's |
| - | MiniCPM-V vision tower | V | No | a new host ViT (not Qwen's) |
| - | MiniCPM-V-4.7-35B-A3B-DSpark | S: spec decode | **No** | not supported; see Track S |

Nothing is out of the box. #1, #2 and #4 are the closest, and the missing piece in each is shared
infrastructure that every Qwen3.5/3.6/3.8 GGUF needs, so those models are the payoff of the work.

## Foundation: what GGUF-direct (#180) gives us today

From `.opencode/skill/open-gguf-direct/SKILL.md`, `gguf_file.cpp`, `pools.cpp`, `recipes/dense.py`
and the oflm-add installer on upstream/main:

- `model.gguf` is read by the engine (`GgufFile`, behind the `WeightFile` seam). Q4_0 and Q4_1 pack
  byte-exact into the q4 pool; Q8_0, Q4_K and Q6_K are requantized to q4_1 at load (validated on
  Qwen3-8B, all four quants). Config derives from GGUF metadata.
- `oflm add` installs one compatible GGUF as a `*-GGUF` directory and links the family kernel set,
  but refuses unless that set's manifest carries the nested `gguf` section
  (`manifest_supports_gguf`). Only the **dense** recipe emits it (`dense.py:649`, f32-scale `*_f32`
  kernel twins). Dense families only: qwen3, llama, gemma3, hunyuan, granite and kin.
- Not covered, found by grep on upstream/main (not built or run):
  - `qwen35.py` has no `gguf`/`f32` mention; `qwen36moe.py` has `q4_1_f32` pack-op plumbing but no
    kernel twins.
  - `GgufFile::gguf_name` maps llama-style `attn_*`/`ffn_*` only: no `ssm_*`/`attn_qkv`/`attn_gate`
    (DeltaNet), no 3D `ffn_*_exps` (experts), no shared-expert or router names.
  - `q8_perm` needs a q4nx container, so a q8 role (the 35B's) cannot come from GGUF.
  - `lm_head_q8` f32-scale numerics bug (maxrel ~5e-3 vs 4.5e-6); the skill calls it the blocker
    for the MoE family. Every qwen35/qwen36moe model uses that head.
  - Block prefill is off for GGUF pools (decode-as-prefill instead). Vision weights are read from
    `vision_weight.q4nx`. `layer_x/gen_kernels.py` emits unprefixed kernel names, so f32 twins fail
    to link there.
- **Decided by the memory rule: no f32 twins.** `specs/open-engine/plans/gguf-scale-width.md`
  left it open ("keep the f32 twins, or narrow scales to bf16 at pack time"). Narrow at pack
  time: the 5120-byte chunk, one build, no `*_f32` kernels, no nested `gguf` manifest. That also
  deletes the `lm_head_q8` f32 numerics bug and the layer_x link landmine, and saves ~4 GB on a
  22 GB model (the f32 pool is +20% on q4, +6% on the q8 head). Narrowing loses nothing: fp16 ->
  bf16 is one rounding either way (the plan's exhaustive check over all finite fp16 values).
  Catch: #180 already merged the f32 path for the dense families, so new families (qwen35,
  qwen36moe, mellum) take the 5120 chunk and the dense families should migrate in a follow-up so
  the double export goes away. The plan's one caveat applies: derive requantized codes with
  `requant_q4_1_chunks`' law (m toward -inf, d toward +inf), not round-to-nearest. Atomic-Germ
  wrote #180, so tell them before reversing it; ask Cyrus before posting.
- **Spec-hash finding.** `ModelSpec.spec_hash()` covers every field except `extra`, including
  `real_vocab`, and `oflm add` links a kernel set by that hash. Qwen3.6-35B is 248070; the
  MiniCPM-V GGUFs carry 248094 (1B) and 248144 (35B). By hash they would not link to the existing
  sets even though the padded lm_head (248320) and every kernel are identical. Likely fix: leave
  `real_vocab` out of the kernel-match hash. Unverified; needs the maintainer.
- Nothing GGUF exists on the diffusion, embedding-multimodal or speculative sides.

## Track A. Qwen3.5-family GGUF-direct

Work, in order:

1. `std_perm_gguf` targets the 5120-byte bf16-scale chunk (the "decided" bullet above). The block
   prefill route reads that layout already, so it comes back for GGUF models for free.
2. The qwen35 recipe reads GGUF pools through its existing q4_1 programs; no twins, no nested
   manifest.
3. `gguf_name` for the DeltaNet tensors plus the inverse of llama.cpp's conversions. llama.cpp
   reorders the value heads and stores A/dt in derived form; `q4nx-build`'s `models/qwen35.py` already documents these (read-only reference, see #142), and the engine consumes the HF
   layout.
4. A GGUF q8 head pack at the 8704-byte chunk (the f32 head bug goes away with the f32 layout).
5. `oflm-test --llm` through `oflm serve`.

### 1. MiniCPM-V-4.7-1B (text path)

Repo `openbmb/MiniCPM-V-4.7-1B-gguf`, Apache 2.0. Header read by range request from the Q4_1 file.

- GGUF arch `qwen35`: 24 layers, hidden 1024, ff 3584, 8 heads over 2 kv at head dim 256, rotary
  dim 64 (mrope sections 11/11/10/0), theta 1e7, DeltaNet with 16 key heads, 16 value heads,
  conv 4, state 128, full attention every 4th layer. Context 262144. This is the Qwen3.5-0.8B
  shape the catalogue already validates (attention tuple (256, 8, 2, 64), h1024 builds).
- Quants: Q4_0 644 MB, Q4_1 691 MB (packs exactly), Q4_K_S/M, Q5_*, Q6_K, Q8_0 1.08 GB, F16 2.0 GB.
- Best bring-up vehicle for track A: under 1 GB, kernels exist, seconds per iteration.
- Vocab is 248094 ids; see the spec-hash finding.
- The "V" is the vision tower; see track V. Text-only ships first or the registry entry must not
  claim vision.

### 2. Qwen3.8-3.6-27B-blend

A 50/50 merge of Qwen3.6-27B and Qwen3.8-27B with Qwen3.8's config. Shape equals
`recipes/specs/qwen35-27b.json` field for field (64 layers, hidden 5120, 24/4 heads at 256, ff
17408, vocab 248320), so the 27B kernel set applies. The repo is BF16 safetensors, no GGUF
children; a GGUF has to be made with llama.cpp's converter (`qwen3_5`; third parties publish
Qwen3.8-27B GGUFs). Vision is the Qwen 27-layer ViT the engine already handles, but from a separate
`mmproj` GGUF rather than `vision_weight.q4nx`. Work beyond track A: the 27B's q8 head and mixed
core program-memory limits (`MIXED_CORE_FITS`, `ROLLED_BANDS` in catalogue.py).

## Track B. MoE GGUF-direct

### 4. MiniCPM-V-4.7-35B-A3B (text path)

Repo `openbmb/MiniCPM-V-4.7-35B-A3B-gguf`, Apache 2.0. Header read from the Q4_1 file.

- GGUF arch `qwen35moe`: 40 layers, hidden 2048, 16/2 heads at 256, 256 experts top 8, expert ff
  512, shared expert ff 512, DeltaNet 16 key / 32 value heads. Identical to
  `recipes/specs/qwen36-35b-a3b.json` except the vocab. **The existing 35B kernels apply unchanged;**
  no new kernel point, no new recipe. The backbone is Qwen3.6, as the card says.
- Tensors are standard llama.cpp names: `attn_qkv`, `attn_gate`, `ssm_a/alpha/beta/conv1d/dt/norm/out`,
  3D `ffn_{gate,up,down}_exps` [n_embd, 512, 256], `ffn_*_shexp`, `ffn_gate_inp`, `ffn_gate_inp_shexp`.
  In the Q4_1 file the experts and embedding are Q4_1 (exact pack) and `output.weight` is Q6_K.
- Quants: Q4_0 19.7 GB, Q4_1 21.8 GB, Q4_K_S/M 19.9/21.2, Q5_*, Q6_K 28.5, Q8_0 36.9, F16 69.4, and
  a `mix_q3gud_im` mixed file with a tensor-types sidecar (refuse). Q4_1 is the exact-pack choice.
  Keep ~25 GB free (engine holds ~21.6 GB of NPU buffers).
- Work: everything in track A that applies, plus GGUF forms of `expert_stripes`/`expert_down` (3D
  tensors), the router and shared-expert names, a q8 head
  pack from GGUF at the bf16-scale chunk. Existing validated kernels make this the smallest step that proves
  MoE-from-GGUF.
- Why it matters: it is the flagship family, and Mellum needs the same ops.

## Track E. Embeddings

### 3. EmbeddingGemma-2 (text path)

`google/embeddinggemma-2`: Apache 2.0, not gated, safetensors BF16 only, **no GGUF** and no
sentence-level GGUF arch in llama.cpp that I could confirm. Not a bigger EmbeddingGemma 1; a new
architecture (`embedding_gemma2`, transformers 5.18 dev), 740M total with a multimodal front end.

- Text tower: 24 layers, hidden 512, 4 heads over 2 kv at head dim 256; layers 5, 11, 17, 23 are
  full attention with head dim 512 and 1 kv head, the other 20 are sliding (window 512); gelu-tanh
  FFN ff 2048; `hidden_size_per_layer_input` 512 (per-layer inputs, as in Gemma 3n/4); vocab 262144;
  mean pooling then a 512 -> 768 projection; Matryoshka 768/512/256/128; context 8192.
- Vision encoder (Gemma 4 vision, 16 layers, 768) and audio encoder (12-layer conformer, 1024):
  out of scope for a text embedding release. The headline feature of the model is lost if they are
  skipped; say so in the registry.
- The existing engine (`src/open_embedding`) is a CPU forward that mirrors `Gemma3TextModel`, with
  an NPU bf16 matmul option, reading safetensors plus a `weights_manifest.json`. It never touches
  q4nx, so the no-conversion rule is already met. What is missing is the new tower: heterogeneous
  head dims per layer, per-layer inputs, new pooling and projection. `q4nx-build`'s `gemma4.py`
  already documents Gemma 4's per-layer-embedding tensor layout, useful as a read-only reference.
- Oracle: HF transformers 5.18 (dev). The task prefixes are listed on the card.
- Low hardware risk (CPU first), so it ranks above the MoE and diffusion work despite being a new
  architecture.

## Track L. Embedders and rerankers built on LLM backbones

Four community and official models, none supported. I took "qwen 0.8 embedder and reranker" to
mean the two Qwen3.5-0.8B fine-tunes below; there is no Qwen-org 0.8B embedder, the official ones
are the 0.6B Qwen3 pair (searched the Qwen org on the hub, 2026-10-13). Correct me if you meant
others.

What exists today: the embeddings endpoint is served by a separate class, `AutoEmbeddingModel`
(`--embeddingmodel`), with seven registry entries, all BERT-family encoders on the NPU (bge,
all-MiniLM, nomic, gte) plus EmbeddingGemma-300M on the CPU engine. **There is no reranker
anywhere in the tree and no `/v1/rerank`** (grep for `rerank` finds only a binary). The open LLM
engine (`open_qwen36`) returns logits; it has no pooled or hidden-state output. The DFlash2 branch
added residual taps for spec decode, which is the nearest reusable hook (local, unpushed).

Two missing layers, both shared by all four models:

1. **Embedding mode.** Prefill the text, take the last token's final-norm hidden state, L2
   normalize, optionally truncate (Matryoshka). Needs a hidden-state output from the engine, no lm
   head, routing so an LLM-family model can be an `--embeddingmodel`, and a registry label. Prefill
   must be fast: with f32-twin pools the block prefill route is off for GGUF, which the 5120-byte
   chunk decision brings back.
2. **Rerank route.** Score is sigma(logit_yes - logit_no) at the first generated token after a
   fixed ChatML prompt (query, document, instruction). Only two logit rows are needed, so the
   head can be two rows instead of 248320. A `/v1/rerank` (Cohere/Voyage-style) endpoint has to be
   added to the server spec; none exists.

### 0. Qwen3-Embedding-8B, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B (official)

**Qwen3-Embedding-8B** (`Qwen/Qwen3-Embedding-8B`, Apache 2.0, from Qwen3-8B-Base): config read
from the hub. `Qwen3ForCausalLM`, 36 layers, hidden 4096, ff 12288, 32/8 heads at 128, qk-norm,
theta 1e6, untied, vocab 151665, 7.57B parameters (Qwen3-8B minus its 0.6B lm_head, so the head is
absent), context 32K, output up to 4096 dims with Matryoshka down to 32.

- Official GGUFs exist (`Qwen/Qwen3-Embedding-8B-GGUF`: Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0, f16).
- **Shape-identical to Qwen3-8B**, which upstream/main already has a recipe spec for
  (`recipes/specs/qwen3-8b.json`), kernels exported for, and runs end to end from GGUF in all four
  quants (Q4_1, Q4_0, Q8_0, Q4_K_M; open-gguf-direct skill, 2026-10-07). The 8B kernel point
  (`gemv_q4` K = 12288, `PER_CALL` 1) is validated. So the weights path works today; the head is
  simply not used, and the job layers are what is missing. About 4.5 GB resident at q4_1.
- Embedding is prefill-bound (whole passages, no decode), so this is where the dense prefill work
  pays: #170 and the 8-bit dit_gemm prefill (#193, Qwen3-8B 981 tokens in 4.7 s). The GGUF pool's
  block prefill route is off until the 5120-byte chunk lands; without it prefill is
  decode-as-prefill, which would make this model unusably slow.
- Vocab 151665 vs the 8B spec's 151936: the `real_vocab` hash finding applies (kernels identical).
- Q4_K_M, the file most people take, requantizes to q4_1 on load; check vector cosine against F16.

**Qwen3-Embedding-0.6B**

- Official GGUF (`Qwen/Qwen3-Embedding-0.6B-GGUF`: Q8_0 639 MB, F16 1.2 GB; header read from the
  Q8_0 file): arch `qwen3`, 28 layers, hidden 1024, ff 3072, 16/8 heads at 128, qk-norm, theta 1e6,
  vocab 151669, `pooling_type = 3` (last), **no `output.weight`**, Apache 2.0, context 32768.
- This is Qwen3-0.6B exactly, whose points the catalogue already validates (hidden 1024, ff 3072,
  attention tuple (128, 16, 8, 128, qk_norm)), and dense-family GGUF-direct works on upstream/main
  today. So the *weights* path is already there; the work is the two layers above. Smallest
  possible embedder/reranker job in this document, and the right first test of both.
- Query instruction format: `Instruct: ... Query: ...`; documents take none. Output dim 32..1024.
- 4B of both exist (Qwen3-Embedding-4B has official GGUFs; the 4B recipe spec exists).
- Q8_0 requantizes to q4_1 on load. For an embedder, check cosine against the F16 vectors before
  accepting that: retrieval quality is more sensitive to weight error than a chat model.

**Qwen3-Reranker-0.6B** (`Qwen/Qwen3-Reranker-0.6B`, Apache 2.0): official repo is safetensors
only, no official GGUF. Community GGUFs exist (mradermacher, DevQuasar, QuantFactory, Mungert:
Q4_K_M, Q8_0, f16, ...); the two I listed carry arch `qwen3`, context 40960, 596M parameters, i.e.
the plain Qwen3-0.6B layout with the tied head and no cls/score tensors (inferred from the
parameter count and arch; I did not read their tensor lists).

- Prompt (card): system "Judge whether the Document meets the requirements based on the Query and
  the Instruct provided. Note that the answer can only be \"yes\" or \"no\".", then a user turn
  `<Instruct>: ...`, `<Query>: ...`, `<Document>: ...` on separate lines, then the suffix
  `<|im_end|>`, `<|im_start|>assistant`, and an empty `<think></think>` block.
  Score is softmax over the final token's "yes" and "no" logits. Length up to 8192; default
  instruction "Given a web search query, retrieve relevant passages that answer the query".
- The tied head makes the rerank route cheap: the two logits are two rows of `token_embd`, so no
  head GEMV at all. Query and instruction are shared across a batch of documents, so a prefix KV
  cache over the common head of the prompt saves most of the work.
- Needs a `qwen3-0.6b.json` recipe spec and a kernel export; the catalogue already validates the
  0.6B points, but upstream/main ships specs only for the 4B and 8B.

### 1b. Qwen3.5-Embedding-0.8B (Rebine) and Prism-Qwen3.5-Reranker-0.8B (infgrad)

- Embedder: `Rebine/Qwen3.5-Embedding-0.8B`, fine-tuned from Qwen3.5-0.8B-Base, Apache 2.0, 752M
  parameters, 24 layers (18 linear, 6 full), hidden 1024, vocab 248320, last-token pooling, L2
  normalize, 1024 dims with Matryoshka 128..1024, context 2048. A `-GGUF` repo has F16 (1.52 GB)
  and Q8_0 (812 MB). A `-Memory` variant (agent-memory retrieval) exists too.
- Reranker: `infgrad/Prism-Qwen3.5-Reranker-0.8B`, MIT, causal LM, no score head; yes/no logits,
  token ids 9405 / 2083 per the GGUF card, ChatML prompt ending in an empty `<think>` block.
  `grimmjoww578/...-GGUF` has Q4_K_M (~505 MB) and F16. The fine-tune dropped the MTP head, so the
  converter needs `mtp_num_hidden_layers: 0`. Sizes 2B, 4B, 9B exist.
- Both are the Qwen3.5-0.8B shape that MiniCPM-V-4.7-1B uses, so they ride track A first.
- Smallest models on the NPU in this list. Also the most useful to ship: they are the retrieval
  stack that goes with an on-device assistant.

## Qwen3-Coder

Three models carry the name; only one is a realistic target.

### 4b. Qwen3-Coder-30B-A3B-Instruct (realistic)

`Qwen/Qwen3-Coder-30B-A3B-Instruct`, Apache 2.0, July 2025, `Qwen3MoeForCausalLM`. Config read from
the hub: 48 layers, hidden 2048, 32 heads over 4 kv at head dim 128 (qk-norm), **128 experts, top 8,
expert ff 768, no shared expert**, `norm_topk_prob`, full attention in every layer (no DeltaNet, no
sliding window), theta 1e7, vocab 151936, untied head, context 262144, about 30.5B parameters
(roughly 19 GB at q4_1, the same class as the 35B's 21.6 GB). No official GGUF (the Qwen org has
only an FP8 repo); community GGUFs are plentiful (arch `qwen3moe`). Its tool calls use the
`<tool_call><function=...>` XML form, which the repo's `chat_templates/qwen3.x` template already
emits; whether the server's parser reads it back is unchecked.

- **Same shape as Qwen3-30B-A3B** and its Instruct-2507 / Thinking-2507 releases, so one recipe
  covers a whole family and its fine-tunes. Better payoff per kernel point than any other MoE here.
- Hidden 2048 is the validated MoE hidden, and expert ff 768 is 3 tiles of 256, but it is 6 stripes of 128
  rows and the recipe needs the stripe count to divide the 8 cores, so a new MoE host sequence is needed
  (`qwen36moe.py:561-568`, OPEN-MOE-WIDE-FF; the same wall as gpt-oss; found 2026-10-17 by the memory-budget
  survey, `.claude/plans/round3-prep-9d-C.md`). What is new against the validated MoE point (ff 512, 256 experts, shared expert):
  - 128 experts (`router`, `moe` templates take 256 only);
  - ff 768 (`moe`, `gemv_q4_prep_f32` take 512 only; the down projection contracts over 768);
  - no shared expert (`qwen36moe.py:382` refuses);
  - the attention tuple (128, 32, 4, 128, qk_norm): a GQA group of 8 at head dim 128 is unvalidated;
  - a layer plan with full attention and MoE in *every* layer, i.e. only the `ax` image.
- No new op: SwiGLU experts, softmax router with renormalised top-8, RMSNorm, RoPE, qk-norm.
  Reference is HF transformers `Qwen3MoeForCausalLM` and llama.cpp (supported).
- Needs track B (3D expert tensors from GGUF, router names, q8 head pack at the bf16-scale chunk).
- It is the MoE-without-shared-expert work Mellum also needs, minus Mellum's sliding window, YaRN
  and tile-misaligned ff 896. Do this before Mellum.
- Context: KV is 48 layers x 2 x 4 kv heads x 128 x 2 B = 96 KB a token at bf16, so 32K tokens is
  3 GB. Small enough for a coder's long context; the engine's own window limits are what to check.

### 4c. Qwen3-Coder-Next (80B-A3B, hard)

`Qwen/Qwen3-Coder-Next`, Feb 2026, `Qwen3NextForCausalLM`; official GGUF repo exists. Config: 48
layers (every 4th full attention, the rest Gated DeltaNet with 16 key / 32 value heads, same as the
35B), hidden 2048, 16/2 heads at 256, rotary 0.25, **512 experts, top 10**, expert ff 512, shared
expert 512, theta 5e6. `qwen3next` is already in `GGUF_FAMILIES` (spec.py:1463), so spec derivation
anticipated it. Against the 35B everything matches except layers (48 vs 40), experts (512 vs 256)
and top-k (10 vs 8).

- One layer's expert pool is 512 experts x 1.97 MB = about 1 GB at q4_1, but `POOL_BYTES` is a
  fixed 512 MB buffer sized for the 35B's 256 experts (`qwen36moe.py`). A bigger pool, or splitting
  experts across buffers, changes the layer design. `router` and `moe` templates take 256 and 8 only.
- About 50 GB of weights resident (80B at q4_1). The 35B needs ~25 GB free beyond its 21.6 GB; this
  needs a 64 GB box and will not run on 32 GB. Under the "smaller is better" rule it wants a
  3-bit pool that does not exist.
- Defer. The only change that makes it cheap is a deliberate pool redesign for more experts.

### Not a target

Qwen3-Coder-480B-A35B-Instruct: far beyond any NPU box. The older dense Qwen2.5-Coder line
(0.5B to 32B, official GGUFs) is the `qwen2` family; the 3B is validated, other sizes would each
need a spec point.

## Track G. Muse Glimmer-30B (Meta)

`meta-models/Muse-Glimmer-30B`, Apache 2.0, August 2026, `MuseGlimmerForConditionalGeneration`.
A dense vision-language model: 28B text decoder, 2B vision encoder. Config read from the hub
(`config.json`) plus the Hugging Face blog; neither was cross-checked against a GGUF header.
Official GGUFs exist (`meta-models/Muse-Glimmer-30B-GGUF`, Q4_K_M is the one Meta's workflow uses,
plus `mmproj` files), llama.cpp supports it from day 0, and a DFlash drafter ships separately
(`Muse-Glimmer-30B-assistant`, block of 16).

**Text decoder:** 52 layers, hidden 6656, ff 19968, 32 heads over 2 kv at head dim 128 (a GQA
group of 16), qk-norm plus an extra query scale, gated attention, vocab 202048, untied head, RoPE
theta 500000, final logit softcap 20.0, context 131072. Three sliding-window layers (2048) with
RoPE, then one full-attention layer with **no** positional embedding, repeated 13 times. The
parameter count only works out with an untied head (about 25 B in layers + 2.7 B in two 202k x 6656
tables).

### What already exists in the dense recipe

- Sliding-window layers with their own RoPE table: `dense_local` (dense.py:13, built for Gemma 3's
  1024 window). Window 2048 is a parameter, not a new kernel.
- The attention output gate (`attn_gate`, dense.py:194), used today by the Qwen3.5 family.
- qk-norm; untied q8 head; dense GGUF-direct (arch handling in `GGUF_FAMILIES`).
- Dims are tile-aligned: 6656 = 26 x 256, 19968 = 78 x 256. No misaligned-ff problem like Mellum's.

### What is new

1. **A `muse_glimmer` spec + GGUF names.** Tensor names are unverified (llama.cpp's converter is the
   authority); expect llama-style `attn_*`/`ffn_*` plus a gate tensor. Config derives from GGUF
   metadata as for other families.
2. **The three features composed.** No validated family has sliding window, output gate, qk-norm
   and GQA group 16 together. The attention tuple (128, 32, 2, 128, qk_norm, attn_gate) is a new
   point: the closest validated ones are the 27B (24/4 at 256, group 6) and Gemma 3. Sliding-window
   and gated layers are both excluded from the block-prefill route (dense.py:420, 431), so prefill
   runs on the slower dxB route, not the 8-bit GEMM one (#193).
3. **NoPE on the full-attention layers.** Probably an identity rotation in the position table (cos 1,
   sin 0) rather than a new kernel; confirm the rotary_dim 0 path in the attention design.
4. **Extra query scale and logit softcap.** The query scale folds into the q-norm weight at pack
   time if it is a per-layer constant (the card does not say). The softcap is a tanh on 202k logits
   on the host after the head, off the device; greedy decode is unaffected, sampling is not.
5. **New kernel point at hidden 6656 / ff 19968.** The 27B's ff is 17408 (68 tiles), Glimmer's is 78.
   The 27B already needed `MIXED_CORE_FITS` and `ROLLED_BANDS` to fit 16 KB program memory, so the
   longer contraction is the hardware risk: it may not fit without a new band split. This is the one
   thing that can sink the plan; measure it first.

### Size and speed

- Weights ~17.5 GB at q4_1 (28B x 5 bits), under the 21.6 GB the 35B holds, so the 25 GB free rule
  covers it. q8 head 202240 x 6656 = 1.4 GB.
- KV is small. Full-attention layers cost 1 KB per token each, 13 KB a token: 128K context is 1.7 GB.
  The 39 sliding layers are bounded at ~80 MB.
- Decode is memory-bound: a dense 28B reads all weights per token. Our dense Qwen3.8-27B decodes at
  2.2 tok/s; expect about the same here, ~2 tok/s. That is usable for batch and agent work and slow
  for chat, and it is a quarter of what the A3B MoEs get.
- Official Q4_K_M requantizes to q4_1 on load; check NLL against F16 before accepting it.

### Vision and the drafter

- The 2B vision tower is a new host ViT: 50 layers, width 1536, 16 heads, ff 8960, patch 14 with a
  2-frame temporal patch, three window layers then one full, 2D RoPE, pixel-shuffle projector
  (4096 -> 6144), video at 2 fps up to 96 frames. Not Qwen's and not MiniCPM-V's; same status as
  track V. Text ships first; the registry must not claim vision.
- The DFlash drafter rides track S (DFlash2 is unpushed and stacked on #201). Not near-term, and a
  drafter pays off most on exactly this kind of slow dense target, so it is the best later win here.

### Priority

Medium, behind the shared infrastructure. Reasons:

- It cannot start until the pool-layout decision lands (bf16-scale 5120 chunk vs the f32 twins), since
  it is a new family and the memory rule says no f32. That is PR A's step 1.
- Payoff is one model at ~2 tok/s. The same engineering effort on the A3B MoEs (Coder-30B,
  MiniCPM-V-35B) gets ~7 tok/s and a whole family each, so those go first.
- It is cheaper than Mellum (no MoE, aligned dims, no YaRN) and than any new engine, and it
  exercises the sliding-window + gate composition that other Meta and Google families will reuse.
- Cheap first move, before committing: build the one new attention point and the hidden-6656
  gemv/down kernels in the catalogue and see whether they fit program memory. Hours, no new loader.

## Track T. Speech

What exists: one ASR path, built around Whisper. `--asr 1` loads a single speech model;
`POST /v1/audio/transcriptions` decodes the upload with ffmpeg to 16 kHz mono s16le
(`load_audio`), then calls `Whisper::generate` (`src/common/whisper/`: `modeling_whisper.cpp`,
`generation_hf.cpp` for the token protocol with language and timestamp tokens). The engine
behind it is a `whisper_engine` (`encode_audio(mel)`, `decode_audio(last_id) -> logits`,
`clear_context`), with the open implementation in `src/open_whisper`: encoder GEMMs on the NPU
through the `whisper_gemm` kernel set (one xclbin, one instruction stream per shape), everything
else on the host, decoder entirely on the host, weights bf16 from `model.open.safetensors`.
**No TTS exists anywhere**: no engine, no `/v1/audio/speech`, no speech code in `src/server`.

The shared prerequisite for Moonshine and Parakeet is an engine-neutral ASR interface: audio in,
text (and optional segments) out. The `whisper_engine` interface takes a mel spectrogram and a
last token id and returns vocabulary logits; Moonshine takes raw samples and Parakeet decodes a
transducer, so neither fits it. The handler's `load_audio` and the ffmpeg decode are reusable.

### 3b. Moonshine

Useful Sensors, MIT, English. Two generations.

- **v2 streaming** (`UsefulSensors/moonshine-streaming-{tiny,small,medium}`, transformers classes
  `MoonshineStreamingForConditionalGeneration`): tiny 34M (6 enc / 6 dec layers, dim 320), small
  123M (10/10, 620/512), medium 245M (14/14, 768/640). Frontend: 50 Hz features from time-domain
  ops, CMVN and two causal stride-2 convs (no mel, no FFT). Encoder: sliding-window self-attention,
  windows (16,4) on the first and last two layers and (16,0) between, no positional embeddings
  (an adapter adds them); decoder: causal transformer with RoPE. Weights safetensors F32.
- **v1** (`UsefulSensors/moonshine-tiny` / `-base`, about 27M and 61M): the same family with a
  conv frontend on raw waveform and a RoPE encoder-decoder. Sizes are from search results only;
  the config was not read.
- GGUF: only a community one, `myflow-ai/moonshine-streaming-tiny-gguf` (arch `moonshine_streaming`,
  F32 169 MB, F16 85 MB, Q8_0 48 MB, runtime transcribe.cpp), tiny only, tensor names undocumented.
  Official weights are safetensors; for small and medium, safetensors is the only source.
- NPU fit: these are tiny networks (hidden 320 to 768); the GEMMs are well below what the
  whisper_gemm tiles are built for, and a 34M model runs far faster than real time on the CPU.
  Start host-only; the case for the NPU is power, not speed. Encoder cost grows with audio length
  (no 30 s padding), so the fixed-shape instruction streams do not apply as they do for Whisper.
- Work: ASR interface; raw-audio frontend and CMVN; sliding-window attention with the lookahead
  masks; the decoder with RoPE and its tokenizer (not Whisper's); a reference check against
  transformers on LibriSpeech clips (WER within noise of the card's 4.5% for tiny).
- The "streaming" part (chunked, incremental encoding) is not exposed by the card's API and the
  OpenAI transcription endpoint does not stream audio in; one-shot is the realistic scope.

### 4d. Parakeet (NVIDIA)

Family of FastConformer ASR models, `nvidia/parakeet-*`: tdt-0.6b-v3 (Aug 2025, 25 European
languages, language auto-detect), tdt-0.6b-v2 (English), ctc-0.6b / ctc-1.1b, rnnt-0.6b / 1.1b,
tdt-1.1b, tdt_ctc-110m, unified-en-0.6b (Apr 2026), realtime_eou_120m, multitalker-streaming-0.6b.
CC-BY-4.0, 16 kHz mono, word, segment and char timestamps, up to 24 min per pass with full
attention.

- The v3 card reports safetensors **and a GGUF** (`parakeet-tdt-0.6b-v3.q8_0.gguf`, for
  NeMo-Speech.cpp). That makes Parakeet the one speech model here that fits the GGUF rule.
- Architecture (from NeMo's published configs, not re-read for this note): FastConformer encoder,
  24 layers, d_model 1024, 8 heads, ff 4096, 8x depthwise-conv subsampling, relative-position
  attention, conv modules (pointwise GLU, depthwise conv kernel 9, batch norm folded at inference);
  TDT decoder = a 2-layer LSTM prediction network plus a joint network that predicts a token and a
  duration. About 600M parameters, 1.2 GB at bf16.
- Maps well onto the open_whisper pattern: the big matrices (attention projections, ff, pointwise
  convs) are GEMMs for the NPU; attention with relative-position bias, the depthwise conv, layer
  norms and the LSTM and joint decode loop run on the host, as Whisper's attention already does.
  New GEMM shapes mean new instruction streams in the whisper_gemm set (d_model 1024, ff 4096),
  not new kernels.
- New work: the NeMo mel frontend (its own normalization, not Whisper's); relative-position
  attention; the transducer decode loop with durations (host, 640-wide, cheap per step but serial);
  the ASR interface; a GGUF reader for NeMo-Speech.cpp's tensor layout; a NeMo reference.
- Multilingual v3 needs its SentencePiece vocabulary and detokenization.
- Under the memory rule: bf16 weights are what open_whisper uses today (1.2 GB). A Q8_0 GGUF could
  stay 8-bit if the GEMM stream took 8-bit operands; not required for a first version.

### 5b. Kokoro-82M (text to speech)

`hexgrad/Kokoro-82M`, Apache 2.0, v1.0 (Jan 2025): 82M parameters, 24 kHz, 8 languages, 54 voices.
StyleTTS 2 with an ISTFTNet vocoder, decoder-only (no diffusion).

- **No GGUF anywhere.** Formats are the torch checkpoint, 13 ONNX variants (fp32, fp16, int8) and
  MLX safetensors conversions. The GGUF rule cannot be met; the realistic source is safetensors
  (the MLX bf16 conversion) or ONNX. The original has weight-norm parametrizations that must be
  folded. Voices are small style-vector tables, not weights as such.
- Components (from the StyleTTS 2 / Kokoro design; not re-read from the code): a PL-BERT text
  encoder (ALBERT-style, shared layers, hidden 768), a text encoder (convs + BiLSTM), a duration
  predictor (BiLSTM with adaptive layer norm), F0 and energy predictors (convs with AdaIN), and the
  ISTFTNet decoder (transposed convs with Snake activations, resblocks, producing magnitude and
  phase for an inverse STFT). That is LSTMs, AdaIN, Snake and 1-D transposed convs: ops the engines
  here do not have. The ALBERT part is the one block the BERT-class NPU encoder (`open_npue`) could
  plausibly take.
- It runs faster than real time on a CPU, so the engine would be host-first. The NPU adds little.
- Text frontend is the real obstacle: `misaki` G2P (Python) with an `espeak-ng` fallback.
  espeak-ng is GPL-3, which cannot simply be linked into this codebase; a C++ dictionary phonemizer
  for English (misaki's lexicons) is a project of its own, and the other seven languages each have
  their own G2P.
- Server work: `POST /v1/audio/speech` (OpenAI shape: `model`, `input`, `voice`, `response_format`,
  `speed`), WAV and PCM at least. ffmpeg DLLs are bundled for decoding; whether they carry an mp3
  or opus encoder was not checked.
- **Measured, dictionary-only English (2026-10-15).** `misaki` 0.9.4 with `fallback=None` against
  the same with `espeak-ng`, over 32 passages (435 words) in six categories. 22 words (5.1%) came
  back unknown, **all in "names" and "tech"**; chat, news, prose and numbers had none. Unknown
  words are emitted as a literal `❓` in the phoneme string. Examples: Nguyen, Saoirse, Siobhan,
  Xiaomi, Huawei, Qualcomm, Kilimanjaro, Kubernetes, PostgreSQL, nginx, Redis, bfloat16, softmax,
  tokenizer, config.json. 24 of 32 passages were identical either way. espeak's own output on
  the hard words is not always right either (it reads "Redis" as "reed-iz"). Small sample of my own
  text, `en_core_web_sm`, and **nobody listened to audio**, so what `❓` does to the speech is
  untested. Script: `g2p_test.py` (scratch).
- Smallest model in this document and the largest amount of new surface. Scope the first version
  to English, WAV out.

## Track D. Diffusion

Today's engine is geometry-locked to FLUX.2 klein 4B: stream layouts, constants and the exporter
(`klein_pipeline.py` module constants, `export_dit_kernels.py`, `q4nx/open_diffusion.py`'s
`TRANSFORMER`/`TEXT_ENCODER` checks). `engine.cpp` replays a schedule and is mostly generic.
Klein's weights are pre-packed into dit_gemm's bfp16 tile layout (block floating point, about 9
bits a weight, so already smaller than bf16) by `q4nx-build --open-diffusion` into an
`-NPU2` repo because packing 7.5 GiB per load "would take minutes". With no conversion, the engine
packs on first load from the source files (safetensors, or community GGUFs) and caches the result
locally keyed by source hash plus layout hash. That cache is derived data on the user's disk, not a
published model. Design decision for the maintainer. Suggested first move: make klein's constants a
per-model config.

### 5. Z-Image-Turbo

Nearest relative of klein.

- Text encoder: `Qwen3ForCausalLM`, hidden 2560, ff 9728, 32/8 heads, 36 layers: klein's TE exactly,
  so those kernels are reused. Only the tap differs (one hidden state vs klein's layers 9/18/27).
  A standard Qwen3-4B GGUF is a plausible source.
- VAE: Flux.1 AutoencoderKL, klein's topology at 16 latent channels, scale 0.3611, shift 0.1159
  (klein folds BN stats). Only `latent_in` in `vae_decoder.py` changes.
- DiT (new): `ZImageTransformer2DModel`, dim 3840, 30 heads at 128, 30 layers plus 2 noise-refiner
  and 2 context-refiner, ff 10240 SwiGLU, qk-norm, sandwich norms, adaLN from a 256-wide timestep
  embedding, RoPE axes [32, 48, 48], patch 2 on 16 channels. Klein is 3072 wide with 5 double and 20
  single blocks, so the block plan is new; dit_gemm/fa/ew are shared.
- 8 steps, no CFG, flow-match Euler with fixed shift 3.0 (klein's `empirical_mu` does not apply).
- Z-Image base is the same network at 50 steps with CFG (about 12x the DiT passes; needs an
  NPU-side CFG combine to stay inside OPEN-DIFFUSION-NPU-ONLY). Edit and Omni-Base are still "to be
  released" on the card. Apache 2.0. ~12 GB as shipped (bf16); roughly 7 GB at bfp16 and 6 GB at 8 bits resident, which is what the memory rule asks for.

### 7. Iris-3B (Turbo)

The Space `iris-3b-turbo` is a Gradio demo over the `turbo/` folder of the main repo; one model.
The depth and upscaler variants take image input; out of scope.

Pixel-space rectified-flow DiT, no VAE: 24 blocks at 2560 (8 dual-stream, 16 single-stream), GQA 20
over 5 heads, gated attention, sandwich RMSNorm, shared-bias adaLN, patch 16, then a 4-block pixel
head decoding each patch with per-pixel modulation (hidden 16 per pixel, attention across patches).
Text: Qwen3-VL-4B-Instruct, hidden states from 12 layers (2, 5, ..., 35) fused by a learned 12-layer
adapter. Turbo: 4 steps, CFG baked in, bf16, ~6 GB, safetensors only, Apache 2.0.

- The text tower has Qwen3-4B's dimensions, so klein's TE kernels fit; taps and adapter are new.
- The DiT body is the same kind of work as Z-Image.
- The pixel head is the risk: 4096 patches of 256 pixels at 1024 square, 16-wide hidden, every step.
  Tiny K for a tile GEMM; whether the NPU helps is unmeasured.
- Only reference is the authors' code (arXiv 2610.09450, weeks old).

### 8. Qwen-Image-2.1 and the uncensored GGUF repo

`abenzerps/...-Uncensored-Quantized` is the same architecture (GGUF DiT in BF16 and Q8_0 to Q4_0;
separate text encoder and VAE); one engine port serves both, and it is the one DiT here that already
ships as GGUF.

- DiT: `QwenImage21Transformer2DModel`, dim 4096, 32 layers at 32 heads x 128, ff ratio 3, 64 input
  channels, patch 1, RoPE axes [16, 56, 56], `causal_condition=true`. "Mixed-granularity attention"
  and "prefix KV cache reuse" are named on the card and documented nowhere; the mask has to be read
  from diffusers' `QwenImage21Pipeline`.
- Text encoder: Qwen3-VL-8B (hidden 4096, 36 layers), 17.5 GB bf16: a new TE kernel geometry.
- VAE: `AutoencoderKLQwenImage21`, causal-3D (Wan lineage), 64 latent channels, 16x spatial, decoder
  widths up to 1152, 4-channel RGBA out; far wider than the 512-channel VAE the conv kernels target.
- Default output 2048 square (16384 tokens x 32 layers); editing takes up to 10 references.
- DiT-GGUF reader (ComfyUI-GGUF, leejet fork): tensor-name map plus a pack of the shipped quant.
  Under the memory rule the quant should stay resident, not widen to bf16 at load: Q8_0 maps onto
  the 8-bit dit_gemm that #193 built for dense prefill, Q4_0/Q4_1 onto a 4-bit variant of it. K-quants
  (the repo's recommended Q4_K_M) would be requantized to one of those at pack time, as the LLM path
  does. Until a 4/8-bit DiT GEMM exists, the floor is bfp16.
- The 17.5 GB bf16 text encoder is too big by the same rule: use a Qwen3-VL-8B GGUF through the q4_1
  LLM pool path.
- License: Qwen Research License, not Apache. The uncensored fine-tune is third-party with no safety
  checker. Whether either enters the registry is Atomic-Germ's call; ask before building.

### 9. SDXL-Lightning

Distilled SDXL base 1.0 UNet at 1/2/4/8 steps (UNet-only, all-in-one, LoRA), safetensors, no GGUF
from ByteDance. Needs a UNet engine (ResBlocks, self and cross attention over 77 x 2048 context,
GEGLU, down/up sampling, skip concats), two CLIP encoders and tokenizer, pooled and size
micro-conditioning, Euler trailing (the 1-step model predicts x0). Reusable: `dit_conv` and the
GroupNorm ops from the VAE decoder, plus the SDXL VAE. Conventional ops and a diffusers reference
make it low-risk, but it is the highest-volume item.

## 6. Mellum2.1-12B-A2.5B-Thinking (new LLM family)

GGUF header read from the Q8_0 file by range request (arch `mellum`, 339 tensors): 28 layers,
hidden 2304, 32 heads over 4 kv at 128, per-head q/k norm, no biases or gate; every layer MoE with
64 experts, top 8, `norm_topk_prob`, expert ff 896, F32 router, no shared expert; sliding window 1024
on 3 of 4 layers, full attention with YaRN (factor 16, attn factor 1.277) on layers 4, 8, ..., 28;
vocab 98304, untied head. Quants: BF16, Q8_0, Q6_K, Q4_K_M, MXFP4_MOE (refuse). Expert tensors are
3D `[n_expert, ...]`.

Needs track B, and in addition:

- A `mellum` recipe and `GGUF_FAMILIES` entry; `qwen36moe` refuses a model without a shared expert
  (qwen36moe.py:382).
- The MoE kernels are validated at one point (ff 512, 256 experts, top 8, shared expert, hidden 2048:
  catalogue.py `router`, `moe`, `gemv_q4_prep_f32`). Mellum is ff 896, 64 experts, no shared
  expert, hidden 2304. The expert down projection contracts over 896, 3.5 of the 256-wide k-tiles.
  The attention tuple (128, 32, 4, 128, qk_norm) is new too.
- No recipe composes MoE with sliding-window layers; that is one of the four gpt-oss blockers
  (`families.py` NOT_IMPLEMENTED). Mellum has none of the other three.
- Q8_0 experts requantize to q4_1 through GGUF-direct: a fidelity drop against a native q8 path.
- Reference: llama.cpp does not list `mellum` and the card says GGUF support is "coming soon", so
  the oracle is HF transformers 5.8 `MellumForCausalLM`.
- Highest hardware risk of the LLMs (hand-placed 16 KB-program-memory cores).

## Track R. ALoDLM-8B (looped diffusion LM)

`amazon/ALoDLM-8B` (and `-1.7B`), Amazon, paper arXiv 2610.04198, code `amazon-science/ALoDLM`, repo created
2026-10-02. Config and docs read 2026-10-16. **License CC BY-NC 4.0** on the weights (noncommercial); the
upstream Qwen3 parts are Apache 2.0, and Amazon's optimized engine carries WeDLM-derived code with its own terms,
including a territorial restriction. We would write our own engine, not copy theirs. Whether a noncommercial
model belongs in the registry at all is Atomic-Germ's call; ask before building (the same flag as Qwen-Image).

**Shape: it is Qwen3-8B.** `Qwen3ForCausalLM`, 36 layers, hidden 4096, ff 12288, 32 heads over 8 kv at head dim
128, qk-norm, theta 1e6, vocab 151936, untied head, all full attention, 16.4 GB BF16 in four shards. That is
`recipes/specs/qwen3-8b.json` field for field, so the validated 8B kernel set applies. The checkpoint has **no
extra tensors** (index read: only `model.layers.0..35`, embeddings, norm, head).

**What is different is the schedule, not the weights:**

- The 36 layers split into a prelude `[0,10)`, a shared recurrent core `[10,26)` (16 layers) and a coda
  `[26,36)`. Each step takes a window of 16 tokens (optimized-engine default) and runs the prelude once, the
  core up to four times (K = 4, forced exit at depth 4), then the coda.
- Tokens that commit (entropy plus a small position penalty below `tau`, 0.4 in the examples; the leftmost
  token is forced if none qualify) feed back as embeddings; unresolved tokens keep a latent state that goes
  through the core again. Between passes the carry is normalized with the model's final RMSNorm.
- A **halting gate** stops the loop when the mean exit CDF over unresolved tokens reaches `q` (0.5 at 8B). The
  gate is a shared linear layer on detached readout features plus a bias per depth. It lives in a separate
  **`exit_gate.pt`**, a PyTorch pickle, not in the safetensors; the README names no shapes.
- **Depth-specific KV:** each core layer keeps a separate cache per recurrent depth; prelude and coda share one.
  That is 20 layers x 4 KB plus 16 layers x 4 depths x 4 KB = about 336 KB a token against 144 KB for plain
  Qwen3-8B, 2.3x. At 8K context, 2.7 GB instead of 1.2 GB. When a window stops early, uncomputed depths
  inherit the last completed one ("cache repair").

**Why it might fit the NPU, and what decides it.** The pass over 16 rows is nearly free against one row on this
hardware: the `utilities/pass-width-probe` measurement (diffusion-lm-plan, 2026-10-02) has the q4_1 GEMM at 16
rows costing 1.41x the one-row pass. Decode is memory-bound, so tokens per weight read is the lever, which is
what this schedule buys. But the arithmetic is not obviously a win, and the paper does not publish the two
numbers that decide it (no average pass count, no tokens committed per step). A step reads
20 + 16p layers, p being the passes actually run. Using our 121 ms/token dense decode and the 1.41x factor:
p = 2 and 3 tokens committed per step is about 12 tok/s against 8.3 for autoregressive, 1.5x; p = 3 and 2 tokens
per step is about 6 tok/s, slower than autoregressive. That is my estimate from our measurements, not a
measurement of the model; the first job is to find out where it lands. Two more costs it ignores: the lm head
(about 0.6B parameters, an eighth of the weights) is needed each pass to get entropies, and the 8-bit GEMM
route (#193) is what makes a 16-row pass cheap, so this rides on that.

**What exists already.** `feat/dllm` (unpushed worktree `C:\code\openflowlm-dllm`, tracking issue #154) has
`src/open_dllm` policies for WeDLM, SDAR and LLaDA2.2 behind `policy.hpp`, replayed against fixtures, with the
Phase 0 pass-width probe. ALoDLM is a fourth policy plus something none of those need: **running a layer range
repeatedly with per-depth KV pointers and a per-row latent carry**. WeDLM runs on today's per-token `step()`;
this cannot (16 sequential steps would erase the gain), so it needs the batched-row route the prefill work built.

Work, in order:

1. **No-hardware first:** read `alodlm/` and the paper, record the commit statistics (average passes, tokens per
   step) by running Amazon's portable decoder on CPU or any CUDA box on a few prompts. That settles the speed
   question before any NPU work. Needs the 8B, which is 16 GB at BF16.
2. Get the gate out of `exit_gate.pt` into a small safetensors file once, with a script in `utilities/`. We do
   not read pickles in the engine. Whether that counts as a forbidden conversion is a call for Cyrus; the gate
   is a few thousand parameters, not model weights.
3. The weights themselves: the community GGUFs (`mradermacher/ALoDLM-8B-GGUF`, i1 imatrix, `liodon-ai`) are plain
   `qwen3` conversions and load through GGUF-direct today as ordinary Qwen3-8B. They carry no gate, which is
   fine. The quality of a q4_1 requant of a recurrent model is untested; recurrence may amplify weight error
   across passes, so check NLL and the exit depths, not just single-pass logits.
4. Engine: layer-range execution, per-depth KV, the latent carry and final-norm between passes, readout and
   entropy per pass, the gate, cache repair. Validate against Amazon's portable decoder on a short prompt (same
   exit depths, same tokens at `tau` 0.4).
5. Server: a streaming loop that commits several tokens per step; the OpenAI stream has to emit them together.

Interest, honestly: 1,120 downloads on the base in its first two weeks, 51 likes, plus five community GGUF repos
in six days (the top one 2,667). That is a strong start for a research release and a low absolute number next
to the 9.6M for Qwen3-8B. Scored provisionally at 2 in Part 3; it moves up if the numbers grow or if the speed
estimate above holds. Its attraction is strategic rather than volume: of the diffusion LMs this is the one whose
backbone is already a validated kernel set (WeDLM and SDAR are also Qwen3-8B, LLaDA2.2 is not).

## Track V. MiniCPM-V vision tower

The `mmproj-model-f16.gguf` and `mmproj-model-f16-4x.gguf` files (BF16/F32, 460 tensors, 548M) are
`clip` arch, projector `minicpmv4_7`: SigLIP2-derived, 27 layers, width 1152, 16 heads, ff 4304,
patch 14, 448 px, 4900 learned positions, mean/std 0.5; an in-ViT early merger (`v.vit_merger`, a
window-attention layer at index 6 and a 4608 -> 17216 -> 1152 FFN), then `mm.up`/`mm.down`
(4608 -> 4608 -> LLM hidden) with scale factor 4, and LLaVA-UHD slicing up to 9 slices. The two
files are the 16x and 4x token-compression variants. The engine's host ViT is Qwen3-VL's (patch 16,
conv3d patch embed, deepstack, M-RoPE), so this is a new host path, separate from the text work. It
can follow the text path; the registry must not label either model "vision" until then.

## Track S. Speculative decode

### MiniCPM-V-4.7-35B-A3B-DSpark: not supported

`grep -i dspark` finds nothing in the working tree, `upstream/main`, any branch or any PR title. The
nearest thing that exists is DFlash2 for Ternary Bonsai 2 27B: local branch
`feat/bonsai2-dflash2-integ`, unpushed, stacked on PR #201, which is open, targets
`feat/qwen35-27b` and not `main`. The merged drafter work is Uno for Granite (#192).

DSpark is a drafter, not a model: `openbmb/MiniCPM-V-4.7-35B-A3B-DSpark`, Apache 2.0, safetensors
only (no GGUF), paired to the 35B only, 389M parameters.

- It is DFlash-lineage (the config literally has `dflash_config.target_layer_ids`): 5 layers,
  hidden 2048, 8 heads over 2 kv at head dim 256, ff 6144, draft vocab 248144, mask token, taps at
  target layers 3, 11, 19, 27, 35 of 40, block size 15. New against DFlash2: a confidence head and a
  rank-256 Markov head that make the block semi-autoregressive (arXiv 2607.05147; the card and
  abstract describe confidence-scheduled verification, the paper body was not read).
- Reuse: the tap mechanism and the block-drafter run on prefill-GEMM and batched-attention kernels
  that the DFlash2 work built.
- New: a 35B MoE target with speculative verify (the 35B plan is #169, open, with DeltaNet rollback
  and no MoE verify); verify width is capped at 4 by data memory, so a 15-token block has to be
  truncated by the confidence head; each verified position reads its own experts; drafter weights
  from safetensors, since no GGUF exists. Only vLLM is documented for running it.
- Depends on track B and on #201/#169 reaching `main`. Not a near-term item.

## Recommendation on "one PR"

The models do not share a code path: A and B are the LLM engine's GGUF-direct, D is the diffusion
engine, E is the embedding engine, and none of it exists today. One PR would be unreviewable and a
hardware regression in one would block the rest. Suggested stack:

1. Tell Atomic-Germ the pool layout is bf16-scale 5120 (not f32) for new families, and raise the
   `real_vocab` hash question. Ask Cyrus before posting.
2. PR 0: embedding mode and `/v1/rerank` on the dense qwen3 path, proven with Qwen3-Embedding-0.6B
   and Qwen3-Reranker-0.6B. No new kernels and no dependency on the pool-layout decision.
3. PR A: qwen35 GGUF-direct + MiniCPM-V-4.7-1B (text) + the blend + the Qwen3.5 0.8B embedder and
   reranker. Gains every Qwen3.5/3.6/3.8 GGUF.
4. PR B: MoE GGUF-direct + MiniCPM-V-4.7-35B-A3B (text); then Qwen3-Coder-30B-A3B (MoE without a
   shared expert); then Mellum on top of that. Muse Glimmer-30B (track G, dense, ~2 tok/s) slots
   in after the two A3B models and before Mellum; run the kernel-fit probe early since it is cheap.
5. PR C: EmbeddingGemma-2 text tower (independent; can go any time).
   PR E (independent): an engine-neutral ASR interface, then Moonshine, then Parakeet. Kokoro is
   its own PR with a new endpoint and a phonemizer decision.
6. PR D: diffusion loader + cache with per-model constants + Z-Image-Turbo.
7. Later: MiniCPM-V vision, Iris, Qwen-Image-2.1, SDXL-Lightning, DSpark.

## What changed

- Rev 7 adds Muse Glimmer-30B (track G): text path feasible from existing dense-recipe parts, one
  new kernel point at hidden 6656 / ff 19968 is the risk; vision and drafter deferred.
- Rev 1 called the blend "out of the box": true of the kernels, false of the weights path, and read
  from a local `main` that predates #180.
- Rev 2 added the GGUF-direct constraint and the diffusion loader.
- Rev 6 adds speech: Moonshine, Parakeet, Kokoro (track T).
- Rev 5 adds Qwen3-Coder (30B-A3B realistic, Next deferred) and points the Qwen3 embedder/reranker
  rows at their own plan, `qwen3-embedding-reranker.md`.
- Rev 4 adds the LLM-backed embedders and rerankers (track L) and the finding that nothing in the
  tree can pool a hidden state or rerank.
- Rev 3 adds MiniCPM-V 1B and 35B-A3B, EmbeddingGemma-2, the DSpark answer, the vision track and the
  `real_vocab` hash finding.

## Unchecked

- Muse Glimmer: GGUF tensor names and metadata keys (config was read from the HF repo, not a GGUF);
  whether the query scale is a per-layer constant; whether the 6656/19968 kernels fit program memory;
  tokenizer format and chat template for the 202k vocab.
- Whether `qwen35.py` truly has no nested gguf route (grep only).
- Whether llama.cpp's `qwen3_5` converter handles the blend's merged vision tower.
- Whether the MiniCPM-V text tower is bit-identical to Qwen3.6-35B-A3B / Qwen3.5-0.8B in layout
  (shape matches; weights differ, as fine-tunes do) and whether the 35B GGUF carries MTP tensors.
- Peak NPU memory for Z-Image-Turbo's DiT plus TE; klein's packed bundle is 7.5 GiB.
- Whether Mellum's YaRN attention factor fits the existing position-record rope table.
- The DSpark paper body (architecture of the Markov head, exact verify rules).

---

# Part 2. FastFlowLM catalogue gaps, ranked separately

2026-10-15. A separate list from Part 1, which is Hugging Face targets outside the FLM catalogue.
Part 2 covers what FLM users already have (https://fastflowlm.com/docs/models/, all 15 families, each
family page read for its tags) and the open engine cannot yet run. Status from `specs/open-engine/spec.md`
(upstream/main at a77fc458), `families.py`, `catalogue.py` and the local `docs/docs/models/index.md`
matrix. Interest = Hugging Face 30-day downloads of the base model, pulled 2026-10-15. Item numbers here
are independent of Part 1's row numbers. The earlier Atomic-Germ priority list
(`.claude/plans/archive/open-kernel-model-priority.md`) is done and untouched.

## The answer

Of 15 families and about 50 tags, the open engine already runs most of the catalogue. What is left is
nine items, and one of them (Gemma 4) is three tags plus two "Flash" aliases:

| # | Item | Tags | Interest (30d dl) | Difficulty | Why |
|---|---|---|---|---|---|
| 1 | Hy-MT2 1.8B | `hy-mt2-flash:1.8b` | 28k | S | hunyuan recipe exists; one new size |
| 2 | Whisper from stock weights | `whisper-v3:turbo` | 6.3M | S-M | open engine is built and wired, but only reads a converted container |
| 3 | Gemma 3 1B | `gemma3:1b` | 2.0M | M | no kernel point at hidden 1152 / ff 6912 / 1 kv head |
| 4 | Gemma 3 images | `gemma3:4b`, `medgemma:4b`, `medgemma1.5:4b`, `translategemma:4b` | 1.1M + 0.67M + 0.08M + 0.03M | M | host SigLIP tower; images still go to the closed engine |
| 5 | Gemma 4 12B | `gemma4-it:12b` | 1.7M | M-L | Gemma-3-shaped dense model plus 512-wide global attention |
| 6 | Gemma 4 E4B / E2B | `gemma4-it:e4b`, `:e2b` (+ `gemma4e-flash:*`) | 4.2M + 3.0M | L | per-layer inputs, KV sharing across layers |
| 7 | gpt-oss | `gpt-oss:20b`, `gpt-oss-safeguard:20b` | 6.0M + 0.03M | XL | MoE with biases, clamped SwiGLU, sliding window; on paper only |
| 8 | Gemma 4 images and audio | same Gemma 4 tags | (rides 5, 6) | XL | two new towers, one a conformer |
| 9 | SmolVLA | `smolvla` | 0.14M | XL | a robot policy, not a chat model; no serve path |

Do them in that order. The reasoning, and what to build first inside each, is below.

## What already runs (so it is not on the list)

Open engine, text, per the spec's recorded results: Llama 3.2 1B/3B and 3.1 8B; both DeepSeek-R1
distills; Qwen3 0.6B to 8B plus the 2507 Instruct and Thinking 4B; Qwen3-VL-4B and Qwen2.5-VL-3B
(images too: `oflm-test --vision` passes, spec 2026-09-13); Qwen2.5-3B; Qwen3.5 0.8B to 9B; Qwen3.6-35B-A3B;
Qwen3.8-27B; Gemma 3 4B and its MedGemma and TranslateGemma fine-tunes (text); Phi-4-mini; Nanbeige4.1-3B;
the LFM2 family (the matrix lists all five tags as open; only the 1.2B has a recorded hardware run, see
"Verify, don't build"). EmbeddingGemma-300M runs on the CPU engine.

The local matrix in `docs/docs/models/index.md` is out of date in two places. It says the open engine has
no vision path for Qwen3-VL and Qwen2.5-VL (the spec says it does), and it marks `gemma3:1b` open when
there is no catalogue point for it (below). Fix the matrix when the next of these lands.

Not ours, by design:

- **`qwen3-rai:*` and `phi4-mini-it-rai:4b`.** These run on AMD's `ryzenai_corelib` backend from a
  Q8_0 GGUF requantized to group 64, 4095 tokens of context, and are not in the installers. Different
  engine, different kernels, closed library. Out of lane.
- **The `-flash` tags** (`gemma4e-flash`, `qwen3vl-flash`, `hy-mt2-flash`). Per the FLM v1.0.6 notes they
  are the same weights through faster closed kernels, single-turn, fixed 1k context. We would serve the
  base tag; at most the registry needs an alias. Note that `hy-mt2-flash:1.8b` is the only FLM tag for
  that model, so item 1 is how it shows up.

## Ranked list

### 1. Hy-MT2 1.8B (S)

`tencent/Hy-MT2-1.8B`, Apache 2.0, a translation model. Config read from the hub: `hunyuan_v1_dense`, 32
layers, hidden 2048, ff 6144, 16 heads over 4 kv at head dim 128, qk-norm, vocab 120818, tied head,
dynamic-NTK rope (alpha 1000).

The hunyuan recipe already runs the 7B, including the two things that make it unusual (qk-norm after
RoPE, folded rope base). What is new is the size: hidden 2048 / ff 6144 is Qwen3-1.7B's GEMV and norm
widths, which the catalogue validates, and the attention tuple (128, 16, 4, 128, qk-norm-post-rope) is a
new point. One export and a validation run. Interest is low (28k), but it is the cheapest item and it
closes a tag. Hy-MT2's translation quality under q4_1 has never been scored (open item in
`open-kernels-recipes-manifest`); do that here with FLORES pairs.

### 2. Whisper from stock weights (S-M)

`whisper-large-v3-turbo`, MIT, 6.3M downloads: the most-used model in the catalogue that we do not serve
on the open path. The open engine exists (`src/open_whisper`, issue #72, `oflm serve --asr 1` wired in
phase 3b) and has an offline reference. But it loads `model.open.safetensors`, which `q4nx-build` writes
into an `-NPU2` repo, and the published FLM container ships only `model.q4nx`, so the open path is never
selected for a user who runs `oflm pull whisper-v3:turbo`.

That is exactly the situation the no-conversion rule (CLAUDE.md, "Kernels belong to families") is meant
to remove. Decision (Cyrus, 2026-10-15): **keep the safetensors loader and add a GGUF loader beside it.**
Nothing is removed. The two paths:

- **Safetensors:** the existing `weights.cpp` path, 16-bit as today. Whisper is a small non-LLM engine, so
  the weights rule allows it. Open question: the loader reads `model.open.safetensors` with a
  `weights_manifest.json`; whether it can read OpenAI's stock `model.safetensors` directly, or needs a
  name-and-layout map for that file, is unchecked. Reading the stock file is what removes the need for an
  `-NPU2` repo.
- **GGUF:** a new reader behind the same weights seam. whisper.cpp's own format is ggml, not GGUF, and
  community GGUF conversions exist; I have not read one's tensor names or layout, so the work is unsized
  until one is opened. A quantized GGUF (Q8_0, Q5_0) would be dequantized into the bf16 tiles the encoder
  streams expect, unless the GEMM stream gets 8-bit operands, which would also bring resident weights below
  the 1.2 GB bf16 figure.

The decoder is host-side fp32 and the encoder GEMM streams are already built, so the work is two readers
and the layout maps, not an engine.

### 3. Gemma 3 1B (M)

`google/gemma-3-1b-it` (Gemma license), 2.0M downloads, the second most-downloaded small model in the
list. The `Gemma3_Text_Only` adapter exists, but there is no kernel set: `catalogue.py` has no point at
hidden 1152 or ff 6912, and the 2026-09-10 batch plan (`archive/remaining-models-batch.md`) dropped it as
"not a packer job". Geometry: 26 layers, hidden 1152 (4.5 K-tiles of 256), ff 6912 (13.5 bands of 512),
4 heads over 1 kv at head dim 256, vocab 262144. Four things to build:

- a padding law in the packer for the two non-tile-multiples (1152 and 6912);
- a one-kv-head attention point; the design assumes an even kv count;
- LayerNorm and lm_head kernels at width 1152;
- the 1280-byte-chunk regroup, unless the model comes in through GGUF-direct, which does not care.

Take it by GGUF-direct. Gemma 3 is a dense GGUF family, so the chunk problem disappears, and the
one-kv-head attention point is also what every Gemma 4 global layer needs (items 5 and 6), so it is not
single-use work. Its real value is as the cheapest vehicle for that point.

### 4. Gemma 3 images (M)

MedGemma 4B (0.67M) is the second-biggest user here, then Gemma 3 4B (1.1M); TranslateGemma is a text
model in practice. Text runs open; any image drops to the closed `gemma_npu` DLL ("Images always need the
closed engine", spec). The tower is SigLIP, 27 layers, width 1152, patch 14, 896 px, followed by a pooling
projector to 256 tokens. The host ViT pattern in `src/open_qwen36/vision/` (reference, host port,
`vit_test`) applies; the tower is not Qwen's, so it is a new port, but the 27-layer, 1152-wide SigLIP
shape is shared with MiniCPM-V's (Part 1, track V), so one port serves both.

The risk is the mask. Gemma 3 attends bidirectionally inside an image block while the text stays causal.
The dense decode path is token-at-a-time with a position record; the batched-prefill route is where a
mask change would go. The batch plan flagged this as the likely blocker; nobody has checked. Read the
prefill route before estimating further.

### 5. Gemma 4 12B (M-L)

`gemma4-it:12b`, 1.7M downloads, Apache 2.0, QAT Q4_0 (the q4_0 container already packs, OPEN-PACK-Q4-0).
Config read from the hub (`gemma4_unified`): 48 layers, hidden 3840 (15 tiles), ff 15360 (60 tiles),
16 heads over 8 kv at head dim 256, five sliding layers (window 1024) then one global, **global layers
have 1 kv head at head dim 512 and `attention_k_eq_v`**, rope theta 1e6 with partial rotary 0.25 on the
global layers and 1e4 on the sliding ones, logit softcap 30, vocab 262144, tied head. No MoE, no
per-layer inputs, no KV sharing.

So it is Gemma 3 12B (which we run, `gemma3-12b.json`) with four changes: the 512-wide global attention
point, K = V (one fewer projection, and the cache holds one tensor), proportional partial rotary, and a
softcap on the host after the head. Of those only the first can fail on hardware: head dim 512 doubles the
per-head tile and it may not fit L1 or program memory. **Probe that first** (build the (512, 16, 1)
attention point and see whether it links; hours, no loader). The server already parses Gemma 4 tool calls
(PR #59). Decode is memory-bound at ~7 GB of weights per token, so expect the same class of speed as the
27B scaled by size, roughly 4 to 5 tok/s.

### 6. Gemma 4 E4B and E2B (L)

The highest raw demand among the open gaps (4.2M and 3.0M) and the most work of any text item that
fits the engine. Both read from the hub:

- E2B: 35 layers, hidden 1536, ff 6144, 8 heads over 1 kv, head dims 256 / 512, window 512.
- E4B: 42 layers, hidden 2560, ff 10240, 8 heads over 2 kv, same head dims, window 512.
- Both: per-layer input width 256, KV shared across the last 20 (E2B) or 18 (E4B) layers, global every
  fifth or sixth layer, softcap 30.

What is new against the 12B, in order of risk:

1. **KV sharing.** The last layers reuse an earlier layer's K and V rather than computing their own, so
   they have no K/V projection and the cache is smaller than the layer count suggests. The engine's KV
   pool is per layer; layers need to point at another layer's buffer. This is structural, not a kernel.
2. **Per-layer inputs.** Each layer adds a small vector looked up from a 262144 x 256 table per layer
   (about 2.4B parameters of "embedding" for E2B) and gated into the residual. The tables are lookups, not
   matmuls, so they live on the host like the main embedding; the container's odd chunk sizes (1536 to
   10752 bytes, from the earlier census) are these tensors. The per-layer projection and gate are small
   GEMVs at width 256.
3. **Double-wide MLP** in the KV-shared layers (ff 12288 for E2B): extra GEMV points.
4. The 512-wide global attention and softcap, as in item 5.

Do E2B first as the bring-up vehicle (about 3.5 GB at 4 bits for the dense weights plus a host-resident
per-layer table), then E4B, which is the same structure at the larger size. Do not start either before
item 5's attention probe has passed. The Flash tags need nothing beyond a registry alias.

Text only. Be explicit in the registry: `gemma4-it:*` advertises image and audio input and would be
mislabelled until item 8.

### 7. gpt-oss (XL)

20B MoE, Apache 2.0, 6.0M downloads (the safeguard fine-tune, 27k, is the same architecture and free once
the base runs). By demand it belongs near the top; by difficulty it is last of the LLMs, and the tree
already says why. `NOT_IMPLEMENTED["gptoss"]` in `families.py` lists four blockers: the experts want a
clamped SwiGLU (`moe_silu.cc` does not clamp); `o_proj`, the router and all three expert projections carry
a bias that no design has room for; the MoE FFN has to compose with sliding-window layers, which no recipe
does; and expert ff equals hidden at 2880, which the MoE core layout does not survive. Attention sinks
landed and are no longer a gap. The arithmetic is tested (`replica_gptoss.py`), the MXFP4 expert layout is
decoded and fixture-tested, and the packer knows where each expert's tensors are; no kernel computes it.

Two facts shape the plan. It is 15.1 GB resident per FLM, so the memory budget is fine (it fits under the
35B's 21.6 GB). And FLM's own page calls for more than 32 GB of system RAM, so the user base is
narrower than the download count suggests. Start by deciding the expert layout for ff 2880 (11.25 tiles of
256); the other three blockers are mechanical once that is settled, and this is shared work with
Mellum (Part 1) which also needs MoE plus sliding window.

### 8. Gemma 4 images and audio (XL)

E2B/E4B carry a 16-layer vision tower (width 768, patch 16, pool 3, 280 soft tokens) and a 12-layer
conformer audio tower (width 1024, chunked attention with logit cap 50, depthwise conv kernel 5). 12B has
its own vision front end and a 640-wide audio tower. The vision towers follow item 4's pattern. The
audio conformer shares most of its design with Parakeet's FastConformer (Part 1, track T), so if
Parakeet is built the audio half of Gemma 4 is cheaper than it looks. Last, after the text models ship.

### 9. SmolVLA (XL, low priority)

`lerobot/smolvla_base`, Apache 2.0, 144k downloads, about 0.5B parameters, bf16, up to three camera
images and 48 tokens of context. It is a SmolVLM2-500M backbone cut to its first half, plus an action
expert trained by flow matching that emits a chunk of motor actions (the architecture is from the paper,
arXiv 2506.01844, and a search summary; the denoising-step count of 10 is from an Arm page, not read from
the config). The ops are ordinary transformer ops, so the engine pieces mostly exist.

The reasons it is last are not kernel reasons. Its output is an action vector, not a token stream, so
there is no OpenAI-shaped endpoint for it; FLM publishes it as a LeRobot policy at
`FastFlowLM/smolvla-NPU2`, not through `flm serve`, and our own docs page already says it "needs a
separate inference path". Someone would have to design that interface. The audience (people running
LeRobot arms on a Ryzen AI laptop) is small. Build it only if someone asks.

## Verify, don't build

Three things the matrix claims that I have not seen run, each an hour of checking before it counts as
done:

- **LFM2-2.6B, LFM2-2.6B-Transcript, LFM2.5-1.2B Instruct and Thinking.** Only LFM2-1.2B has a
  validated spec and run. LFM2-2.6B (config read) is hidden 2048, 30 layers with 8 attention layers at
  2, 5, 9, 13, 17, 21, 24, 27, ff 10752 as written (`block_auto_adjust_ff_dim` false, so not the 1.2B's
  two-thirds rule), vocab 65536. Every width is a point the catalogue already has, so I expect a new
  layer plan and one export, not new kernels; LFM2.5-1.2B should be the 1.2B shape. Interest: LFM2.5
  Instruct 116k, the rest under 10k.
- **Qwen3.5 0.8B to 9B, Qwen3.6, Qwen3.8 images.** The local matrix says closed; issue #16's host ViT and
  M-RoPE path was coded on 2026-09-08. Check what the current build does with an image.
- **EmbeddingGemma-300M on the NPU.** It runs on the CPU by default and the NPU matmul is opt-in. That is
  a decision, not a gap.

## Cross-references

MiniCPM-V-4.7-1B (`minicpm-v-4.7:1b`, Q4_K, tool calling, toggleable thinking) is in the FLM catalogue
and is row 1 of Part 1 (track A, qwen35 GGUF-direct). It is the single
FLM tag that doc already covers. The Qwen3.5 / 3.6 / 3.8 Q4_K tags are the same GGUF-direct track.

## Unchecked

- Gemma 3 1B: whether any tool in the tree already handles hidden 1152; I found no catalogue point and
  the batch plan, but did not try an export.
- Whether the 12B's 512-wide global attention fits L1 and program memory; the probe is the answer.
- Gemma 4 `attention_k_eq_v` exact semantics, and whether E-series global layers use it (E2B and E4B
  configs say no; 12B says yes).
- Gemma 3 image mask: bidirectional inside the image, read from the model card only.
- Download counts are Hugging Face's rolling 30-day figure for the base repo only. They miss
  fine-tunes and re-uploads, and say nothing about how many downloaders have a Ryzen AI NPU.
- The speed guesses (12B at 4 to 5 tok/s) are scaled from the 27B's 2.2 tok/s, not measured.
- Which FLM tags currently have a registry entry and a kernel set installed; I checked the code and spec,
  not a running install.

---

# Part 3. Combined priority list

2026-10-15. Parts 1 and 2 merged into one ranking. Parts 1 and 2 stay as the detail; this is only the order.

**Scoring.** `score = interest + 2 x closed - difficulty`, each 0 to 5, ties broken by interest.

- **Interest** from Hugging Face 30-day downloads of the base repo: 5 is over 5M, 4 is 2M to 5M, 3 is 0.5M to 2M,
  2 is 0.1M to 0.5M, 1 is under 0.1M. Where a family shares one recipe, the family total is used (Qwen3-Coder
  counts with Qwen3-30B-A3B). Rebine's embedder, Moonshine, Mellum, Iris, Qwen-Image and the blend had no
  count I could pull; those interest scores are my guess.
- **Difficulty** from the Part 1 and Part 2 write-ups: 1 is an export run, 3 is a new recipe point or a host
  port, 4 is new kernels or a new engine surface, 5 is several unsolved pieces at once. Halves are used where
  a write-up said "M-L".
- **Closed** is the bonus for retiring a closed kernel. 2 means the model runs only on a closed FLM engine
  today (`gemma4e_npu`, `gemma4_12b_npu`, `gpt_oss_npu`, `whisper_npu`, the Gemma 3 text DLL, the Qwen3.5-VL
  DLL for MiniCPM-V) and this gives it an open path. 1 means the text already runs open and this removes the
  closed fallback for part of it (images). 0 means FLM has no engine for it, so there is nothing closed to
  retire; it is new capability, not replacement.

The formula treats a closed engine as worth four points, about one full interest step plus a difficulty
step. That is a judgment, not a measurement. If you weigh it more or less, only the 0 to 4 band reorders.

| Rank | Item | Part | Interest | Diff | Closed | Score |
|---|---|---|---|---|---|---|
| 1 | Whisper: keep safetensors, add GGUF | 2 | 5 | 2.5 | 2 | 6.5 |
| 2 | Gemma 3 1B | 2 | 4 | 3 | 2 | 5 |
| 3 | Hy-MT2 1.8B (official GGUF: 295k dl, so interest 2) | 2 | 2 | 1 | 2 | 5 |
| 4 | gpt-oss 20B (+ safeguard) | 2 | 5 | 5 | 2 | 4 |
| 5 | Gemma 4 E4B / E2B text | 2 | 4 | 4 | 2 | 4 |
| 6 | Gemma 4 12B text | 2 | 3 | 3.5 | 2 | 3.5 |
| 7 | Qwen3-Embedding / Reranker (0.6B, 4B, 8B) | 1 | 5 | 2 | 0 | 3 |
| 8 | Gemma 3 images (4B, MedGemma, TranslateGemma) | 2 | 3 | 3 | 1 | 2 |
| 9 | MiniCPM-V-4.7-1B text (opens track A) | 1 | 1 | 3 | 2 | 2 |
| 10 | Kokoro-82M TTS | 1 | 5 | 4 | 0 | 1 |
| 11 | Qwen3-Coder-30B-A3B (+ Qwen3-30B-A3B family) | 1 | 4 | 4 | 0 | 0 |
| 12 | Gemma 4 images and audio | 2 | 3 | 5 | 1 | 0 |
| 13 | SmolVLA | 2 | 1 | 5 | 2 | 0 |
| 14 | Parakeet | 1 | 3 | 4 | 0 | -1 |
| 15 | EmbeddingGemma-2 text | 1 | 2 | 3 | 0 | -1 |
| 16 | Moonshine | 1 | 2 | 3 | 0 | -1 |
| 17 | Qwen3-Coder-Next 80B | 1 | 3 | 5 | 0 | -2 |
| 18 | Z-Image-Turbo | 1 | 3 | 5 | 0 | -2 |
| 19 | Muse Glimmer-30B text | 1 | 2 | 4 | 0 | -2 |
| 20 | SDXL-Lightning | 1 | 2 | 4 | 0 | -2 |
| 21 | ALoDLM-8B (provisional interest; see below) | 1 | 2 | 4 | 0 | -2 |
| 22 | Qwen3.5 0.8B embedder and reranker | 1 | 1 | 3 | 0 | -2 |
| 23 | Qwen3.8-3.6-27B blend | 1 | 1 | 3 | 0 | -2 |
| 24 | MiniCPM-V-4.7-35B-A3B text (opens track B) | 1 | 1 | 3 | 0 | -2 |
| 25 | Qwen-Image-2.1 | 1 | 2 | 5 | 0 | -3 |
| 26 | Mellum2.1-12B-A2.5B | 1 | 2 | 5 | 0 | -3 |
| 27 | Iris-3B Turbo | 1 | 1 | 5 | 0 | -4 |

**ALoDLM-8B (rank 21).** The formula puts it here because FLM has no engine for it (closed 0) and 1.1k
downloads in two weeks scores interest 2. Difficulty 4: the backbone is a validated kernel set, but the
recurrent schedule, per-depth KV and gate are new engine work and need the batched-row route. Two things
could move it. Interest 4 (over 2M downloads) would give a score of 0 and rank about 11 to 13; it is two weeks
old, so that is not impossible but not the current number. Or the first no-hardware step shows the commit
statistics make it clearly faster than autoregressive, which would justify a strategic override. Do the
no-hardware step first (track R, step 1); it costs no NPU time and decides whether to rank it at all.

Dropped from the ranking: MiniCPM-V-4.7-35B-A3B-DSpark (not supported, depends on unpushed work) and the
MiniCPM-V vision tower as its own row (it shares the SigLIP-shaped host port with row 8; build it once).
The Part 2 "verify, don't build" items are checks, not rows.

## What the order says

The top is mostly closed-engine replacement, which is what the bonus was for: Whisper, Gemma 3 1B, Gemma 4,
gpt-oss and Hy-MT2 all run only on a closed FLM engine today. Of those, Whisper is first by a wide margin
because the open engine is already built and the missing piece is a weight reader. The first Part 1 row to
appear is Qwen3-Embedding at 7, because it is 8.6M downloads for a job layer with no new kernels. Everything
else in Part 1 is new capability with no closed engine to retire, so it sorts below the replacement work
unless its interest is very high (Kokoro, Qwen3-Coder).

## Build order

Score order, with three corrections for dependencies and risk. The ranking says what is worth doing; this
says what to start.

1. **Whisper from stock weights**, then **Gemma 3 1B** with **Hy-MT2 1.8B** beside it (both are export-sized
   or a bounded kernel point, and Gemma 3 1B builds the one-kv-head attention point that Gemma 4 needs).
2. **Probe before Gemma 4:** build the 512-wide, one-kv-head attention point and see whether it links.
   Hours of work. It gates rows 4 and 6 (and Part 2's item 8), so a failure here changes the plan before
   anyone commits weeks to it.
3. **Gemma 4 12B before E4B/E2B**, although the E-series ranks higher. The 12B carries the 512-wide
   attention and softcap on a Gemma-3-shaped model; the E-series adds KV sharing and per-layer inputs on
   top. Do E2B as the bring-up vehicle, then E4B.
4. **gpt-oss: start the design spike, expect it last of this group.** It scores the same as E4B but has four
   unresolved blockers, one of them a core layout decision (expert ff 2880). Run the spike for the expert
   layout in parallel with Gemma 4 and commit to the build only once that is settled.
5. **Qwen3-Embedding and Reranker can go any time.** No new kernels, and no dependency on the pool-layout
   decision, so it is a good filler PR between the Gemma steps (Part 1's PR 0).
6. **Track A (qwen35 GGUF-direct) rides on row 9.** Its rows each score low (the models have few
   downloads), but it gates the MiniCPM-V pair, the blend, the 0.8B embedder and reranker, and the MoE
   track behind it. It earns its place as infrastructure, not on its own score. Start it when the first of
   those is pulled forward, and tell Atomic-Germ about the pool-layout choice first (Part 1, "Decided by the
   memory rule").
7. **The safetensors quantizer (foundation, below) can start any time** and has no dependency on the
   others. Do it before any LLM whose only source is safetensors, and before track D's DiT packer, since
   the two share the cache. It does not block rows 1 to 6.

## Weights: GGUF and safetensors, any model

Per the rule at the top: every model can load from GGUF or from safetensors, with no q4nx. Small non-LLM
engines load safetensors in 16-bit; everything else quantizes safetensors to q4_1 at load. What that
changes, from the Part 1 and Part 2 write-ups (GGUF availability is as those write-ups found it; I did not
re-check each repo today):

- **Foundation item, unranked: the safetensors quantizer.** One shared C++ quantizer from bf16/fp16 to
  q4_1 (and q8 for the roles that have it), pinned byte-for-byte to `recipes/pack.py` the way
  `pools_test` pins `requant_q4_1`, plus a local cache keyed by source hash and layout hash. Difficulty 3.
  It has no closed-engine bonus and no model of its own, so it does not rank; it is a prerequisite that
  removes "wait for a GGUF" from every LLM row. It also makes the Qwen3.8-3.6 blend loadable from its own
  repo (no GGUF has to be made), and every future HF release loadable on day one. Costs to budget: minutes
  of host quantization per first load (the cache amortizes it), a transient copy of each tensor, and
  quality (plain min/max q4_1 is below llama.cpp's K-quants, so validate by NLL against the 16-bit
  weights, as the requantized GGUF paths were). Sources already quantized in safetensors need their own
  decode first: gpt-oss ships MXFP4, whose layout is decoded and fixture-tested in the tree; FP8 repos
  (Qwen3-Coder's only official one) would need a decoder.

- **Whisper (rank 1)** keeps its safetensors loader and gains a GGUF reader beside it (Part 2, item 2). That
  is two readers rather than one, so difficulty moves from 2 to 2.5 until a real Whisper GGUF has been
  opened and the stock-safetensors question is answered; the score is 6.5, still rank 1.
- **Gemma 3 1B, Gemma 4 12B, Gemma 4 E2B/E4B, gpt-oss, Hy-MT2 1.8B** need GGUF tensor names for their
  families. Gemma 3 and hunyuan (`hunyuan-dense`) are already in `GGUF_FAMILIES`; Gemma 4 and gpt-oss are
  not, so each adds a name map. GGUFs exist for all of them (checked 2026-10-15): `tencent/Hy-MT2-1.8B-GGUF`,
  `ggml-org/gemma-3-1b-it-GGUF`, Google's own `gemma-4-E4B-it-qat-q4_0-gguf` and `gemma-4-12B-it-qat-q4_0-gguf`,
  `ggml-org/gpt-oss-20b-GGUF`, and for Whisper `handy-computer/whisper-large-v3-turbo-gguf` (419k downloads). Gemma 3 1B gets easier: its 1280-byte chunk problem was a q4nx problem and goes away.
- **The q4nx chunk-geometry work** (1280, 2560 and 5120-byte chunks in the container-formats plan, Gemma 4's
  odd per-layer-embedding chunks) is no longer needed. The q4_0 transcode stays; it is a GGUF block layout.
- **No GGUF exists, small non-LLM engines:** EmbeddingGemma-2, Kokoro-82M, Moonshine, the DSpark drafter.
  These load 16-bit safetensors, so they do not wait for a GGUF. Their scores stand, and the cost of
  "unsized" is only that nobody has checked how their engines would read the file. Parakeet also has an
  official Q8_0 GGUF for v3.
- **No GGUF exists, large models:** the diffusion DiTs other than Qwen-Image's community GGUF (Z-Image-Turbo
  and others) take safetensors through the same load-time pack and cache, which is Part 1's track D
  design; the quantizer above is the LLM half of that machinery and the DiT packer the other half.
- **Any LLM without a GGUF** loads from safetensors once the quantizer lands; until then it waits.
- **The closed bonus is unchanged.** It is about which engine runs the model, not the file format.

## Unchecked

- Every score is mine; none was agreed with the maintainer. Interest ignores who has an NPU, and download
  counts include automated pulls.
- "Closed" is read from the local docs matrix and `NOT_IMPLEMENTED`, not from a running install. The matrix
  is stale in places (Part 2, "What already runs"), so a row's closed bonus could be wrong in either direction.
- Qwen3-Coder interest sums the Coder-30B (0.42M), Qwen3-30B-A3B (1.26M) and its 2507 Instruct (1.05M)
  downloads; the recipe covers all three but I have not confirmed the 2507 variants share every dimension.
