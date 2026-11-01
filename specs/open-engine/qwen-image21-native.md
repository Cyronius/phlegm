# Qwen Image 2.1 native NPU implementation

Checkpoint: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1).
The [Diffusers pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py),
[scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py),
and [VAE](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py)
are arithmetic references. Local checkpoint and generated kernels are in
`/home/cyrus/Documents/ChatGPT/current/work/phlegm-image-models/`; they are
not Git assets. The target is native NPU text-to-image and editing through
Phlegm's CLI and image API.

## Current state

The Qwen3-VL text tower, image entry, 32-layer image transformer, final
velocity projection, and complete first-frame VAE decoder run on the NPU.
A connected four-step FlowMatch Euler diagnostic decoded a coherent 512-pixel
red-cube image. The CLI and image API now route `backend=qwen21` through this
native path. The published default 40-step schedule and variable-prompt CLI/API
integration have passed at 512 pixels.

The native text package uses Q8_0 for the 36-layer Qwen3-VL tower. Its
15 × 4096 red-cube text output differs from the BF16 checkpoint CPU reference
by 7.589% relative L2, cosine 0.99712. The text package is
`qwen21-text-kernels-q8/`; this establishes execution, not prompt-quality
equivalence.

`qwen21-packed-transformer-full/packed_weights.json` records all 136 DiT
matrix slots and `full_model=true`. `qwen21_image_edge_run.py` runs the real
entry and velocity head. From a real Q8 text embedding, its text/image entry
errors are 1.054%/0.925% relative L2 against independent BF16 CPU execution;
the final head is 0.867%, cosine 0.999962. The 32-layer native stack executes
768 NPU operations. Layer 31 differs by 4.259% relative L2, cosine 0.999134
from the independent BF16 stack with identical entry and timestep. Fixtures:
`qwen21-image-entry-fixture-v2/`, `qwen21-native-stack-real-cond/`, and
`qwen21-image-final-fixture-v2/`.

`qwen21_native_dit_sample.py` follows the checkpoint's FlowMatch Euler
exponential dynamic shift and terminal stretch at 1024 tokens for a 512-pixel
image. Its four-step fixture is `qwen21-native-4step-latents/`: final latent
RMS 1.57527, all finite. The corrected full 40-step fixture is
`qwen21-native-40step-latents-v3/` using raw native Qwen3-VL text embeddings;
all steps were finite and the final latent RMS was 1.37692. The full native
VAE decoded it into `qwen21-vae-full-40step-v3/native-qwen21.png`: a clean red
cube on a light table. An earlier `v2` 40-step attempt fed projected image-entry
text into the entry projection again and produced visual noise. The sampler
now checks text-tower fixture metadata and rejects that input before NPU work.

## Still-image VAE contract

`qwen21_vae_plan.py` validates the downloaded FP32 checkpoint and emits the
first-frame decode topology at 512 pixels in
`qwen21-vae-decoder-plan-512.json`. It records **150 logical operations**,
**128 active checkpoint tensors**, and six unused `time_conv` tensors. The
published `_decode` passes a feature cache and `first_chunk=True` for the
first frame. Three `upsample3d` stages skip `time_conv` on that first frame,
but each residual up block still adds a `DupUp3D` first-frame shortcut.
The actual still-image path therefore needs `post_quant_conv` 64→64,
`decoder.conv_in` 64→1152, two mid residual blocks, a 1152-dimensional
single-head attention, five stages with widths 1152/1152/576/288/144,
four spatial doubling operations, channel RMSNorm and SiLU, residual adds,
and a 144→4 RGBA output convolution. `latents_mean` and `latents_std` apply
before VAE decode. The decoder output is clamped to [-1,1].

`qwen21_vae_pack.py` has packed all 82 active parameter groups (46 convolutions
and 36 norms, 399,211,008 bytes) in
`qwen21-vae-packed-decoder-fused-affine/`. The latent mean/std affine is folded
into the native `post_quant_conv` weights. Four nearest-exact upsample plus
convolution stages use the native `dit_conv` phase-weight formulation.
`export_qwen21_vae_kernels.py` compiled all 20 unique 512-pixel convolution
streams into `qwen21-vae-conv-kernels-512/` and maps each of the 46 calls to
its stream. `qwen21_vae_ew` compiled 17 unique RMSNorm/SiLU and residual-add
streams in `qwen21-vae-ew-kernels-512/`. Convolution output channels
are padded to multiples of 128, so the 144-, 288-, and 576-channel norms use
physical widths 256, 384, and 640 with gamma scaled by the square root of
logical/physical width.

Native decoder probes now cover:

| Operation | Native result against CPU reference |
| --- | --- |
| FP32 checkpoint `decoder.conv_in`, 64→1152 at 32² | 0.749352% relative L2, zero border |
| Checkpoint `post_quant_conv` with packed weights | 0.673881% relative L2, zero border |
| First fused upsample convolution, 32→64 grid | 1.1132% relative L2, zero border |
| Third fused upsample convolution, 128→256 grid | 0.9110% relative L2, zero border |
| 512-pixel width-changing residual convolution, 288→144 | 1.0359% relative L2, zero border |
| RMSNorm 144 in physical 256 / 288 in 384 / 576 in 640 | 0.2756% / 0.2928% / 0.2913% relative L2, zero padded tails |
| RMSNorm 1152 | 0.1655% relative Frobenius error |
| Checkpoint `decoder.mid_block.attentions.0.to_qkv` projection | 0.8538% relative L2, zero border |
| Full 1152-wide, 1024-token attention from that native projection | 3.6243% relative L2, cosine 0.999379 on sampled rows |

The 144-channel vector is padded because the direct 144-wide elementwise
stream produced a wrong tail; an attempted generic scalar-tail fix exceeded
the AIE program-memory budget and was reverted. `dit_fa` now has a specialized
8-query-row, 1152-wide single-head kernel for this VAE. A random BF16 QKV
probe differed from CPU attention by 2.5763% relative L2, cosine 0.999669;
the checkpoint-projection probe above passed the same 5% gate. Rebuilding the
shared 128-wide variant after that change also passed its two-head regression
at worst 2.456% relative Frobenius error. Native convolution, RMSNorm, and
attention primitives are established. `qwen21_vae_shortcut.py` reproduces the
first-frame `DupUp3D` rearrangement exactly. `qwen21_vae_run.py` executes the
full **150-operation** decoder plan using **105 native NPU kernel calls**.
On the same four-step latent fixture, its output differs from the official
FP32 Diffusers VAE by **0.671008% relative L2**, cosine **0.999984503**, and
MAE **0.00438891**. The resulting `qwen21-vae-full-4step-v1/native-qwen21.png`
is a coherent centered red cube on a light background.

The DiT block now stages 1..64 text tokens. Three RTP values change with
prompt length: the q/k RoPE text offset and the text/image attention valid
lengths. `qwen21_prompt_kernels.py` compares each compiled 15-token stream to
its 16-token probe, requires that every changed word be the expected RTP value,
then patches a per-request instruction copy. This keeps causal masking and
image-token positions aligned with the actual prompt without rebuilding
kernels for every length.
At 15 tokens, its patched streams are byte-identical to the baseline streams;
at 16 tokens, they are byte-identical to all three independently compiled
probe streams.

## Local CLI and API

The local asset configuration is
`/home/cyrus/Documents/ChatGPT/current/work/phlegm-image-models/phlegm-qwen21-image.json`.
It points to the downloaded checkpoint, Q8 native text tower, complete DiT
pack, VAE pack and streams, and three 16-token RTP probe streams. Run from the
repository root:

```bash
target/debug/phlegm image --config /home/cyrus/Documents/ChatGPT/current/work/phlegm-image-models/phlegm-qwen21-image.json \
  --prompt 'a red cube on a white table' --out /tmp/phlegm-qwen21-red-cube --steps 40
target/debug/phlegm serve-images --config /home/cyrus/Documents/ChatGPT/current/work/phlegm-image-models/phlegm-qwen21-image.json
```

The server exposes `GET /v1/models` and `POST /v1/images/generations` with
`model: "Qwen/Qwen-Image-2.1"`. The native route supports 512×512 output,
2..50 steps (default 40), and prompts yielding 1..64 text tokens. It rejects
negative prompts and guidance scale, which this checkpoint pipeline does not
use. The published scheduler's terminal stretch is undefined for one step.
`qwen21_native_generate.py` performs native text encoding, prompt RTP
patching, DiT sampling, and native VAE decoding in sequence.
The CLI completed a four-step generation from a 21-token prompt and produced
`qwen21-cli-21token-4step-v2/decode/native-qwen21.png`, a coherent red cube.
The API returned HTTP 200 with one base64-encoded 512×512 PNG for a separate
16-token, two-step prompt. A one-step request returned HTTP 400 before NPU
execution. `GET /v1/models` returned the Qwen model ID.

## Editing status (2026-10-31)

The published edit path encodes the condition image *inside* the Qwen3-VL
text encoder (`<|vision_start|><|image_pad|><|vision_end|>` slots, vision
tower with deepstack taps 8/16/24, M-RoPE) and also VAE-encodes it; each
image-pad slot becomes a 2x2 group of condition latents in the DiT sequence,
with t=0 modulation for text and condition tokens. None of the vision tower,
LM image-embedding input or VAE encoder runs natively yet, so
`/v1/images/edits` and `phlegm image --image` reject Qwen with HTTP 400 /
an error. The Gemma 3 SigLIP NPU kernels share the tower's 1152/16x72/4304
geometry; see the FLUX 3 spec and the image plan for the proposed route.

## Editing pieces (2026-11-01)

Decision: the Qwen3-VL text encoder's image embeddings (vision tower with deepstack)
will enter a batched (block) prefill on Phlegm's own NPU kernels; no
`open_qwen36_cli` option.

**VAE encoder** (`qwen21_vae_encoder.py`, reference `qwen21_vae_encoder_reference.py`).
First frame of `AutoencoderKLQwenImage21.encode` (RGBA in [-1, 1], argmax latents,
`(mu - latents_mean) / latents_std`): 117 operations, 84 native NPU runs over 23
`dit_conv` streams, 17 `qwen_vae_ew` streams and a 768-wide single-head `dit_fa`
(`DF_VAE_DIM=768`). Stride-2 downsamples are host space-to-depth plus a 3x3 conv with
zero -1 taps; `AvgDown3D` on one frame (one zero frame in front) is a fixed 1x1 conv on
the space-to-depth grid; `conv_out`, the mean half of `quant_conv` and the latent
normalization are one folded 3x3 conv. Six `downsampler.time_conv` tensors are inactive.

| Check (512 px red cube) | Result |
| --- | --- |
| Plan emulated in float64 vs published FP32 encoder | <= 2.1e-5 rel L2 every stage |
| 768-wide attention, random QKV | 2.739% rel L2, cos 0.999626 |
| RMSNorm+SiLU, fp32 Pade tanh (C 96/128, 512^2) | 0.387% / 0.356% (bf16 LUT: 1.148% / 1.340%) |
| Native encoder vs published, conv_in / down 0..4 / mid | 0.78 / 8.5 / 16.6 / 31.7 / 34.2 / 29.7 / 17.8% |
| Native latents vs published | 11.12% rel L2, cos 0.993824 |

The latent error is `dit_conv`'s BFP16 activation quantization (8 channels share an
exponent): emulating it reproduces the native stages (latents 11.03%), while BFP16
weights with BF16 activations give 1.56% and a two-pass hi/lo activation split 1.50%.
The encoder is not yet accurate enough for an edit gate.

**Condition-latent layout.** Published order [text][condition, 4x the pad slots][text]
[target]. `qwen21_attention_plan.segment_passes` (one causal pass through the last text
row, one bidirectional prefix pass per image block) equals the published block-causal
mask exhaustively. `qk_qwen21` takes `cond_row`: text advances all three axes, each
image block freezes the frame on a centred grid, and text after a block resumes
max(h, w) later; NPU probe 0.34-0.37% per segment. Text and condition rows use the
t = 0 modulation row. `qwen21_edit_block_run.py` ran block 0 of a published-transformer
fixture (`qwen21_edit_layout_reference.py`; text states without the vision tower):
0.908% rel L2, cos 0.999970; segments 1.13 / 0.97 / 0.54 / 1.00%.

## Remaining gates

1. Vision tower and batched image-embedding prefill; two-pass `dit_conv`
   activations for the VAE encoder; the 32-layer edit stack, per-length RTP patching
   of the edit streams, sampler, then the API and CLI editing routes.
2. Export and validate 1024-pixel kernels and longer text buckets. The current
   registered route explicitly supports 512 pixels and 1..64 text tokens.

Run NPU work, builds, and large downloads under Dibs. No hosted inference or
paid resources are part of this implementation.
