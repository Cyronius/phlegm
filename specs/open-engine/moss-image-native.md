# MOSS Image SFT native NPU path

Checkpoint: `OpenMOSS-Team/MOSS-Image-SFT`, revision
`f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`. The pinned
`pipeline_moss_image.py` is the arithmetic reference. The model is text to
image; its published pipeline has no image input.

Phlegm's current native route has a Q4NX Qwen3.5 text encoder, checkpoint
weighted text refinement, image embedding, timestep modulation, 42 joint DiT
blocks, output velocity head and VAE decoder. NumPy performs only token/input
preparation and the reference's FP32 CFG/Euler latent update. The default
negative CFG branch omits zero-based block 22. The sampler is
`open_kernels/moss_native_sample.py`; no Diffusers inference executes in that
runtime. The reference modules are offline numeric oracles.

The sampler derives ordinary prompt-length kernel variants from independently
exported 10-token and 17-token witnesses in the same padded NPU bucket. It
checks every source file hash and stream shape, patches only the repeated q/k
and attention RTP length words, and publishes a complete variant atomically.
The checkpoint, packed weights, kernel
sets and generated images live outside Git. `phlegm image --config JSON` and
`phlegm serve-images --config JSON` use an explicit local asset configuration.
On shared hosts it can name a Dibs launcher, which reserves `npu` and `cpu`
for each generation. Reserve the API listening port separately with Dibs.
The API endpoint is `POST /v1/images/generations`; it accepts one prompt and
returns `data[0].b64_json`. It supports 512×512 and 1024×1024 square requests,
one image per request, 1–100 steps, a seed, guidance scale, and a negative
prompt. A given configuration must contain kernel sets for the requested
resolution and token counts. The 1024-pixel route has not yet passed a full
image test.

## Numeric gates on Strix Halo

- One 512-pixel, 17-text-token joint block: 10 NPU operations, 2.526% relative
  L2 and 0.999665 cosine against the checkpoint BF16 block.
- All 42 blocks from native image/text/time stages: 420 NPU operations. The
  final velocity differs by 2.974% relative L2 and has 0.999755 cosine from
  the full one-step BF16 reference built from the same encoder hidden rows and
  initial latents.
- Empty negative prompt: 10 real text tokens. The skipped-block branch runs
  410 NPU operations; its final hidden state differs by 4.260% relative L2
  and has 0.999091 cosine from the published BF16 block stack with block 22
  omitted.
- The complete 512-pixel VAE decoder runs 102 NPU operations. From identical
  random latents its output differs from the FP32 Diffusers decoder by 0.772
  RGB mean absolute levels, with 47.67 dB PSNR.
- A one-step guidance-1 and a one-step guidance-3 sampler run each emitted a
  512×512 PNG through the NPU. Their diffuse images are execution smoke tests.
- The default 50-step, guidance-3 native run completed both CFG branches per
  step and 102 VAE operations. For `a red cube on a white table`, seed 41, it
  produced a coherent red geometric block on a light background. The final
  latent RMS was 0.809319, and no step diverged. Output:
  `work/phlegm-image-models/moss-sample-native-50step-512/decode/native-vae.png`.
- A 13-token prompt passed an end-to-end `phlegm image` NPU run using derived
  kernels. Its q/k and attention instruction streams were byte identical to
  independent direct 13-token exports. A separate valid image API request
  returned HTTP 200 with one base64 encoded 512×512 PNG.

The short prompt's native Q4/Q8 text hidden states differ from the BF16
checkpoint encoder by 18.535% relative L2, although the NPU agrees with its
exact packed-weight CPU replica within 0.5224% relative L2. A 290-token
prompt shows 57.912% relative L2 to the BF16 encoder. Long-context quality
and 1024-pixel output remain unvalidated. The 512-pixel route is operational;
unsupported resolutions are rejected before generation.
