# FLUX 3 Image Klein native NPU implementation

Checkpoint: [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)
revision `337337621b8526e120f2275d250242f4a6bedd75`; reference source
[flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein)
at `bfc8889a4497f846416f253a3972a1941ac698b9` (`src/flux3/pipeline.py`,
`dit.py`, `sampling.py`, `autoencoder.py`). Weights, packed weights, kernels
and captures live in
`/home/cyrus/Documents/ChatGPT/current/work/phlegm-image-models/`, outside Git.
The working log is `.plans/native-image-models.md`.

## One-reference editing contract

The published pipeline conditions an edit through the DiT's `image_cond`
mode. Phlegm reproduces it on the NPU as follows:

- **Reference preparation** (`flux3_native_generate.prepare_reference`):
  the published `prepare_reference` (EXIF transpose, RGB, fit under 4 MiB
  pixels, centred reflect padding to the 16-pixel grid, crop box kept). The
  native kernels cover a 512×512 canvas, so other canvases are refused; the
  output defaults to the reference size, as the published `default_size`.
- **Reference latents**: the native 89-operation VAE encoder
  (`flux3_vae_encoder_run`) produces the BatchNorm-normalized
  [1024,128] tokens of `AutoEncoder.encode`.
- **Entry**: `emb_in.image_cond` runs once per request on the compiled
  `emb.image` stream (identical shape) with the `image_cond` weight.
- **Modulation**: reference tokens have timestep 0, so their vector is
  `time_in(0) + ref_time_in(target)`, the text row of the conditioning
  pass. `early/single_stream_modulations.image_cond` run on the image-mode
  modulation streams with their own weights (15 operations per step).
- **Per-mode blocks**: five `content_mode_blocks.image_cond` blocks per step
  (their modulation follows the flow-map target). Only the q/k stream differs
  from the image blocks; its RoPE ids are the published `(0, h/2, w/2, -1)`.
- **Shared blocks**: one sample of text + image + reference rows (2128 or
  2208 live rows in a 2560-row stream), per-mode modulation and residual
  gates, full attention. `qk_flux3` takes `ref_row`: image rows at or past it
  restart `h` and carry `l = -1`.
- Only the generated image's rows feed the final layer, as published.

The I2I prompt upsampler needs its 27-layer vision tower, which has no native
NPU route yet. Edits therefore pass the prompt as typed; this is the
published `generate(..., pup=False)` path, recorded as `"upsampled": false`.

## Kernel notes

- `dit_ew` RTP slot `R_REF_ROW` (default 64, disabled) carries `ref_row`.
- Core program memory is 16,384 B. The generic `dit_ew` program (ln/res/qk/bin/
  SiLU) is 15,952 B; Qwen's GELU-tanh is a separate `gelu_row` entry linked only
  by `gelu_tanh` streams (inlined into `un_row` it pushed the program to 16,880 B
  and every generic rebuild failed in `XAie_LoadElf`; fixed 2026-11-01). Sets
  other than the 80/160 text-to-image buckets use `"lean": 1` (ln_mod,
  res_ln_mod, binary rows only); both programs build. A from-scratch generic
  80-token shared set is bit-identical on the NPU to the original.
  `export_flux3_block.py --reuse DIR` copies identically specified compiled sets.
- `designs/dit_ew/compare.py` scores the live rows of the q/k streams and checks
  that padded rows are exactly zero.

## Hardware results (Strix Halo NPU, 512 px, 16 steps, alpha 16)

| Check | Result |
| --- | --- |
| `qk_flux3` with `ref_row=32`, 80 text + 1024 image + 1024 reference rows | rel L2 0.249% text, 0.259% image, 0.267% reference rows (independent float64 RoPE); padded rows zero |
| Native VAE encoder vs published FP32 encoder, real 512 px input | tokens 9.96% rel L2, cosine 0.99503 |
| Native text taps vs published BF16 encoder, 14 real tokens | 5.55% rel L2, cosine 0.99882 |
| One DiT pass on native step-0 inputs vs published `DiT.forward` | image-early 3.46%, reference-early 4.32%, text-early 4.83%, shared image rows 5.67% (cos 0.99840), velocity 4.13% (cos 0.99928) |
| Same at step 8 (t=0.941) | reference-early 4.42%, shared image rows 5.16%, velocity 2.36% (cos 0.99973) |
| Same at step 15 (t=0.516) | reference-early 4.02%, shared image rows 14.06%, velocity 12.74% (cos 0.99191) |
| Full native edit vs unchanged published `Pipeline.generate` (CPU, same seed-7 noise, same input and prompt) | MAE 2.33 / 255, PSNR 35.86 dB |

Shared-stack relative L2 over all rows (31–53%) is dominated by text pad rows
with activations near 6e6; the image rows that feed the velocity are listed
separately. The CLI (`phlegm image --image`) and `POST /v1/images/edits`
(multipart, HTTP 200 in 337.5 s) both completed real 16-step NPU edits.

## 1024 pixels (text-to-image)

Kernel sets: `flux3-block-kernels-1024`, `flux3-shared-block-kernels-{80,160,240}-1024`,
`flux3-image-edge-kernels-{80,160,240}-1024`, `flux3-text-block-kernels-{80,160,240}-1024`
(same specs as 512) and `flux3-vae-kernels-1024`. The shared stream is 4608 rows
(4176-4336 live). The flow-map schedule and RoPE scale do not depend on resolution.
Editing stays at 512: `qk_flux3` stacks the reference under the image grid (grid_h <= 64).
The Rust config takes them as `resolutions["1024"]`; `size=1024x1024` selects them.

| Check (16 steps, seed-7 noise, 43-token prompt, pup=False) | Result |
| --- | --- |
| Native vs published `Pipeline.generate` (CPU) | MAE 5.02 / 255, PSNR 26.44 dB, same composition |
| One pass at steps 0 / 8 / 15, velocity | 6.64 / 2.42 / 14.41% rel L2 (cos 0.99857 / 0.99973 / 0.98957) |
| Same, image-early / text-early / shared image rows | 3.2-3.8% / 3.1-4.0% / 5.4-15.1% |
| Native VAE decode vs official FP32, identical latents | MAE 1.02 / 255, PSNR 44.69 dB |
| CLI `phlegm image --size 1024x1024` (upsampled, 160 bucket) | 1024x1024 PNG |

The generations API at 1024 shares the CLI path but has not been run on hardware.

## Text buckets

The DiT kernel sets cover 80, 160 and 240 text rows (and 320-480 compiled).
The native text encoder's pad masking (`open_qwen36_cli --tap-pad-start`)
requires the padded sequence to fit one 256-token prefill block, so 240 is
the longest usable bucket. A 236-token caption at 240 rows ran 16 steps on
the NPU: versus the published pipeline (BF16 text encoder, same noise) MAE
10.07/255, PSNR 21.99 dB, same composition; step-0 velocity 12.93% rel L2,
cosine 0.99289, on identical native inputs.

One-reference edit at the 240 bucket (214-token instruction, 2026-11-01): native vs
published `Pipeline.generate` MAE 1.84 / 255, PSNR 38.89 dB; one-pass velocity at steps
0 / 8 / 15 4.29 / 2.37 / 11.97% (cos 0.99924 / 0.99972 / 0.99282); reference-early
4.0-4.4%.

Image-embedding prefill (decided 2026-11-01): the upsampler's I2I path will feed
vision-tower embeddings into a batched (block) prefill on Phlegm's own NPU kernels,
reusing the block-prefill GEMM routes; no option is added to `open_qwen36_cli`.

Do not register FLUX 3 editing as complete until the upsampler's I2I path
(vision tower) is native and multi-reference/aspect-ratio cases are covered.
