# FLUX.2 [klein] native NPU route in Phlegm

Checkpoint: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` (Apache-2.0, step-distilled,
`is_distilled: true`). Reference: the unchanged diffusers `Flux2KleinPipeline`
(0.42.0.dev0, torch 2.14.1+cpu). Weights, packed weights, kernels and captures live in
`/home/cyrus/Documents/ChatGPT/current/work/phlegm-image-models/`, outside Git. The
working log is `.plans/flux2-klein-native.md`.

## Route

- Schedule: `open_kernels/klein_pipeline.py` (text encoder Qwen3-4B layers 1-27 with
  taps 9/18/27, conditioning, the denoising steps, the VAE decoder; an edit adds the VAE
  encoder and the reference tokens after the generated ones). Every op runs on the NPU;
  the host tokenizes, gathers the prompt's embedding rows, writes the noise (and an
  edit's reference pixels), patches `te_attn`'s valid length, and encodes the PNG.
- Kernels: `open_kernels/export_dit_kernels.py --resolutions 512,1024 --edits 512,1024`.
  The dit_gemm marker `dit_kernels.json` lists `resolutions` and `edits`; the server
  accepts exactly those sizes.
- Runner: `open_kernels/flux2_native_generate.py` on the pyxrt host
  (`open_kernels/harness/npu_host.py`), one hardware context per kernel set, reading
  the diffusers checkpoint directly (no q4nx bundle). It refuses a checkpoint whose
  transformer/text-encoder geometry differs from the schedule and undistilled (Base)
  checkpoints. Every queued `xrt::run` is kept alive until it completes: Linux XRT
  aborts the process when one is destroyed in flight.
- Server/CLI: `backend=flux2` in `phlegm-core/src/image.rs` (`Flux2ImageConfig`:
  `model_id`, `model`, `kernels`, `python`, `repo_root`, optional `env_script`/`dibs`,
  `output_root`). Steps 1-50 (default 4); `guidance_scale` and `negative_prompt` are
  refused.
- Noise: without `--noise`, numpy PCG64 from the seed; an edit uses the stream
  `[seed, 1]`. With the text-to-image stream, editing an image generated with the same
  seed reproduced the reference unchanged (measured below).

## Results (2026-11-08, Strix Halo XDNA2 (Ryzen AI MAX+ 395), Fedora 44, XRT 2.27.0 a29919c)

Native runs with the reference's exact initial latents, 4 steps, against the bf16
published pipeline. Pixel PSNR/MAE are on 8-bit RGB.

| case | final latents rel L2 / cosine | MAE / PSNR | NPU time (one run) |
|---|---|---|---|
| 512 t2i "A red fox in fresh snow" | 34.2% / 0.9411 | 10.33 / 21.97 dB | 5.00 s |
| 1024 t2i, same prompt | 36.8% / 0.9327 | 9.15 / 23.04 dB | 12.47 s |
| 512 edit "Make the fox blue" | 8.96% / 0.99598 | 1.73 / 39.20 dB | 7.24 s |
| 1024 edit, same | 8.37% / 0.99656 | 1.58 / 39.63 dB | 25.93 s |
| 512 edit "Replace the noodles with sushi" | 19.7% / — | — / 26.37 dB | — |

All images are coherent and follow their prompts; text-to-image keeps the reference's
composition with small pose/detail drift, which matches the OFLM study of this
arithmetic (all-NPU LPIPS 0.107 mean against bf16, `OPEN-DIFFUSION-QUALITY`; LPIPS
not measured here). NPU times are single runs under Dibs, not benchmarks.

`phlegm image` and `phlegm serve-images` each completed text-to-image and one-reference
edits at 512 and 1024 on hardware (HTTP 200; `/v1/models` lists
`black-forest-labs/FLUX.2-klein-4B`; `guidance_scale` -> 400).

## Swapped text encoder (2026-11-08)

The route takes any text encoder with Qwen3-4B's exact geometry: build a checkpoint
directory whose `text_encoder/` is the replacement and whose other entries link to the
published checkpoint, and point the config's `model` at it. Give it its own `kernels`
directory (links to the kernel set plus its own `packed/`), or the packed-weight cache
repacks every time the two are swapped. `embed_rows` rounds a non-BF16 embedding table to
bf16 (it used to copy the raw bits, which is wrong for an F16 file);
`specs/open-engine/tests/test_flux2_embed_rows.py` covers both.

Measured: `ponpoke/flux2-klein-4b-uncensored-text-encoder` at `633217e5`
(`model.safetensors` sha256 `3d6f27fd...`, matches the Hub). It is an F16 re-save of the
published encoder: same 398 tensors and shapes; layers 0-13 equal the published weights
after an f16 round trip; `o_proj` and `down_proj` changed in layers 14-35 (2.0-3.1%
relative L2 each). Its prompt embeddings for "A red fox in fresh snow" differ from the
published encoder's by 0.79/0.78/0.80% at taps 9/18/27. 512 text-to-image, 4 steps, the
reference's noise, against the unchanged diffusers pipeline with the same encoder:
latents 38.9% rel L2 / cosine 0.9238, pixels MAE 11.04 / 21.56 dB (published encoder:
34.2% / 21.97 dB, table above). `phlegm image` t2i and a "Make the fox blue" edit at 512
both produced the expected images. Edits and 1024 were not compared against the
reference. License per its card: flux-non-commercial-v2.1.

## Open

- FLUX.2 [klein] 9B (gated, non-commercial): config unread, so no schedule yet.
- Base (undistilled) variants: CFG, deferred by decision.
- LPIPS against the reference; prompts over 512 tokens are truncated as the published
  pipeline does.
