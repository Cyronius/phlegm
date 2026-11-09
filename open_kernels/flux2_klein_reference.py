r"""flux2_klein_reference: the unchanged published diffusers ``Flux2KleinPipeline`` on
CPU, with its inputs pinned and its intermediates saved. This is the reference that
Phlegm's native FLUX.2 [klein] route is compared against (not the OFLM engine and not
q4nx bundles).

Saved in --out:
    ids.npy, mask.npy             the prompt's 512 token ids and attention mask
    prompt_embeds.npy             [512, 3 * hidden] text conditioning (f32 of the dtype run)
    noise.npy                     [128, h/16, w/16] initial latents (bf16-representable f32)
    sigmas.npy, timesteps.npy     the scheduler's schedule after dynamic shifting
    latents_step<k>.npy           packed latents [h*w/256, 128] after step k
    reference_latents.npy         an edit's packed reference tokens
    published.png                 the decoded image
    summary.json                  revision, arguments, hashes, timings

Run with the reference venv (diffusers with Flux2KleinPipeline), e.g.
    $WORK/.venv/bin/python open_kernels/flux2_klein_reference.py \
        --model $WORK/flux2-klein-4b --prompt "A red fox in fresh snow" --size 512 \
        --steps 4 --seed 1 --out $WORK/flux2-klein-4b-published-fox-512-4step
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--image", type=Path, help="reference image for an edit")
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--noise", type=Path, help="initial latents [128, h, w] (.npy) to reuse")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from diffusers import Flux2KleinPipeline
    from PIL import Image

    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(a.model, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    t_load = time.time() - t0

    # Text conditioning through the pipeline's own encoder, saved with its token ids.
    messages = [{"role": "user", "content": a.prompt}]
    text = pipe.tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
    tok = pipe.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                         max_length=512)
    np.save(a.out / "ids.npy", tok["input_ids"][0].numpy().astype(np.int32))
    np.save(a.out / "mask.npy", tok["attention_mask"][0].numpy().astype(np.int32))
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(prompt=a.prompt, device="cpu",
                                              num_images_per_prompt=1, max_sequence_length=512)
    np.save(a.out / "prompt_embeds.npy", prompt_embeds[0].float().numpy())

    lat_hw = a.size // 16
    if a.noise:
        noise = torch.from_numpy(np.load(a.noise)).float()
    else:
        g = torch.Generator("cpu").manual_seed(a.seed)
        noise = torch.randn((128, lat_hw, lat_hw), generator=g, dtype=torch.float32)
    noise = noise.to(torch.bfloat16).float()          # what a bf16 pipeline holds
    np.save(a.out / "noise.npy", noise.numpy())

    steps: list[np.ndarray] = []

    def on_step(_pipe, i, t, kw):
        lat = kw["latents"][0].float().numpy()
        np.save(a.out / f"latents_step{i}.npy", lat)
        steps.append(np.asarray(float(t)))
        return kw

    kwargs = {}
    if a.image:
        kwargs["image"] = Image.open(a.image).convert("RGB")
        # Save the packed reference tokens exactly as the pipeline builds them.
        orig = pipe.prepare_image_latents

        def capture(*args, **kw):
            lat, ids = orig(*args, **kw)
            np.save(a.out / "reference_latents.npy", lat[0].float().numpy())
            np.save(a.out / "reference_ids.npy", ids[0].numpy())
            return lat, ids
        pipe.prepare_image_latents = capture

    t1 = time.time()
    with torch.no_grad():
        image = pipe(prompt_embeds=prompt_embeds, height=a.size, width=a.size,
                     num_inference_steps=a.steps, guidance_scale=1.0,
                     latents=noise[None].to(dtype), callback_on_step_end=on_step,
                     **kwargs).images[0]
    t_gen = time.time() - t1
    image.save(a.out / "published.png")
    np.save(a.out / "timesteps.npy", np.array(steps, dtype=np.float32))
    np.save(a.out / "sigmas.npy", pipe.scheduler.sigmas.float().numpy())

    model_index = json.loads((a.model / "model_index.json").read_text())
    integrity = a.model / "checkpoint-integrity.json"
    summary = {
        "pipeline": "diffusers.Flux2KleinPipeline (unchanged)",
        "diffusers": __import__("diffusers").__version__, "torch": torch.__version__,
        "model": str(a.model), "is_distilled": model_index.get("is_distilled", False),
        "revision": json.loads(integrity.read_text())["revision"] if integrity.exists() else None,
        "prompt": a.prompt, "size": a.size, "steps": a.steps, "seed": a.seed,
        "dtype": a.dtype, "image": str(a.image) if a.image else None,
        "image_sha256": sha256(a.image) if a.image else None,
        "valid_tokens": int(tok["attention_mask"].sum()),
        "published_png_sha256": sha256(a.out / "published.png"),
        "seconds": {"load": round(t_load, 1), "generate": round(t_gen, 1)},
        "threads": torch.get_num_threads(),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
