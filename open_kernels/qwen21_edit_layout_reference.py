"""Published Qwen Image 2.1 transformer on an edit's condition-latent layout (CPU reference).

``QwenImage21Transformer2DModel.forward`` (BF16, unchanged) runs one edit step: the
published ti2i template with one condition image, ``img_mask`` marking its
``<|image_pad|>`` slots plus the target slots, ``img_shapes`` [(1, h, w) condition,
(1, h, w) target], and ``hidden_states`` = [condition latents; target latents].

The text hidden states come from the checkpoint's Qwen3-VL text model on the template's
token ids with the pad slots left as plain tokens: the vision tower (not native yet)
is not run, so these are realistic text states but not the published edit's prompt
embedding. The DiT discards the pad-slot rows and drops the condition latents into
them, so this fixture pins the DiT side of the layout: entry, 4x slot expansion, 3-axis
RoPE with the after-block gap, t = 0 modulation for text and condition tokens, the
block-causal mask, and the velocity.

Captures (float32): ``prompt_embeds``, ``txt_in``, ``img_in``, ``block0_input`` (the
joint sequence in published order), ``block0_output``, ``block31_output``, ``output``,
``modulation`` [2, 4*4096] (rows: sampled t, then t = 0) and ``layout.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SYSTEM = "Comprehend and analyze the provided prompt."


def template_ids(tokenizer_json: Path, prompt: str, slots: int) -> tuple[list[int], int, int]:
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(tokenizer_json))
    prefix = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    chat = (prefix + "<|im_start|>user\n<image1><|vision_start|>" + "<|image_pad|>" * slots +
            f"<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n")
    return (t.encode(chat, add_special_tokens=False).ids,
            len(t.encode(prefix, add_special_tokens=False).ids), t.token_to_id("<|image_pad|>"))


def main() -> None:
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json")
    ap.add_argument("--cond-latents", type=Path, required=True,
                    help="normalized [h, w, 64] float32 condition latents (VAE encoder)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--timestep", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hw = a.grid * a.grid
    ids, drop, pad = template_ids(a.tokenizer, a.prompt, hw // 4)

    text = Qwen3VLForConditionalGeneration.from_pretrained(
        str(a.model / "text_encoder"), torch_dtype=torch.bfloat16, local_files_only=True).eval()
    captured = []
    lm = getattr(text.model, "language_model", text.model)
    handle = lm.norm.register_forward_pre_hook(lambda _m, i: captured.append(i[0].detach()))
    with torch.inference_mode():
        text(input_ids=torch.tensor([ids]), use_cache=False)
    handle.remove()
    prompt_embeds = captured[0][:, drop:].to(torch.bfloat16)          # [1, n_vlm, 4096]
    del text
    vlm_mask = torch.tensor([[i == pad for i in ids[drop:]]])
    n_vlm = prompt_embeds.shape[1]

    cond = np.fromfile(a.cond_latents, "<f4").reshape(hw, 64)
    rng = np.random.default_rng(a.seed)
    target = rng.standard_normal((hw, 64)).astype(np.float32)
    latents = torch.from_numpy(np.concatenate((cond, target))).to(torch.bfloat16)[None]
    img_mask = torch.cat([vlm_mask, torch.ones(1, hw // 4, dtype=torch.bool)], 1)

    model = QwenImage21Transformer2DModel.from_pretrained(
        str(a.model / "transformer"), torch_dtype=torch.bfloat16, local_files_only=True).eval()
    caps = {}

    def save(name):
        def fn(_m, _i, o):
            caps[name] = (o[0] if isinstance(o, tuple) else o).detach().float().numpy()
        return fn

    model.txt_in.register_forward_hook(save("txt_in"))
    model.img_in.register_forward_hook(save("img_in"))
    model.modulation.register_forward_hook(save("modulation"))
    model.transformer_blocks[0].register_forward_hook(
        lambda _m, _a, kw, o: caps.__setitem__("block0_input", kw["hidden_states"].detach().float().numpy()),
        with_kwargs=True)
    model.transformer_blocks[0].register_forward_hook(save("block0_output"))
    model.transformer_blocks[-1].register_forward_hook(save("block31_output"))
    with torch.inference_mode():
        output = model(hidden_states=latents, encoder_hidden_states=prompt_embeds,
                       encoder_hidden_states_mask=torch.ones(1, n_vlm, dtype=torch.bool),
                       timestep=torch.tensor([a.timestep], dtype=torch.bfloat16),
                       img_shapes=[[(1, a.grid, a.grid), (1, a.grid, a.grid)]],
                       img_mask=img_mask, return_dict=False)[0]
    caps["output"] = output.float().numpy()
    caps["prompt_embeds"] = prompt_embeds.float().numpy()
    n1 = int(vlm_mask[0].int().argmax())
    n2 = n_vlm - n1 - hw // 4
    shapes = {}
    for name, value in caps.items():
        value = value[0] if value.ndim == 3 else value
        value.astype("<f4").tofile(out / f"{name}.f32")
        shapes[name] = list(value.shape)
    latents[0].float().numpy().astype("<f4").tofile(out / "latents.f32")
    (out / "layout.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "scope": __doc__.split("\n\n")[0],
        "prompt": a.prompt, "timestep": a.timestep, "seed": a.seed, "grid": a.grid,
        "segments": [["text", n1], ["image", hw], ["text", n2], ["image", hw]],
        "text_before": n1, "text_after": n2, "vlm_tokens": n_vlm, "joint": n1 + n2 + 2 * hw,
        "shapes": shapes, "vision_tower": "not run (pad slots as plain tokens)",
    }, indent=2) + "\n")
    print(f"Qwen edit-layout reference: text {n1}+{n2}, joint {n1 + n2 + 2 * hw} -> {out}")


if __name__ == "__main__":
    main()
