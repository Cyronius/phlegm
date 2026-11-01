"""Capture pinned FLUX 3 prompt embeddings for native DiT numerical validation.

This is a CPU reference fixture, not Phlegm's native prompt encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-src", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.reference_src.resolve() / "src"))
    from flux3.text_encoder import OUTPUT_LAYERS, MAX_LENGTH, PAD_MULTIPLE
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    torch.set_num_threads(8)
    model_path = str(args.model.resolve() / "text_encoder")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True)
    real = tokenizer(rendered, return_tensors="pt", truncation=True,
                     max_length=MAX_LENGTH)["input_ids"].shape[1]
    length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
    inputs = tokenizer(rendered, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=length)
    mask = inputs["attention_mask"]
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
    with torch.inference_mode():
        output = model(input_ids=inputs["input_ids"], attention_mask=mask,
                       position_ids=positions[None].expand(4, -1, -1),
                       output_hidden_states=True, use_cache=False)
    stacked = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
    tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
    p = torch.arange(length)
    ids = torch.stack([torch.zeros_like(p)] * 3 + [p], dim=-1)
    if (tokens.ndim != 2 or tokens.shape[1] != 20480 or tokens.shape[0] % 80 or
            not torch.isfinite(tokens.float()).all()):
        raise ValueError("published FLUX 3 text encoder returned invalid tokens")
    np.asarray(tokens.contiguous().view(torch.int16).numpy(), dtype="<u2").tofile(out / "tokens.bf16")
    ids.float().numpy().astype("<f4").tofile(out / "ids.f32")
    (out / "fixture.json").write_text(json.dumps({
        "model": "black-forest-labs/flux-3-image-klein",
        "scope": "published CPU text encoder fixture; native NPU text encoder outstanding",
        "prompt": args.prompt, "text_length": tokens.shape[0],
        "channels": tokens.shape[1],
    }, indent=2) + "\n")
    print(f"published FLUX 3 text encoder: {tuple(tokens.shape)} -> {out / 'tokens.bf16'}")


if __name__ == "__main__":
    main()
