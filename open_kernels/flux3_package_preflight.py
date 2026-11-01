"""Check that the pinned FLUX 3 reference, encoder, VAE and PUP are present.

This is an acquisition gate, not an inference backend. It deliberately refuses
to call a checkpoint ready when the separate prompt upsampler is absent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from flux3_weight_layout import inspect as inspect_dit
from image_safetensors import read_header


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _tensors(files: list[Path]) -> tuple[int, int, set[str]]:
    if not files:
        raise ValueError("missing safetensors weights")
    keys: set[str] = set()
    count = 0
    params = 0
    dtypes: set[str] = set()
    for path in files:
        for name, entry in read_header(path).items():
            if name in keys:
                raise ValueError(f"duplicate tensor {name} in {path}")
            keys.add(name)
            count += 1
            params += math.prod(entry["shape"])
            dtypes.add(entry["dtype"])
    return count, params, dtypes


def inspect(checkpoint: Path, pup: Path, reference: Path) -> dict:
    dit = inspect_dit(checkpoint)
    config = _json(checkpoint / "config.json")
    te = _json(checkpoint / "text_encoder" / "config.json")
    te_text = te.get("text_config", {})
    if (config.get("text_encoder", {}).get("layers") != [4, 8, 12, 16, 20, 24, 28, 32]
            or te.get("model_type") != "qwen3_vl"
            or te_text.get("hidden_size") != 2560
            or te_text.get("num_hidden_layers") != 36):
        raise ValueError("FLUX 3 Qwen3-VL encoder differs from the pinned reference")
    te_weights = _tensors(sorted((checkpoint / "text_encoder").glob("*.safetensors")))
    if te_weights != (713, 4_437_815_808, {"BF16"}):
        raise ValueError(f"FLUX 3 encoder weight contract differs: {te_weights}")
    vae_weights = _tensors([checkpoint / "ae.safetensors"])
    if vae_weights != (251, 84_046_372, {"F32", "I64"}):
        raise ValueError(f"FLUX 3 VAE weight contract differs: {vae_weights}")

    pup_config = _json(pup / "config.json")
    pup_text = pup_config.get("text_config", {})
    pup_vision = pup_config.get("vision_config", {})
    if (pup_config.get("model_type") != "qwen3_5"
            or pup_text.get("hidden_size") != 4096
            or pup_text.get("num_hidden_layers") != 32
            or pup_vision.get("depth") != 27
            or pup_text.get("layer_types") !=
            ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(32)]):
        raise ValueError("FLUX 3 PUP architecture differs from the pinned reference")
    pup_weights = _tensors(sorted(pup.glob("*.safetensors")))
    if pup_weights != (760, 9_409_813_744, {"BF16"}):
        raise ValueError(f"FLUX 3 PUP weight contract differs: {pup_weights}")
    prompts = _json(pup / "system_prompts.json")
    if set(prompts) != {"t2i", "i2i", "t2i_bbox", "i2i_bbox"} or any(
        not isinstance(value, str) or not value.strip() for value in prompts.values()
    ):
        raise ValueError("FLUX 3 PUP system prompts are missing")
    for path in (checkpoint / "text_encoder" / "tokenizer.json",
                 pup / "tokenizer.json"):
        if not path.is_file():
            raise ValueError(f"missing tokenizer: {path}")
    for relative in ("src/flux3/generate.py", "src/flux3/pipeline.py",
                     "src/flux3/text_encoder.py", "src/flux3/pup.py",
                     "src/flux3/dit.py", "src/flux3/autoencoder.py"):
        if not (reference / relative).is_file():
            raise ValueError(f"missing FLUX 3 reference source: {relative}")
    return {"dit": dit, "encoder_tensors": te_weights[0], "vae_tensors": vae_weights[0],
            "pup_tensors": pup_weights[0], "pup_prompts": sorted(prompts),
            "reference_source": str(reference.resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pup", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.checkpoint, args.pup, args.reference), indent=2))
