"""Which ModelSpec a design build / packer run uses.

    OPEN_KERNELS_SPEC=<file.json>   an explicit spec (export_qwen36_kernels.py sets it)
    otherwise                        recipes/specs/qwen36-35b-a3b.json, the checked-in 27B

`spec_from_model_dir` derives one from a model directory's config.json, with
the real vocab from tokenizer.json when it is there.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .spec import ModelSpec

HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "specs" / "qwen36-35b-a3b.json"


def load_spec(path: Path) -> ModelSpec:
    return ModelSpec.from_json(Path(path).read_text(encoding="utf-8"))


def default_spec() -> ModelSpec:
    return load_spec(DEFAULT_SPEC)


def current_spec() -> ModelSpec:
    p = os.environ.get("OPEN_KERNELS_SPEC")
    return load_spec(Path(p)) if p else default_spec()


def tokenizer_vocab(tokenizer_json: Path) -> int | None:
    """The tokenizer's id count: max id over model.vocab and added_tokens, + 1."""
    try:
        t = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ids = list(t.get("model", {}).get("vocab", {}).values()) + [a["id"] for a in t.get("added_tokens", [])]
    return max(ids) + 1 if ids else None


def spec_from_model_dir(model_dir: Path) -> ModelSpec:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    rv = tokenizer_vocab(model_dir / "tokenizer.json")
    spec = ModelSpec.from_hf_config(cfg, real_vocab=rv)
    spec.extra["model"] = model_dir.name
    return spec


def current_recipe(max_ctx: int = 4096):
    from .qwen36moe import recipe
    return recipe(current_spec(), max_ctx)
