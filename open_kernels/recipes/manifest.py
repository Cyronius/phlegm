"""manifest.json: everything src/open_qwen36 reads about a kernel set.

Written by export_qwen36_kernels.py beside the six xclbin / insts pairs. The
driver derives every layout constant, context, kernel, per-layer program and
packing law from it -- no HID, no POOL_*, no lx0 / ax0 names in C++. A model
whose config.json disagrees with `hf_config_check` is refused at startup
(OPEN-MANIFEST).

    python -m recipes.manifest [--spec FILE | --model-dir DIR] [--out manifest.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import qwen36moe as Q
from .cache import build_key
from .load import spec_from_model_dir
from .spec import ModelSpec, hf_model_types

MANIFEST_VERSION = 1


def manifest(spec: ModelSpec, max_ctx: int = 4096, key: str | None = None) -> dict:
    R = Q.recipe(spec, max_ctx)
    L, C = R.layout, R.common
    prog = Q.programs(spec)
    m = {
        "manifest_version": MANIFEST_VERSION,
        "family": spec.family,
        "spec": spec.to_dict(),
        "spec_hash": spec.spec_hash(),
        "build_key": key if key is not None else build_key(spec),
        "max_ctx_default": max_ctx,
        "hf_config_check": {
            "model_type": hf_model_types(spec.family),
            "hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_experts": spec.num_experts, "num_experts_per_tok": spec.experts_per_tok,
            "moe_intermediate_size": spec.moe_intermediate, "head_dim": spec.head_dim,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "layer_types": list(spec.layer_types),
        },
        "layout": {
            "hidden": spec.hidden, "vocab": spec.vocab, "real_vocab": spec.real_vocab,
            "chunk_bytes": Q.CHUNK, "pool_bytes": L.POOL_BYTES,
            "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": Q.Q8_CHUNK,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
            "rout_idx_off": Q.ROUT_IDX_OFF,
            "moe": {"experts": spec.num_experts, "topk": spec.experts_per_tok,
                    "stripe": C.STRIPE, "up_bytes": C.UP_BYTES, "down_core": C.DOWN_PER_CORE * C.DOWN_BAND,
                    "pool_down": L.POOL_DOWN, "share_up": L.POOL_SHARE_UP, "share_gate": L.POOL_SHARE_GATE,
                    "share_down": L.POOL_SHARE_DOWN},
        },
        "layers": list(spec.layer_types),
        "contexts": prog["contexts"],
        "kernels": prog["kernels"],
        "layer_types": prog["layer_types"],
        "tail": prog["tail"],
        "globals": prog["globals"],
        "builds": Q.builds(spec),
    }
    plan = Q.pack_plan(spec)
    for lt, d in plan.pop("layer_types").items():
        m["layer_types"][lt]["pack"] = d
    m["pack"] = plan          # pool_bytes, chunk_bytes, lm_head, embed, norm
    return m


def dumps(m: dict) -> str:
    return json.dumps(m, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--spec", help="a ModelSpec JSON (default: the checked-in 27B spec)")
    g.add_argument("--model-dir", help="a model directory with config.json (+ tokenizer.json)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out", help="write here (default: stdout)")
    a = ap.parse_args(argv)
    if a.model_dir:
        spec = spec_from_model_dir(Path(a.model_dir))
    else:
        from .load import default_spec, load_spec
        spec = load_spec(Path(a.spec)) if a.spec else default_spec()
    s = dumps(manifest(spec, a.max_ctx))
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8", newline="\n")
        print(f"wrote {a.out} ({spec.family}, {spec.num_layers} layers, {spec.spec_hash()[:19]})")
    else:
        sys.stdout.write(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
