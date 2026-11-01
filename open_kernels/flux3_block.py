"""One checkpoint-weighted FLUX 3 per-mode transformer block on the NPU.

This is the published per-mode block (before the shared blocks) for the
generated image, the text, or one reference image (``image_cond``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.flux3_weight_layout import HIDDEN, HEAD_DIM, MLP, gemm_slots
from open_kernels.image_safetensors import read_header
from open_kernels.designs.dit_ew.make_test import flux3_qk_params

EL = HIDDEN
HEADS = 24


@dataclass
class BlockPlan:
    resolution: int
    layer: int
    mode: str
    tokens: int
    valid_tokens: int
    buffers: dict[str, int]
    streams: dict[str, dict[str, dict]]
    ops: list[dict]


def plan(resolution: int = 512, layer: int = 0, *, mode: str = "image",
         text_length: int = 80) -> BlockPlan:
    if resolution not in (512, 1024) or not 0 <= layer < 5 or mode not in ("image", "image_cond", "text"):
        raise ValueError("FLUX 3 per-mode block supports 512/1024 pixels and layers 0..4")
    if mode == "text" and not 1 <= text_length <= 512:
        raise ValueError("FLUX 3 text block currently supports 1..512 tokens")
    grid = resolution // 16
    valid = text_length if mode == "text" else grid * grid
    tokens = (valid + 511) // 512 * 512

    def view(ld=EL, off=0, e=1):
        return {"off": off, "ld": ld, "T": tokens, "E": e}

    slots = gemm_slots()
    slot = lambda name: slots[f"{mode}.{layer}.{name}"]
    gemm = {
        "qkv": {"M": tokens, "K": slot("qkv").k, "N": slot("qkv").n},
        "attn_out": {"M": tokens, "K": slot("attn_out").k, "N": slot("attn_out").n},
        "mlp_in": {"M": tokens, "K": slot("mlp_in").k, "N": slot("mlp_in").n,
                   "layout": {"epi": {"first_cb": 0}}},
        "mlp_out": {"M": tokens, "K": slot("mlp_out").k, "N": slot("mlp_out").n,
                    "layout": {"a_gather": True}},
    }
    ew = {
        "norm": {"op": "ln_mod", "a": view(), "y": view(), "p_off": EL,
                 "n_par": 2, "idx": {"shift": 0, "scale": 1},
                 "sizes": {"X": tokens * EL, "B": EL, "P": 4 * EL,
                           "Y": tokens * EL, "Z": EL}},
        "qk": {"op": "qk_flux3", "a": view(3 * EL),
               "b": view(3 * EL, EL), "y": view(3 * EL),
               "z": view(3 * EL, EL),
               "p_off": EL, "n_par": 3, "tok0": 0,
               "n_txt": text_length if mode == "text" else 0,
               "grid_h": 0 if mode == "text" else grid,
               "grid_w": 64 if mode == "text" else grid, "heads": HEADS,
               # One reference image: every row has the published l=-1 id.
               **({"ref_row": 0} if mode == "image_cond" else {}),
               "sizes": {"X": tokens * 3 * EL, "B": tokens * 3 * EL,
                         "P": 5 * EL, "Y": tokens * 3 * EL,
                         "Z": tokens * 3 * EL}},
        "sum": {"op": "euler", "a": view(), "b": view(), "y": view(),
                "p_off": EL, "n_par": 1,
                "sizes": {"X": tokens * EL, "B": tokens * EL,
                          "P": 3 * EL, "Y": tokens * EL, "Z": EL}},
        "res": {"op": "res_ln_mod", "a": view(), "b": view(),
                "y": view(), "z": view(), "p_off": EL, "n_par": 3,
                "idx": {"gate": 0, "shift": 1, "scale": 2},
                "sizes": {"X": tokens * EL, "B": tokens * EL,
                          "P": 5 * EL, "Y": tokens * EL, "Z": tokens * EL}},
    }
    fa = {"attention": {"L": tokens, "heads": HEADS, "kv_heads": HEADS,
                        "causal": 0, "valid_len": valid,
                        "layout": {"qkv_ld": 3 * EL, "k_col": EL,
                                   "v_col": 2 * EL, "o_ld": EL}}}
    buffers = {"X": tokens * EL, "NORM": tokens * EL,
               "QKV": tokens * 3 * EL, "ATTN": tokens * EL,
               "AOUT": tokens * EL, "MLP": tokens * 2 * MLP,
               "MOUT": tokens * EL, "SUM": tokens * EL,
               "DUMMY": EL}
    ops = [
        {"stream": "norm", "args": ["X", "DUMMY", "P_NORM", "NORM", "DUMMY"]},
        {"stream": "qkv", "args": ["NORM", "w_qkv", "QKV"]},
        {"stream": "qk", "args": ["QKV", "QKV", "P_QK", "QKV", "QKV"]},
        {"stream": "attention", "args": ["QKV", "QKV", "QKV", "ATTN"]},
        {"stream": "attn_out", "args": ["ATTN", "w_attn_out", "AOUT"]},
        {"stream": "mlp_in", "args": ["NORM", "w_mlp_in", "MLP"]},
        {"stream": "mlp_out", "args": ["MLP", "w_mlp_out", "MOUT"]},
        {"stream": "sum", "args": ["AOUT", "MOUT", "P_SUM", "SUM", "DUMMY"]},
        {"stream": "res", "args": ["X", "SUM", "P_RES", "NORM", "X"]},
    ]
    return BlockPlan(resolution, layer, mode, tokens, valid, buffers,
                     {"gemm": gemm, "ew": ew, "fa": fa}, ops)


def params(model: Path, layer: int, modulation: np.ndarray, *,
           mode: str = "image") -> dict[str, np.ndarray]:
    """[shift,scale,gate] modulation and pinned q/k scales as BF16 buffers."""
    path = Path(model) / "model.safetensors"
    if mode not in ("image", "image_cond", "text"):
        raise ValueError("FLUX 3 block mode must be image, image_cond or text")
    prefix = (f"content_mode_blocks.{mode}.{layer}.norm." if mode != "text"
              else f"txt_mode_blocks.{layer}.norm.")
    entries = read_header(path)
    with path.open("rb") as stream:
        base = 8 + struct.unpack("<Q", stream.read(8))[0]

    def scale(name: str) -> np.ndarray:
        entry = entries[prefix + name]
        if entry["dtype"] != "BF16" or entry["shape"] != [HEAD_DIM]:
            raise ValueError(f"FLUX 3 q/k scale {name} differs from checkpoint")
        offset = base + entry["data_offsets"][0]
        return np.asarray(np.memmap(path, mode="r", dtype=bfloat16,
                                    offset=offset, shape=(HEAD_DIM,)), dtype=np.float32)

    mod = np.asarray(modulation, dtype=np.float32)
    if mod.shape != (3, EL) or not np.isfinite(mod).all():
        raise ValueError("FLUX 3 block modulation must be finite [3,3072]")
    zero = np.zeros(EL, bfloat16)

    def block(*rows):
        return np.concatenate([zero, *(np.asarray(row, dtype=bfloat16)
                                       for row in rows), zero])

    dt = np.zeros(EL, bfloat16)
    dt[:2] = np.array([1.0], np.float32).view(np.uint16).view(bfloat16)
    return {"P_NORM": block(mod[0], mod[1]),
            "P_QK": block(*flux3_qk_params(scale("query_norm.scale"),
                                            scale("key_norm.scale")).reshape(3, EL)),
            "P_SUM": block(dt), "P_RES": block(mod[2], mod[0], mod[1])}
