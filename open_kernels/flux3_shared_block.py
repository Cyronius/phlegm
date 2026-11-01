"""Native FLUX 3 shared block for one text-to-image sample.

The published sampler packs image, then reference images, then text. This
NPU schedule packs text first so its q/k kernel can use one text prefix
followed by the image grid and then any same-size reference grids; full
attention is permutation equivariant within a sample.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.designs.dit_ew.make_test import flux3_qk_params
from open_kernels.flux3_weight_layout import HEAD_DIM, HIDDEN, MLP, gemm_slots
from open_kernels.image_safetensors import read_header

EL = HIDDEN
HEADS = 24
GENERIC_EW_BUCKETS = (80, 160)


@dataclass
class SharedPlan:
    resolution: int
    text_length: int
    layer: int
    tokens: int
    valid_tokens: int
    modes: tuple[str, ...]
    buffers: dict[str, int]
    streams: dict[str, dict[str, dict]]
    ops: list[dict]


def plan(resolution: int = 512, text_length: int = 80,
         layer: int = 0, references: int = 0) -> SharedPlan:
    if resolution not in (512, 1024) or not 1 <= text_length <= 512 or not 0 <= layer < 28:
        raise ValueError("FLUX 3 shared block supports 512/1024, 1..512 text rows, layers 0..27")
    if references not in (0, 1):
        raise ValueError("FLUX 3 shared block currently supports zero or one reference image")
    grid = resolution // 16
    image = grid * grid
    valid = text_length + image * (1 + references)
    tokens = (valid + 511) // 512 * 512

    def view(T=tokens, off=0, ld=EL):
        return {"off": off, "ld": ld, "T": T, "E": 1}

    nbuf = tokens * EL
    slots = gemm_slots()
    slot = lambda name: slots[f"single.{layer}.{name}"]
    gemm = {
        "qkv": {"M": tokens, "K": slot("qkv").k, "N": slot("qkv").n},
        "attn_out": {"M": tokens, "K": slot("attn_out").k, "N": slot("attn_out").n},
        "mlp_in": {"M": tokens, "K": slot("mlp_in").k, "N": slot("mlp_in").n,
                   "layout": {"epi": {"first_cb": 0}}},
        "mlp_out": {"M": tokens, "K": slot("mlp_out").k, "N": slot("mlp_out").n,
                    "layout": {"a_gather": True}},
    }
    ew = {}
    layout = [("text", text_length, 0), ("image", image, text_length)]
    if references:
        layout.append(("image_cond", image, text_length + image))
    modes = tuple(mode for mode, _, _ in layout)
    # Sets other than the text-to-image 80/160 buckets use the lean dit_ew program.
    # (It was introduced while the generic program overflowed AIE program memory;
    # that was fixed on 2026-11-01 by moving GELU-tanh out of un_row, and both
    # programs build. The compiled sets keep their specs so they keep validating.)
    lean = {} if not references and text_length in GENERIC_EW_BUCKETS else {"lean": 1}
    for mode, count, start in layout:
        v = view(count, start * EL)
        ew[f"norm_{mode}"] = {
            "op": "ln_mod", "a": v, "y": v, "p_off": EL,
            "n_par": 2, "idx": {"shift": 0, "scale": 1},
            "sizes": {"X": nbuf, "B": EL, "P": 4 * EL,
                      "Y": nbuf, "Z": EL}, **lean,
        }
        ew[f"res_{mode}"] = {
            "op": "res_ln_mod", "a": v, "b": v, "y": v, "z": v,
            "p_off": EL, "n_par": 3,
            "idx": {"gate": 0, "shift": 1, "scale": 2},
            "sizes": {"X": nbuf, "B": nbuf, "P": 5 * EL,
                      "Y": nbuf, "Z": nbuf}, **lean,
        }
    ew["qk"] = {
        "op": "qk_flux3", "a": view(ld=3 * EL),
        "b": view(off=EL, ld=3 * EL),
        "y": view(ld=3 * EL), "z": view(off=EL, ld=3 * EL),
        "p_off": EL, "n_par": 3, "tok0": 0,
        "n_txt": text_length, "grid_h": grid * (1 + references), "grid_w": grid,
        "heads": HEADS, **({"ref_row": grid} if references else {}),
        "sizes": {"X": 3 * nbuf, "B": 3 * nbuf,
                  "P": 5 * EL, "Y": 3 * nbuf, "Z": 3 * nbuf},
    }
    ew["sum"] = {
        "op": "euler", "a": view(), "b": view(), "y": view(),
        "p_off": EL, "n_par": 1,
        "sizes": {"X": nbuf, "B": nbuf, "P": 3 * EL,
                  "Y": nbuf, "Z": EL}, **lean,
    }
    fa = {"attention": {"L": tokens, "heads": HEADS,
                        "kv_heads": HEADS, "causal": 0, "valid_len": valid,
                        "layout": {"qkv_ld": 3 * EL, "k_col": EL,
                                   "v_col": 2 * EL, "o_ld": EL}}}
    buffers = {"X": nbuf, "NORM": nbuf, "QKV": 3 * nbuf,
               "ATTN": nbuf, "AOUT": nbuf, "MLP": 2 * MLP * tokens,
               "MOUT": nbuf, "SUM": nbuf, "DUMMY": EL}
    ops = [
        {"stream": f"norm_{mode}",
         "args": ["X", "DUMMY", f"P_NORM_{mode}", "NORM", "DUMMY"]}
        for mode in modes
    ]
    ops += [
        {"stream": "qkv", "args": ["NORM", "w_qkv", "QKV"]},
        {"stream": "qk", "args": ["QKV", "QKV", "P_QK", "QKV", "QKV"]},
        {"stream": "attention", "args": ["QKV", "QKV", "QKV", "ATTN"]},
        {"stream": "attn_out", "args": ["ATTN", "w_attn_out", "AOUT"]},
        {"stream": "mlp_in", "args": ["NORM", "w_mlp_in", "MLP"]},
        {"stream": "mlp_out", "args": ["MLP", "w_mlp_out", "MOUT"]},
        {"stream": "sum", "args": ["AOUT", "MOUT", "P_SUM", "SUM", "DUMMY"]},
    ]
    ops += [
        {"stream": f"res_{mode}",
         "args": ["X", "SUM", f"P_RES_{mode}", "NORM", "X"]}
        for mode in modes
    ]
    return SharedPlan(resolution, text_length, layer, tokens, valid, modes,
                      buffers, {"gemm": gemm, "ew": ew, "fa": fa}, ops)


def params(model: Path, layer: int,
           modulation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    path = Path(model) / "model.safetensors"
    entries = read_header(path)
    with path.open("rb") as stream:
        base = 8 + struct.unpack("<Q", stream.read(8))[0]
    prefix = f"single_blocks.{layer}.norm."

    def scale(name: str) -> np.ndarray:
        entry = entries[prefix + name]
        if entry["dtype"] != "BF16" or entry["shape"] != [HEAD_DIM]:
            raise ValueError(f"FLUX 3 shared q/k norm {name} differs from checkpoint")
        return np.asarray(np.memmap(path, mode="r", dtype=bfloat16,
                                    offset=base + entry["data_offsets"][0],
                                    shape=(HEAD_DIM,)), dtype=np.float32)

    zero = np.zeros(EL, bfloat16)

    def block(*rows):
        return np.concatenate([zero, *(np.asarray(row, dtype=bfloat16)
                                       for row in rows), zero])

    dt = np.zeros(EL, bfloat16)
    dt[:2] = np.array([1.0], np.float32).view(np.uint16).view(bfloat16)
    result = {"P_QK": block(*flux3_qk_params(scale("query_norm.scale"),
                                               scale("key_norm.scale")).reshape(3, EL)),
              "P_SUM": block(dt)}
    for mode in modulation:
        mod = np.asarray(modulation[mode], np.float32)
        if mod.shape != (3, EL) or not np.isfinite(mod).all():
            raise ValueError(f"FLUX 3 {mode} modulation must be finite [3,3072]")
        result[f"P_NORM_{mode}"] = block(mod[0], mod[1])
        result[f"P_RES_{mode}"] = block(mod[2], mod[0], mod[1])
    return result
