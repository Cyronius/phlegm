"""One native MOSS Image joint-transformer block and its replay parameters.

This is the smallest useful DiT integration gate. The input is already embedded
and the six modulation vectors are supplied by the host; all four projections,
q/k normalization and RoPE, joint attention, SiTU, and gated residuals execute
through Phlegm's NPU kernels. A full denoising pipeline builds on this schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.image_safetensors import read_header
from open_kernels.moss_weight_layout import HIDDEN, HEADS, HEAD_DIM, gemm_slots


EL = HIDDEN
FF_WIDTH = 3 * HIDDEN
FF_FUSED = 6 * HIDDEN


def moss_qk_params(wq: np.ndarray, wk: np.ndarray,
                   grid_h: int, grid_w: int) -> np.ndarray:
    """Three 3072-element NPU parameter rows: q/k weights, then RoPE table."""
    freq = 10000.0 ** (-np.arange(16, dtype=np.float64) / 16)
    table = np.zeros(2 * EL, bfloat16)
    cursor = 0
    for positions in (64 * np.arange(65), np.arange(64),
                      np.array([-(grid_h - 1) / 2]),
                      np.array([-(grid_w - 1) / 2])):
        angle = np.asarray(positions)[:, None] * freq
        values = np.concatenate((np.cos(angle), np.sin(angle)), axis=1)
        count = values.size
        table[cursor:cursor + count] = values.astype(bfloat16).reshape(-1)
        cursor += count
    first = np.zeros(EL, bfloat16)
    first[:HEAD_DIM] = np.asarray(wq, dtype=bfloat16)
    first[HEAD_DIM:2 * HEAD_DIM] = np.asarray(wk, dtype=bfloat16)
    return np.concatenate((first, table))


@dataclass
class BlockPlan:
    resolution: int
    text_length: int
    tokens: int
    valid_tokens: int
    buffers: dict[str, int]
    streams: dict[str, dict[str, dict]]
    ops: list[dict]


def plan(resolution: int, text_length: int, layer: int = 0) -> BlockPlan:
    if resolution not in (256, 512, 1024):
        raise ValueError("MOSS block replay supports square 256, 512, or 1024 pixel grids")
    if not 1 <= text_length <= 4096:
        raise ValueError("MOSS text length must be 1..4096")
    if not 0 <= layer < 42:
        raise ValueError("MOSS layer must be 0..41")
    grid = resolution // 16
    image_tokens = grid * grid
    valid = image_tokens + text_length
    tokens = (valid + 511) // 512 * 512
    if tokens > 8192:
        raise ValueError("MOSS joint sequence exceeds 8192 NPU rows")

    def view(ld=EL, off=0, e=1):
        return {"off": off, "ld": ld, "T": tokens, "E": e}

    def norm_spec(residual=False):
        spec = {"op": "res_ln_mod" if residual else "ln_mod", "norm": "rms",
                "affine_rms": 1, "a": view(), "y": view(), "p_off": EL,
                "n_par": 3 if residual else 2,
                "idx": {"shift": 1, "scale": 2, "gate": 0} if residual else
                       {"shift": 0, "scale": 1},
                "sizes": {"X": tokens * EL, "B": tokens * EL if residual else EL,
                          "P": (5 if residual else 4) * EL,
                          "Y": tokens * EL, "Z": tokens * EL if residual else EL}}
        if residual:
            spec["b"], spec["z"] = view(), view()
        return spec

    qk = {"op": "qk_moss", "a": view(3 * EL), "b": view(3 * EL, EL),
          "y": view(3 * EL), "z": view(3 * EL, EL),
          "p_off": EL, "n_par": 3, "tok0": 0, "n_txt": text_length,
          "grid_h": grid, "grid_w": grid, "heads": HEADS,
          "sizes": {"X": tokens * 3 * EL, "B": tokens * 3 * EL,
                    "P": 5 * EL, "Y": tokens * 3 * EL, "Z": tokens * 3 * EL}}
    situ = {"op": "situ", "a": view(FF_FUSED, 0, 3),
            "b": view(FF_FUSED, FF_WIDTH, 3), "y": view(FF_WIDTH, 0, 3),
            "sizes": {"X": tokens * FF_FUSED, "B": tokens * FF_FUSED,
                      "P": EL, "Y": tokens * FF_WIDTH, "Z": EL}}
    ew = {"norm1": norm_spec(), "qk": qk, "res_norm2": norm_spec(True),
          "situ": situ, "res_final": norm_spec(True)}
    slots = gemm_slots()
    gemm = {}
    for name in ("qkv", "attn_out", "ffn_in", "ffn_out"):
        slot = slots[f"block.{layer}.{name}"]
        gemm[name] = {"M": tokens, "K": slot.k, "N": slot.n}
    fa = {"joint": {"L": tokens, "heads": HEADS, "kv_heads": HEADS,
                    "causal": 0, "valid_len": valid,
                    "layout": {"qkv_ld": 3 * EL, "k_col": EL,
                               "v_col": 2 * EL, "o_ld": EL}}}
    buffers = {"X": tokens * EL, "Z": tokens * EL, "QKV": tokens * 3 * EL,
               "ATTN": tokens * EL, "PROJ": tokens * EL,
               "FF": tokens * FF_FUSED, "UP": tokens * FF_WIDTH,
               "P_NORM1": 4 * EL, "P_QK": 5 * EL,
               "P_RES1": 5 * EL, "P_RES2": 5 * EL, "DUMMY": EL}
    w = lambda name: f"block.{layer}.{name}"
    ops = [
        {"set": "ew", "stream": "norm1", "args": ["X", "DUMMY", "P_NORM1", "Z", "DUMMY"]},
        {"set": "gemm", "stream": "qkv", "args": ["Z", w("qkv"), "QKV"]},
        {"set": "ew", "stream": "qk", "args": ["QKV", "QKV", "P_QK", "QKV", "QKV"]},
        {"set": "fa", "stream": "joint", "args": ["QKV", "QKV", "QKV", "ATTN"]},
        {"set": "gemm", "stream": "attn_out", "args": ["ATTN", w("attn_out"), "PROJ"]},
        {"set": "ew", "stream": "res_norm2", "args": ["X", "PROJ", "P_RES1", "Z", "X"]},
        {"set": "gemm", "stream": "ffn_in", "args": ["Z", w("ffn_in"), "FF"]},
        {"set": "ew", "stream": "situ", "args": ["FF", "FF", "DUMMY", "UP", "DUMMY"]},
        {"set": "gemm", "stream": "ffn_out", "args": ["UP", w("ffn_out"), "PROJ"]},
        {"set": "ew", "stream": "res_final", "args": ["X", "PROJ", "P_RES2", "Z", "X"]},
    ]
    return BlockPlan(resolution, text_length, tokens, valid, buffers,
                     {"gemm": gemm, "fa": fa, "ew": ew}, ops)


def vector(reader: dict[str, tuple[Path, int, tuple[int, ...]]],
           name: str, size: int) -> np.ndarray:
    path, offset, shape = reader[name]
    if shape != (size,):
        raise ValueError(f"{name}: expected [{size}], got {shape}")
    return np.asarray(np.memmap(path, dtype=bfloat16, mode="r", offset=offset,
                                shape=shape), dtype=np.float32)


def tensor_entries(model_dir: Path) -> dict[str, tuple[Path, int, tuple[int, ...]]]:
    """Validate and locate BF16 transformer tensors without loading shards."""
    index = json.loads((Path(model_dir) / "transformer" /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())
    reader = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = Path(model_dir) / "transformer" / shard
        with path.open("rb") as stream:
            base = 8 + struct.unpack("<Q", stream.read(8))[0]
        for name, entry in read_header(path).items():
            if index["weight_map"].get(name) != shard:
                raise ValueError(f"unexpected MOSS transformer tensor {name}")
            if entry["dtype"] != "BF16":
                if name == "rope.bands" and entry["dtype"] == "F32":
                    continue
                raise ValueError(f"unexpected MOSS transformer dtype {name}")
            reader[name] = (path, base + entry["data_offsets"][0],
                            tuple(entry["shape"]))
    return reader


def params(model_dir: Path, layer: int, resolution: int,
           modulation: np.ndarray) -> dict[str, np.ndarray]:
    """Pack actual block norms and q/k weights for the five EW streams.

    ``modulation`` is float32/bf16 [6, 3072] in the checkpoint's
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f order.
    """
    mod = np.asarray(modulation, dtype=np.float32)
    if mod.shape != (6, EL) or not np.isfinite(mod).all():
        raise ValueError("expected finite [6, 3072] MOSS modulation")
    reader = tensor_entries(model_dir)
    prefix = f"layers.{layer}."
    n1 = vector(reader, prefix + "norm1.weight", EL)
    n2 = vector(reader, prefix + "norm2.weight", EL)
    q = vector(reader, prefix + "q_norm.weight", HEAD_DIM)
    k = vector(reader, prefix + "k_norm.weight", HEAD_DIM)
    zero = np.zeros(EL, bfloat16)

    def block(*rows):
        return np.concatenate((zero, *(np.asarray(row, dtype=bfloat16)
                                       for row in rows), zero))

    return {
        "P_NORM1": block(mod[0], n1 * (1 + mod[1])),
        "P_QK": block(*moss_qk_params(q, k, resolution // 16,
                                      resolution // 16).reshape(3, EL)),
        "P_RES1": block(mod[2], mod[3], n2 * (1 + mod[4])),
        "P_RES2": block(mod[5], zero, np.ones(EL, np.float32)),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--text-length", type=int, default=17)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    p = plan(args.resolution, args.text_length, args.layer)
    print(json.dumps({"tokens": p.tokens, "valid_tokens": p.valid_tokens,
                      "buffers": p.buffers, "streams": p.streams, "ops": p.ops}, indent=2))
