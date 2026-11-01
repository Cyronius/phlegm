"""Native NPU schedule for MOSS Image's two text-refinement blocks.

The input is the headless Qwen3.5 encoder's unpadded [tokens,4096] hidden
state. This schedule projects it to 3072, runs both RMSNorm/attention/SiTU
blocks, and leaves padded rows out of every attention key range.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.moss_dit_block import (EL, FF_FUSED, FF_WIDTH,
                                         moss_qk_params, tensor_entries, vector)
from open_kernels.moss_weight_layout import HEADS, gemm_slots


@dataclass
class TextPlan:
    text_length: int
    tokens: int
    buffers: dict[str, int]
    streams: dict[str, dict[str, dict]]
    ops: list[dict]


def plan(text_length: int) -> TextPlan:
    if not 1 <= text_length <= 4096:
        raise ValueError("MOSS text refinement needs 1..4096 real tokens")
    tokens = (text_length + 511) // 512 * 512

    def view(ld=EL, off=0, e=1):
        return {"off": off, "ld": ld, "T": tokens, "E": e}

    def bias_spec(e: int, ld: int, off=0):
        return {"op": "bias", "a": view(ld, off, e), "y": view(ld, off, e),
                "p_off": EL, "n_par": e, "bias_period": e,
                "sizes": {"X": tokens * ld, "B": EL, "P": (e + 2) * EL,
                          "Y": tokens * ld, "Z": EL}}

    qk = {"op": "qk_moss", "a": view(3 * EL), "b": view(3 * EL, EL),
          "y": view(3 * EL), "z": view(3 * EL, EL),
          "p_off": EL, "n_par": 3, "tok0": 1, "n_txt": text_length,
          "grid_h": 1, "grid_w": 1, "heads": HEADS,
          "sizes": {"X": tokens * 3 * EL, "B": tokens * 3 * EL,
                    "P": 5 * EL, "Y": tokens * 3 * EL, "Z": tokens * 3 * EL}}
    situ = {"op": "situ", "a": view(FF_FUSED, 0, 3),
            "b": view(FF_FUSED, FF_WIDTH, 3), "y": view(FF_WIDTH, 0, 3),
            "sizes": {"X": tokens * FF_FUSED, "B": tokens * FF_FUSED,
                      "P": EL, "Y": tokens * FF_WIDTH, "Z": EL}}
    norm = {"op": "ln_mod", "norm": "rms", "a": view(), "y": view(),
            "p_off": EL, "n_par": 1, "idx": {"scale": 0},
            "sizes": {"X": tokens * EL, "B": EL, "P": 3 * EL,
                      "Y": tokens * EL, "Z": EL}}
    residual = {"op": "res_ln_mod", "norm": "rms", "unit_gate": 1,
                "a": view(), "b": view(), "y": view(), "z": view(),
                "p_off": EL, "n_par": 1, "idx": {"scale": 0},
                "sizes": {"X": tokens * EL, "B": tokens * EL,
                          "P": 3 * EL, "Y": tokens * EL, "Z": tokens * EL}}
    ew = {"bias1": bias_spec(1, EL), "bias3": bias_spec(3, 3 * EL),
          "bias_ff_up": bias_spec(3, FF_FUSED),
          "bias_ff_gate": bias_spec(3, FF_FUSED, FF_WIDTH),
          "norm": norm, "residual": residual, "qk": qk, "situ": situ}
    slots = gemm_slots()
    gemm = {}
    for name in ("emb.text", *(f"text.{layer}.{kind}" for layer in range(2)
                   for kind in ("qkv", "attn_out", "ffn_in", "ffn_out"))):
        slot = slots[name]
        gemm[name] = {"M": tokens, "K": slot.k, "N": slot.n}
    fa = {"text": {"L": tokens, "heads": HEADS, "kv_heads": HEADS,
                   "causal": 0, "valid_len": text_length,
                   "layout": {"qkv_ld": 3 * EL, "k_col": EL,
                              "v_col": 2 * EL, "o_ld": EL}}}
    buffers = {"CTX": tokens * 4096, "X": tokens * EL, "Z": tokens * EL,
               "QKV": tokens * 3 * EL, "ATTN": tokens * EL,
               "PROJ": tokens * EL, "FF": tokens * FF_FUSED,
               "UP": tokens * FF_WIDTH, "DUMMY": EL}
    ops = [
        {"set": "gemm", "stream": "emb.text", "args": ["CTX", "emb.text", "X"]},
        {"set": "ew", "stream": "bias1", "args": ["X", "DUMMY", "B_EMB", "X", "DUMMY"]},
    ]
    for layer in range(2):
        prefix = f"text.{layer}"
        ops += [
            {"set": "ew", "stream": "norm", "args": ["X", "DUMMY", f"P_{layer}_NORM1", "Z", "DUMMY"]},
            {"set": "gemm", "stream": f"{prefix}.qkv", "args": ["Z", f"{prefix}.qkv", "QKV"]},
            {"set": "ew", "stream": "bias3", "args": ["QKV", "DUMMY", f"B_{layer}_QKV", "QKV", "DUMMY"]},
            {"set": "ew", "stream": "qk", "args": ["QKV", "QKV", f"P_{layer}_QK", "QKV", "QKV"]},
            {"set": "fa", "stream": "text", "args": ["QKV", "QKV", "QKV", "ATTN"]},
            {"set": "gemm", "stream": f"{prefix}.attn_out",
             "args": ["ATTN", f"{prefix}.attn_out", "PROJ"]},
            {"set": "ew", "stream": "bias1", "args": ["PROJ", "DUMMY", f"B_{layer}_ATTN", "PROJ", "DUMMY"]},
            {"set": "ew", "stream": "residual",
             "args": ["X", "PROJ", f"P_{layer}_NORM2", "Z", "X"]},
            {"set": "gemm", "stream": f"{prefix}.ffn_in", "args": ["Z", f"{prefix}.ffn_in", "FF"]},
            {"set": "ew", "stream": "bias_ff_up",
             "args": ["FF", "DUMMY", f"B_{layer}_FF_UP", "FF", "DUMMY"]},
            {"set": "ew", "stream": "bias_ff_gate",
             "args": ["FF", "DUMMY", f"B_{layer}_FF_GATE", "FF", "DUMMY"]},
            {"set": "ew", "stream": "situ", "args": ["FF", "FF", "DUMMY", "UP", "DUMMY"]},
            {"set": "gemm", "stream": f"{prefix}.ffn_out", "args": ["UP", f"{prefix}.ffn_out", "PROJ"]},
            {"set": "ew", "stream": "bias1", "args": ["PROJ", "DUMMY", f"B_{layer}_FF_OUT", "PROJ", "DUMMY"]},
            {"set": "ew", "stream": "residual",
             "args": ["X", "PROJ", "P_IDENTITY", "Z", "X"]},
        ]
    return TextPlan(text_length, tokens, buffers,
                    {"gemm": gemm, "ew": ew, "fa": fa}, ops)


def params(model_dir: Path) -> dict[str, np.ndarray]:
    """Pack the reference's text biases and norm/qk vectors for native EW."""
    entries = tensor_entries(Path(model_dir))
    zero = np.zeros(EL, bfloat16)

    def block(*rows):
        return np.concatenate((zero, *(np.asarray(row, dtype=bfloat16)
                                       for row in rows), zero))

    def v(name, size):
        return vector(entries, name, size)

    out = {"B_EMB": block(v("text_embedder.proj.bias", EL)),
           "P_IDENTITY": block(np.ones(EL, np.float32))}
    for layer in range(2):
        prefix = f"text_embedder.blocks.{layer}."
        out[f"P_{layer}_NORM1"] = block(v(prefix + "norm1.weight", EL))
        out[f"P_{layer}_NORM2"] = block(v(prefix + "norm2.weight", EL))
        out[f"P_{layer}_QK"] = block(*moss_qk_params(
            v(prefix + "q_norm.weight", 128),
            v(prefix + "k_norm.weight", 128), 1, 1).reshape(3, EL))
        out[f"B_{layer}_QKV"] = block(*v(prefix + "qkv.bias", 3 * EL).reshape(3, EL))
        out[f"B_{layer}_ATTN"] = block(v(prefix + "proj.bias", EL))
        for kind, source in (("FF_UP", "ffn_up.bias"),
                             ("FF_GATE", "ffn_gate.bias")):
            bias = np.zeros(FF_WIDTH, bfloat16)
            bias[:8192] = v(prefix + source, 8192).astype(bfloat16)
            out[f"B_{layer}_{kind}"] = block(*bias.reshape(3, EL))
        out[f"B_{layer}_FF_OUT"] = block(v(prefix + "ffn_down.bias", EL))
    return out


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-length", type=int, default=17)
    args = parser.parse_args()
    p = plan(args.text_length)
    print(json.dumps({"tokens": p.tokens, "buffers": p.buffers,
                      "streams": p.streams, "ops": p.ops}, indent=2))
