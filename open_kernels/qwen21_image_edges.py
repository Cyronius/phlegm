"""Fixed 512-pixel native entry, modulation and output shapes for Qwen Image 2.1."""

from __future__ import annotations

from open_kernels.qwen21_weight_layout import HIDDEN


def specs() -> dict[str, dict[str, dict]]:
    rows = 512
    view = {"off": 0, "ld": HIDDEN, "T": rows, "E": 1}
    unary = {"a": view, "y": view,
             "sizes": {"X": rows * HIDDEN, "B": HIDDEN, "P": HIDDEN,
                       "Y": rows * HIDDEN, "Z": HIDDEN}}
    return {
        "gemm": {
            "short_to_hidden": {"M": rows, "K": 512, "N": HIDDEN},
            "hidden_to_hidden": {"M": rows, "K": HIDDEN, "N": HIDDEN},
            "hidden_to_mod": {"M": rows, "K": HIDDEN, "N": 4 * HIDDEN},
            "hidden_to_latent": {"M": rows, "K": HIDDEN, "N": 1024},
        },
        "ew": {
            "text_rms": {"op": "ln_mod", "a": view, "y": view,
                         "p_off": HIDDEN, "n_par": 1, "idx": {"scale": 0},
                         "norm": "rms", "sizes": {"X": rows * HIDDEN,
                            "B": HIDDEN, "P": 3 * HIDDEN,
                            "Y": rows * HIDDEN, "Z": HIDDEN}},
            "gelu_tanh": {"op": "gelu_tanh", **unary},
            "silu": {"op": "silu", **unary},
            "final_ln": {"op": "ln_mod", "a": view, "y": view,
                         "p_off": HIDDEN, "n_par": 2,
                         "idx": {"shift": 0, "scale": 1},
                         "sizes": {"X": rows * HIDDEN, "B": HIDDEN,
                                   "P": 4 * HIDDEN, "Y": rows * HIDDEN,
                                   "Z": HIDDEN}},
        },
    }
