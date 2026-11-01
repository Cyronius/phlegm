r"""Test vectors for one dit_ew op, for open_kernels/harness's run_kernel.

    python make_test.py --op <op> --out <testdir> [--T 4608] [--seed 0]
    DE_SPEC=<testdir>/spec.json contents -> build_design.py designs/dit_ew/dit_ew.py <build>
    python make_test.py --op <op> --out <testdir> --build <build>    (writes run.cfg)
    open_kernels\harness\out\run_kernel.exe <testdir>\run.cfg
    python compare.py <testdir>

ops: ln_mod, res_ln_mod, qk, qk_moss, swiglu, situ, euler, silu. Most shapes
follow FLUX.2 [klein] 4B:
hidden 3072, 24 heads of 128, the SwiGLU half 9216 read out of an [T, 18432] ff_in
output, q/k read out of an [T, 9216] fused QKV buffer, latents [T, 128] with the
velocity in an [T, 1024] (N-padded proj_out) buffer.

The reference (ref.npz) is float64 math on the bf16 inputs, the formulas of diffusers'
transformer_flux2.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

EL = int(os.environ.get("DE_EL", "3072"))
HD = 128
THETA = 2000.0


def bf(x):
    return np.asarray(x, dtype=np.float32).astype(bfloat16)


def rope_freqs():
    return 1.0 / (THETA ** (np.arange(0, 32, 2, dtype=np.float64) / 32))


def rope_tables() -> np.ndarray:
    """fp32 [3072]: FINE[p] = cos[16], sin[16] of p*f (p < 64), COARSE[a] of 64a*f (a < 8)."""
    f = rope_freqs()
    t = np.zeros(EL, dtype=np.float32)
    for p in range(64):
        a = p * f
        t[p * 32:p * 32 + 16], t[p * 32 + 16:p * 32 + 32] = np.cos(a), np.sin(a)
    for a in range(8):
        ang = 64 * a * f
        o = 64 * 32 + a * 32
        t[o:o + 16], t[o + 16:o + 32] = np.cos(ang), np.sin(ang)
    return t


QWEN_THETA = 1e6


def qwen_rope_tables() -> np.ndarray:
    """fp32 [3072]: T64[a], T8[b], T1[c] = cos[64], sin[64] of (64a | 8b | c) * f_j."""
    f = 1.0 / (QWEN_THETA ** (np.arange(0, 128, 2, dtype=np.float64) / 128))
    t = np.zeros(EL, dtype=np.float32)
    for k, step in enumerate((64, 8, 1)):
        for i in range(8):
            a = i * step * f
            o = k * 8 * 128 + i * 128
            t[o:o + 64], t[o + 64:o + 128] = np.cos(a), np.sin(a)
    return t


def qwen_qk_params(wq, wk) -> np.ndarray:
    e0 = np.zeros(EL, dtype=bfloat16)
    e0[:HD], e0[HD:2 * HD] = bf(wq), bf(wk)
    return np.concatenate([e0, qwen_rope_tables().view(np.uint16).view(bfloat16)])


def moss_qk_params(wq, wk, grid_h: int, grid_w: int) -> np.ndarray:
    """Q/k norm weights and MOSS's compact bf16 four-axis RoPE table."""
    freq = 10000.0 ** (-np.arange(16, dtype=np.float64) / 16)
    tab = np.zeros(2 * EL, bfloat16)
    cursor = 0
    for positions in (64 * np.arange(65), np.arange(64),
                      np.array([-(grid_h - 1) / 2]), np.array([-(grid_w - 1) / 2])):
        angle = np.asarray(positions)[:, None] * freq
        values = np.concatenate([np.cos(angle), np.sin(angle)], 1)
        n = values.size
        tab[cursor:cursor + n] = bf(values).reshape(-1)
        cursor += n
    e0 = np.zeros(EL, bfloat16)
    e0[:HD], e0[HD:2 * HD] = bf(wq), bf(wk)
    return np.concatenate([e0, tab])


def qk_params(wq, wk) -> np.ndarray:
    """dit_ew qk parameter block: 3 elements (bf16 slots)."""
    e0 = np.zeros(EL, dtype=bfloat16)
    e0[:HD], e0[HD:2 * HD] = bf(wq), bf(wk)
    tab = rope_tables().view(np.uint16).view(bfloat16)
    return np.concatenate([e0, tab])


def flux3_qk_params(wq, wk) -> np.ndarray:
    """FLUX 3 q/k scale and θ=10000 tables, including its half image step."""
    freq = 10000.0 ** (-np.arange(0, 32, 2, dtype=np.float64) / 32)
    tab = np.zeros(EL, np.float32)
    for p in range(64):
        angle = p * freq
        tab[p * 32:p * 32 + 16] = np.cos(angle)
        tab[p * 32 + 16:p * 32 + 32] = np.sin(angle)
    for p in range(8):
        angle = 64 * p * freq
        at = 64 * 32 + p * 32
        tab[at:at + 16] = np.cos(angle)
        tab[at + 16:at + 32] = np.sin(angle)
    angle = 0.5 * freq
    at = 64 * 32 + 8 * 32
    tab[at:at + 16] = np.cos(angle)
    tab[at + 16:at + 32] = np.sin(angle)
    e0 = np.zeros(EL, bfloat16)
    e0[:HD], e0[HD:2 * HD] = bf(wq), bf(wk)
    return np.concatenate([e0, tab.view(np.uint16).view(bfloat16)])


def qwen21_qk_params(wq, wk) -> np.ndarray:
    """Qwen Image 2.1's 3-axis theta=10000 table, factored at position 32."""
    freqs = np.concatenate([10000.0 ** (-np.arange(0, dim, 2) / dim)
                            for dim in (16, 56, 56)])
    tables = []
    for step in (1, 32):
        angle = np.arange(32)[:, None] * step * freqs[None]
        tables.append(bf(np.stack((np.cos(angle), np.sin(angle)), -1)).reshape(-1))
    e0 = np.zeros(EL, bfloat16)
    e0[:HD], e0[HD:2 * HD] = bf(wq), bf(wk)
    return np.concatenate([e0, *tables])


REF_T = 10                        # diffusers' time coordinate of an edit's first reference


def rope_ref(y, g, n_txt, grid_w):
    """Interleaved 4-axis RoPE of y [..., 128] (float64) for global token indices g:
    text (0, 0, 0, l), generated (0, h, w, 0), then reference (REF_T, h, w, 0)."""
    f = rope_freqs()
    pos = np.zeros((len(g), 4))
    txt = g < n_txt
    pos[txt, 3] = g[txt]
    k = g[~txt] - n_txt
    ref = k >= grid_w * grid_w
    k = np.where(ref, k - grid_w * grid_w, k)
    pos[~txt, 0] = np.where(ref, REF_T, 0)
    pos[~txt, 1], pos[~txt, 2] = k // grid_w, k % grid_w
    ang = np.concatenate([np.outer(pos[:, a], f) for a in range(4)], axis=1)   # [T, 64]
    cos = np.repeat(np.cos(ang), 2, axis=1)[:, None, :]
    sin = np.repeat(np.sin(ang), 2, axis=1)[:, None, :]
    rot = np.stack([-y[..., 1::2], y[..., 0::2]], axis=-1).reshape(y.shape)
    return y * cos + rot * sin


def ln(x, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    return (x - m) / np.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + eps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True,
                    choices=["ln_mod", "res_ln_mod", "qk", "qk_flux3", "qk_moss", "qk_qwen21", "swiglu", "situ", "euler", "silu", "gelu_tanh", "bias",
                             "rms", "res_rms", "qk_qwen"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--W", type=int, default=None, help="RMSNorm logical channel width")
    ap.add_argument("--E", type=int, default=3, help="bias: 1..3 elements per row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-h", type=int, default=None, help="qk_moss image-token grid height")
    ap.add_argument("--grid-w", type=int, default=None, help="qk_moss image-token grid width")
    ap.add_argument("--text-len", type=int, default=None,
                    help="qk_moss true text length; remaining rows are padded")
    ap.add_argument("--ref-row", type=int, default=None,
                    help="qk_flux3: first image row of a reference image (l=-1)")
    ap.add_argument("--build", default=None, help="build dir: write run.cfg against it")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--edit", action="store_true",
                    help="qk: an edit's [text | generated | reference] sequence at 512² "
                         "(grid 32, T = 512 + 2 x 1024)")
    ap.add_argument("--tanh-gate", action="store_true",
                    help="res_ln_mod: apply Qwen Image 2.1's tanh to the residual gate")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    op = a.op
    T = a.T or {"silu": 16, "gelu_tanh": 512, "euler": 4096}.get(op, 2560 if a.edit else 4608)
    row = lambda T_, off=0, ld=EL, E=1: {"off": off, "ld": ld, "T": T_, "E": E}  # noqa: E731
    pvec = lambda n: (rng.standard_normal((n, EL)) * 0.5).astype(np.float32)  # noqa: E731
    bufs, ref, spec = {}, {}, {"op": op}

    def with_pad(p):
        """P = [pad, p..., pad]: dit_ew reads one vector either side of the run."""
        values = np.asarray(p)
        # Q/k tables store raw FP32 words inside BF16 slots. A BF16 -> FP32
        # -> BF16 conversion here would canonicalize some NaN bit patterns.
        values = values.reshape(-1) if values.dtype == bfloat16 else bf(values).reshape(-1)
        return np.concatenate([np.zeros(EL, bfloat16), values, np.zeros(EL, bfloat16)])

    if op == "bias":
        if a.E not in (1, 2, 3):
            raise ValueError("bias fixture needs E=1, 2, or 3")
        x = bf(rng.standard_normal((T, a.E * EL)) * 2)
        p = pvec(a.E)
        bufs = {"X": x, "P": with_pad(p)}
        spec |= {"a": row(T, 0, a.E * EL, a.E),
                 "y": row(T, 0, a.E * EL, a.E),
                 "p_off": EL, "n_par": a.E, "bias_period": a.E}
        ref["Y"] = (x.astype(np.float64).reshape(T, a.E, EL) +
                    bf(p).astype(np.float64)[None]).reshape(T, a.E * EL)
    elif op == "ln_mod":
        x = bf(rng.standard_normal((T, EL)) * 2 + rng.standard_normal((T, 1)))
        p = pvec(2)                                     # shift, scale
        bufs = {"X": x, "P": with_pad(p)}
        spec |= {"a": row(T), "y": row(T), "p_off": EL, "n_par": 2,
                 "idx": {"shift": 0, "scale": 1}}
        ps = bf(p).astype(np.float64)
        ref["Y"] = ln(x.astype(np.float64)) * (1 + ps[1]) + ps[0]
    elif op == "res_ln_mod":
        x = bf(rng.standard_normal((T, EL)) * 4)
        y = bf(rng.standard_normal((T, EL)))
        p = pvec(3)                                     # gate, shift, scale
        bufs = {"X": x, "B": y, "P": with_pad(p)}
        spec |= {"a": row(T), "b": row(T), "y": row(T), "z": row(T), "p_off": EL, "n_par": 3,
                 "idx": {"gate": 0, "shift": 1, "scale": 2},
                 "tanh_gate": int(a.tanh_gate)}
        ps = bf(p).astype(np.float64)
        gate = bf(np.tanh(ps[0])).astype(np.float64) if a.tanh_gate else ps[0]
        xn = x.astype(np.float64) + gate * y.astype(np.float64)
        ref["Z"] = xn
        ref["Y"] = ln(bf(xn).astype(np.float64)) * (1 + ps[2]) + ps[1]
    elif op == "qk":
        qkv = bf(rng.standard_normal((T, 3 * EL)) * 2)
        wq, wk = rng.uniform(0.5, 1.5, HD), rng.uniform(0.5, 1.5, HD)
        n_txt, grid_w = 512, 32 if a.edit else 64
        bufs = {"X": qkv, "B": qkv, "P": with_pad(qk_params(wq, wk).reshape(3, EL))}
        spec |= {"a": row(T, 0, 3 * EL), "b": row(T, EL, 3 * EL), "y": row(T), "z": row(T),
                 "p_off": EL, "n_par": 3, "tok0": 0, "n_txt": n_txt, "grid_w": grid_w,
                 "heads": 24}
        g = np.arange(T)
        for name, cols, w in (("Y", slice(0, EL), wq), ("Z", slice(EL, 2 * EL), wk)):
            h = qkv[:, cols].astype(np.float64).reshape(T, 24, HD)
            hn = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6) * bf(w).astype(np.float64)
            ref[name] = rope_ref(hn, g, n_txt, grid_w).reshape(T, EL)
    elif op == "qk_flux3":
        n_txt = a.text_len if a.text_len is not None else 80
        grid_w = a.grid_w if a.grid_w is not None else 32
        grid_h = a.grid_h if a.grid_h is not None else 32
        if not 0 <= n_txt <= 512 or not 1 <= grid_w <= 64 or not 1 <= grid_h <= 64:
            raise ValueError("qk_flux3 fixture exceeds the current text/image RoPE table")
        if T < n_txt + grid_h * grid_w:
            raise ValueError("qk_flux3 fixture is smaller than its real token count")
        qkv = bf(rng.standard_normal((T, 3 * EL)) * 2)
        wq, wk = rng.uniform(0.5, 1.5, HD), rng.uniform(0.5, 1.5, HD)
        bufs = {"X": qkv, "B": qkv,
                "P": with_pad(flux3_qk_params(wq, wk).reshape(3, EL))}
        spec |= {"op": "qk_flux3", "a": row(T, 0, 3 * EL),
                 "b": row(T, EL, 3 * EL), "y": row(T), "z": row(T),
                 "p_off": EL, "n_par": 3, "tok0": 0, "n_txt": n_txt,
                 "grid_h": grid_h, "grid_w": grid_w, "heads": 24}
        ref_row = a.ref_row if a.ref_row is not None else 64
        if ref_row != 64:
            spec["ref_row"] = ref_row
        pos = np.zeros((T, 4), np.float64)
        pos[:n_txt, 3] = np.arange(n_txt)
        image = np.arange(grid_h * grid_w)
        rows_h = image // grid_w
        is_ref = rows_h >= ref_row
        # Published FLUX 3 reference ids: h restarts per reference, l = -1.
        pos[n_txt:n_txt + image.size, 1] = np.where(is_ref, rows_h - ref_row, rows_h) * 0.5
        pos[n_txt:n_txt + image.size, 2] = (image % grid_w) * 0.5
        pos[n_txt:n_txt + image.size, 3] = np.where(is_ref, -1.0, 0.0)
        freq = 10000.0 ** (-np.arange(0, 32, 2, dtype=np.float64) / 32)
        angle = (pos[:, :, None] * freq).reshape(T, 64)
        cos = np.repeat(np.cos(angle), 2, axis=1)[:, None]
        sin = np.repeat(np.sin(angle), 2, axis=1)[:, None]
        for name, cols, weight in (("Y", slice(0, EL), wq),
                                   ("Z", slice(EL, 2 * EL), wk)):
            h = qkv[:, cols].astype(np.float64).reshape(T, 24, HD)
            hn = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6)
            hn *= bf(weight).astype(np.float64)
            rot = np.stack((-hn[..., 1::2], hn[..., 0::2]), -1).reshape(hn.shape)
            ref[name] = (hn * cos + rot * sin).reshape(T, EL)
            ref[name][n_txt + grid_h * grid_w:] = 0     # the kernel zeroes padded rows
    elif op == "qk_qwen21":
        n_txt = a.text_len if a.text_len is not None else 15
        grid_h = a.grid_h if a.grid_h is not None else 4
        grid_w = a.grid_w if a.grid_w is not None else 4
        if EL != 4096 or not 0 <= n_txt <= 1023 or not 1 <= grid_h <= 64 or not 1 <= grid_w <= 64:
            raise ValueError("Qwen Image 2.1 q/k fixture needs EL=4096 and supported text/grid size")
        cond = a.ref_row or 0                         # edit: first condition-image row
        hw, gap = grid_h * grid_w, max(grid_h, grid_w)
        valid = n_txt + hw * (2 if cond else 1)
        if T < valid or not 0 <= cond < max(n_txt, 1):
            raise ValueError("Qwen Image 2.1 q/k fixture lacks token rows")
        qkv = bf(rng.standard_normal((T, 3 * EL)) * 2)
        wq, wk = rng.uniform(0.5, 1.5, HD), rng.uniform(0.5, 1.5, HD)
        bufs = {"X": qkv, "B": qkv, "P": with_pad(qwen21_qk_params(wq, wk).reshape(3, EL))}
        spec |= {"a": row(T, 0, 3 * EL), "b": row(T, EL, 3 * EL),
                 "y": row(T), "z": row(T), "p_off": EL, "n_par": 3,
                 "tok0": 0, "n_txt": n_txt, "grid_h": grid_h,
                 "grid_w": grid_w, "heads": 32, **({"cond_row": cond} if cond else {})}
        # Positions as diffusers' QwenImage21Rope: text advances all axes; an image
        # block freezes the frame at the text position before it on a centred grid, and
        # the following text resumes max(h, w) further on.
        image = np.arange(hw)
        grid = np.stack((image // grid_w - (grid_h - grid_h // 2),
                         image % grid_w - (grid_w - grid_w // 2)), -1)
        layout = ([("text", n_txt), ("image", hw)] if not cond else
                  [("text", cond), ("image", hw), ("text", n_txt - cond), ("image", hw)])
        rows, position = [], 0
        for kind, n in layout:
            if kind == "text":
                rows.append(np.repeat(np.arange(position, position + n)[:, None], 3, 1))
                position += n
            else:
                rows.append(np.concatenate((np.full((hw, 1), position), grid), 1))
                position += gap
        coords = np.concatenate(rows).astype(np.int32)
        angles = np.concatenate([
            coords[:, axis, None] * (10000.0 ** (-np.arange(0, dim, 2) / dim))[None]
            for axis, dim in enumerate((16, 56, 56))], axis=1)
        cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
        for name, cols, weight in (("Y", slice(0, EL), wq),
                                   ("Z", slice(EL, 2 * EL), wk)):
            h = qkv[:valid, cols].astype(np.float64).reshape(valid, 32, HD)
            hn = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6)
            hn = bf(hn * bf(weight).astype(np.float64)).astype(np.float64)
            paired = hn.reshape(valid, 32, 64, 2)
            rotated = np.empty_like(paired)
            rotated[..., 0] = paired[..., 0] * cos - paired[..., 1] * sin
            rotated[..., 1] = paired[..., 0] * sin + paired[..., 1] * cos
            values = np.zeros((T, EL), np.float64)
            values[:valid] = rotated.reshape(valid, EL)
            ref[name] = values
    elif op == "qk_moss":
        grid_h = a.grid_h or int(np.sqrt(T // 2))
        grid_w = a.grid_w or grid_h
        if grid_h < 1 or grid_w < 1 or grid_h > 64 or grid_w > 64 or grid_h * grid_w >= T:
            raise ValueError("qk_moss fixture needs grid in [1, 64] and text tokens")
        n_img = grid_h * grid_w
        n_txt = a.text_len if a.text_len is not None else T - n_img
        if not 1 <= n_txt <= min(4096, T - n_img):
            raise ValueError("qk_moss needs 1..4096 real text tokens within T")
        qkv = bf(rng.standard_normal((T, 3 * EL)) * 2)
        wq, wk = rng.uniform(0.5, 1.5, HD), rng.uniform(0.5, 1.5, HD)
        bufs = {"X": qkv, "B": qkv,
                "P": with_pad(moss_qk_params(wq, wk, grid_h, grid_w).reshape(3, EL))}
        spec |= {"a": row(T, 0, 3 * EL), "b": row(T, EL, 3 * EL),
                 "y": row(T), "z": row(T), "p_off": EL, "n_par": 3,
                 "tok0": 0, "n_txt": n_txt, "grid_h": grid_h, "grid_w": grid_w,
                 "heads": 24}
        pos = np.zeros((T, 4), np.float64)
        g = np.arange(n_img)
        pos[:n_img, 1] = n_txt
        pos[:n_img, 2] = g // grid_w - (grid_h - 1) / 2
        pos[:n_img, 3] = g % grid_w - (grid_w - 1) / 2
        pos[n_img:n_img + n_txt, 1] = np.arange(n_txt)
        freq = 10000.0 ** (-np.arange(16, dtype=np.float64) / 16)
        angle = (pos[:, :, None] * freq).reshape(T, 64)
        cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
        for name, cols, weight in (("Y", slice(0, EL), wq),
                                   ("Z", slice(EL, 2 * EL), wk)):
            h = qkv[:, cols].astype(np.float64).reshape(T, 24, HD)
            hn = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6)
            hn *= bf(weight).astype(np.float64)
            rot = np.concatenate([-hn[..., 64:], hn[..., :64]], -1)
            ref[name] = (hn * np.concatenate([cos, cos], -1) +
                         rot * np.concatenate([sin, sin], -1)).reshape(T, EL)
            ref[name][n_img + n_txt:] = 0
    elif op == "swiglu":
        ff = bf(rng.standard_normal((T, 6 * EL)) * 2)
        bufs = {"X": ff, "B": ff}
        spec |= {"a": row(T, 0, 6 * EL, 3), "b": row(T, 3 * EL, 6 * EL, 3),
                 "y": row(T, 0, 3 * EL, 3)}
        g, u = ff[:, :3 * EL].astype(np.float64), ff[:, 3 * EL:].astype(np.float64)
        ref["Y"] = g / (1 + np.exp(-g)) * u
    elif op == "situ":
        ff = np.zeros((T, 6 * EL), bfloat16)
        ff[:, :8192] = bf(rng.standard_normal((T, 8192)) * 2)
        ff[:, 3 * EL:3 * EL + 8192] = bf(rng.standard_normal((T, 8192)) * 2)
        bufs = {"X": ff, "B": ff}
        spec |= {"a": row(T, 0, 6 * EL, 3), "b": row(T, 3 * EL, 6 * EL, 3),
                 "y": row(T, 0, 3 * EL, 3)}
        hidden, gate = ff[:, :3 * EL].astype(np.float64), ff[:, 3 * EL:].astype(np.float64)
        sig = 1 / (1 + np.exp(-gate))
        ref["Y"] = (4 * np.tanh(gate / 4) * sig) * (25 * np.tanh(hidden / 25))
    elif op == "euler":
        n = -(-T // 24)
        n += (-n) % 16                                  # whole elements per core
        Tp = n * 24
        x = np.zeros((Tp, 128), bfloat16)
        x[:T] = bf(rng.standard_normal((T, 128)))
        v = np.zeros((Tp, 1024), bfloat16)
        v[:T, :128] = bf(rng.standard_normal((T, 128)))
        dt = np.float32(-0.2734)
        p = np.zeros((1, EL), dtype=np.float32)
        pb = np.zeros(EL, bfloat16)
        pb[:2] = np.array([dt], dtype=np.float32).view(np.uint16).view(bfloat16)
        bufs = {"X": x, "B": v, "P": np.concatenate([np.zeros(EL, bfloat16), pb,
                                                      np.zeros(EL, bfloat16)])}
        tile = lambda ld: {"kind": "tile24", "off": 0, "ld": ld, "n": n}  # noqa: E731
        spec |= {"a": tile(128), "b": tile(1024), "y": tile(128), "p_off": EL, "n_par": 1,
                 "real_T": T}
        ref["Y"] = x.astype(np.float64) + float(dt) * v[:, :128].astype(np.float64)
    elif op in ("rms", "res_rms"):
        W = a.W or 2560
        if not 32 <= W <= EL:
            raise ValueError("RMSNorm width must be in 32..EL")
        x = np.zeros((T, EL), bfloat16)
        x[:, :W] = bf(rng.standard_normal((T, W)) * 3)
        wv = np.zeros((1, EL), np.float32)
        wv[0, :W] = rng.uniform(0.2, 2.0, W)
        spec |= {"op": "ln_mod" if op == "rms" else "res_ln_mod", "norm": "rms", "W": W,
                 "a": row(T), "y": row(T), "p_off": EL, "n_par": 1, "idx": {"scale": 0}}
        bufs = {"X": x, "P": with_pad(wv)}
        xf = x.astype(np.float64)
        if op == "res_rms":
            y = np.zeros((T, EL), bfloat16)
            y[:, :W] = bf(rng.standard_normal((T, W)))
            bufs["B"] = y
            spec |= {"unit_gate": 1, "b": row(T), "z": row(T)}
            xf = xf + y.astype(np.float64)
            ref["Z"] = xf
            xf = bf(xf).astype(np.float64)
        n = xf[:, :W] / np.sqrt((xf[:, :W] ** 2).mean(-1, keepdims=True) + 1e-6)
        yref = np.zeros((T, EL))
        yref[:, :W] = bf(n).astype(np.float64) * bf(wv[0, :W]).astype(np.float64)
        ref["Y"] = yref
    elif op == "qk_qwen":                  # fused q|k|v [T, 6144]: A = q 0-23, B = q 24-31, k, v
        T = a.T or 512
        qkv = bf(rng.standard_normal((T, 2 * EL)) * 2)
        wq, wk = rng.uniform(0.5, 1.5, HD), rng.uniform(0.5, 1.5, HD)
        bufs = {"X": qkv, "B": qkv, "P": with_pad(qwen_qk_params(wq, wk).reshape(3, EL))}
        spec |= {"op": "qk", "rope": "qwen", "a": row(T, 0, 2 * EL), "b": row(T, EL, 2 * EL),
                 "y": row(T), "z": row(T), "p_off": EL, "n_par": 3, "heads": 24,
                 "b_q_heads": 8, "b_k_heads": 8}
        f = 1.0 / (QWEN_THETA ** (np.arange(0, 128, 2, dtype=np.float64) / 128))
        ang = np.outer(np.arange(T), f)
        cos = np.concatenate([np.cos(ang)] * 2, 1)[:, None, :]
        sin = np.concatenate([np.sin(ang)] * 2, 1)[:, None, :]

        def norm_rope(hd, w):
            hn = hd / np.sqrt((hd ** 2).mean(-1, keepdims=True) + 1e-6) * bf(w).astype(np.float64)
            rot = np.concatenate([-hn[..., 64:], hn[..., :64]], -1)
            return hn * cos + rot * sin

        x64 = qkv.astype(np.float64).reshape(T, 48, HD)
        q = norm_rope(x64[:, :32], wq)
        k = norm_rope(x64[:, 32:40], wk)
        ref["Y"] = q[:, :24].reshape(T, EL)
        ref["Z"] = np.concatenate([q[:, 24:], k, x64[:, 40:]], 1).reshape(T, EL)
    elif op in ("silu", "gelu_tanh"):
        x = bf(rng.standard_normal((T, EL)) * 3)
        bufs = {"X": x}
        spec |= {"a": row(T), "y": row(T)}
        xf = x.astype(np.float64)
        ref["Y"] = (xf / (1 + np.exp(-xf)) if op == "silu" else
                    0.5 * xf * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                                (xf + 0.044715 * xf ** 3))))

    outs = {"Y": ref["Y"].shape}
    if "Z" in ref:
        outs["Z"] = ref["Z"].shape
    sizes = {k: int(np.asarray(bufs[k]).size) if k in bufs else EL for k in ("X", "B", "P")}
    sizes |= {k: int(np.prod(outs[k])) if k in outs else EL for k in ("Y", "Z")}
    if op == "euler":
        sizes["Y"] = int(np.asarray(bufs["X"]).size)
    spec["sizes"] = sizes
    (out / "spec.json").write_text(json.dumps(spec))
    for k in ("X", "B", "P"):
        np.asarray(bufs.get(k, np.zeros(EL, bfloat16)), dtype=bfloat16).tofile(out / f"{k}.bin")
    np.savez(out / "ref.npz", op=op, **{f"ref_{k}": v for k, v in ref.items()},
             **{f"shape_{k}": np.array(v.shape) for k, v in ref.items()})

    if a.build:
        b = Path(a.build).resolve()
        cfg = ["device", f"xclbin G {b / 'final.xclbin'}", f"kernelx k G {b / 'insts.bin'}"]
        cfg += [f"buf {k} {sizes[k] * 2} {k}.bin" for k in ("X", "B", "P")]
        cfg += [f"buf {k} {sizes[k] * 2}" for k in ("Y", "Z")]
        cfg += ["run k X B P Y Z"] * a.runs
        cfg += [f"dump {k} {k}.out {sizes[k] * 2}" for k in outs] + [""]
        (out / "run.cfg").write_text("\n".join(cfg))
    print(f"{op}: T={T} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
