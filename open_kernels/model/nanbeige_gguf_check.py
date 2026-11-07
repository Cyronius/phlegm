r"""Check a dense Llama GGUF view's packed bytes (OPEN-FAMILY-NANBEIGE: the q4_1-law view, recipes/dense.py
gguf_q41_view) against an independent reading of the same GGUF.

`gguf_pack_dump` (upstream-patches/nanbeige-runtime.patch) packs layers and the head from model.gguf exactly
as the engine does (pools.cpp std_perm_gguf / put / lmhead_q8). This script dequantizes the GGUF with
llama.cpp's own gguf-py, undoes the converter's Q / K row permutation (convert_hf_to_gguf
LlamaModel.permute: HF row h*hd + s*hd/2 + i is GGUF row h*hd + 2i + s) in NumPy, and decodes the packed
q4_1-law chunks (std_perm band order) and q8 head chunks (128-row supertiles) back to matrices.

Q4_0 projections must agree to EXACT_Q40 (0 for a pure Q4_0 from BF16, whose scales are bf16 values; the
bf16 rounding of an fp16 scale otherwise, <= 2^-9 relative a block); the Q8_0 head to EXACT_Q80 (its fp16
scale rounded to bf16, codes kept); the norm vectors bit-exactly after bf16 rounding.

    gguf_pack_dump manifest.json model.gguf OUT 0,25,51 head
    python open_kernels/model/nanbeige_gguf_check.py --gguf model.gguf --manifest manifest.json \
        --dump OUT --layers 0,25,51 --head [--gguf-py <llama.cpp>/gguf-py]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35moe_gguf_check import Gguf, bf16, from_bf16, q41_chunks, q8_chunks, stats  # noqa: E402

EXACT_Q40 = 2.5e-3
EXACT_Q80 = 2.5e-3
HF_TO_GGUF = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
              "mlp.down_proj": "ffn_down", "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm"}


def gguf_name(hf: str) -> str:
    if hf == "lm_head.weight":
        return "output.weight"
    if hf == "model.norm.weight":
        return "output_norm.weight"
    if hf == "model.embed_tokens.weight":
        return "token_embd.weight"
    parts = hf.split(".")                    # model.layers.L.<module...>.weight
    l, mod = parts[2], ".".join(parts[3:-1])
    return f"blk.{l}.{HF_TO_GGUF[mod]}.weight"


def unpermute(w: np.ndarray, heads: int) -> np.ndarray:
    """GGUF Q / K rows (adjacent rotary pairs) -> HF order (split halves)."""
    rows = w.shape[0]
    hd = rows // heads
    r = np.arange(rows)
    src = r // hd * hd + (r % (hd // 2)) * 2 + (r % hd) // (hd // 2)
    return w[src]


def std_perm_matrix(chunks: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    """[n, 32, 256] pool chunks in std_perm order -> [out_dim, in_dim] (chunk c: rows 64 (c // per_band)
    + 32 (c % 2), k-tile (c % per_band) // 2)."""
    ncol = in_dim // 256
    a = chunks.reshape(out_dim // 64, ncol, 2, 32, 256)
    return a.transpose(0, 2, 3, 1, 4).reshape(out_dim, in_dim)


def q8_head_matrix(chunks: np.ndarray, rows: int, in_dim: int) -> np.ndarray:
    """[n, 32, 256] head chunks (pools.cpp lmhead_q8, raw GGUF: chunk ((row / 128) ncol + kt) 4 + (row % 128) / 32)."""
    ncol = in_dim // 256
    a = chunks.reshape(rows // 128, ncol, 4, 32, 256)
    return a.transpose(0, 2, 3, 1, 4).reshape(rows, in_dim)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layers", default="0")
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--gguf-py", default=None)
    a = ap.parse_args()
    g = Gguf(a.gguf, a.gguf_py)
    top = json.loads(Path(a.manifest).read_text())
    man = top["gguf"]
    ch = man["pack"]["chunk_bytes"]
    if ch != 5120:
        raise SystemExit(f"the gguf section's chunk is {ch} B; this checks the q4_1 law's 5120")
    spec = man["spec"]
    nh, nkv = spec["num_heads"], spec["num_kv_heads"]
    bad = 0

    def report(name, typ, got, want, lim):
        nonlocal bad
        rel, mx = stats(got, want)
        ok = rel <= lim
        bad += not ok
        print(f"  {name:44s} {typ:5s} rel L2 {rel:.3e}  max {mx:.3e}  {'ok' if ok else 'FAIL'}")

    for l in map(int, a.layers.split(",")):
        lt_name = man["layers"][l]
        lt = man["layer_types"][lt_name]
        pool = np.fromfile(Path(a.dump) / f"pool{l}.bin", np.uint8)
        consts = np.fromfile(Path(a.dump) / f"consts{l}.bin", np.uint8)
        print(f"layer {l} ({lt_name}):")
        for op in lt["pack"]["pool"]:
            hf = op["tensor"].replace("{l}", str(l))
            name = gguf_name(hf)
            W = g.array(name)
            if "q_proj" in hf:
                W = unpermute(W, nh)
            elif "k_proj" in hf:
                W = unpermute(W, nkv)
            rows, cols = W.shape
            if op["op"] != "std_perm_gguf" or op["in_dim"] != cols or op["nch"] * 8192 != rows * cols:
                print(f"  {hf}: op {op} does not cover [{rows}, {cols}]  FAIL")
                bad += 1
                continue
            got = std_perm_matrix(q41_chunks(pool[op["dst"]: op["dst"] + op["nch"] * ch].reshape(-1, ch)), rows, cols)
            report(hf.split("layers.")[1], g.type(name), got, W, EXACT_Q40 if g.type(name) == "Q4_0" else 0.0)
        for op in lt["pack"]["consts"]:
            hf = op["tensor"].replace("{l}", str(l))
            name = gguf_name(hf)
            W = g.array(name).ravel()
            got = from_bf16(consts[op["dst"]: op["dst"] + 2 * W.size].copy().view(np.uint16))
            report(hf.split("layers.")[1], g.type(name), got, bf16(W), 0.0)
    if a.head:
        # by 8192-row slices (64 supertiles, a contiguous run of chunks): the whole head is 4.8 GB as f32
        op = man["pack"]["lm_head"]["ops"][0]
        t = g.t["output.weight"]
        cols, rows = int(t.shape[0]), int(t.shape[1])
        CH8 = op["chunk_bytes"]
        head = np.memmap(Path(a.dump) / "lmhead.bin", np.uint8, "r")
        data = t.data.reshape(rows, -1)
        se = sw = 0.0
        mx = 0.0
        for r0 in range(0, rows, 8192):
            R = min(8192, rows - r0)
            c0, n = r0 * cols // 8192, R * cols // 8192
            got = q8_head_matrix(q8_chunks(np.asarray(head[c0 * CH8: (c0 + n) * CH8]).reshape(n, CH8)), R, cols)
            want = g.dq(data[r0:r0 + R], t.tensor_type).astype(np.float32).reshape(R, cols)
            d = got.astype(np.float64) - want
            se += float((d * d).sum())
            sw += float((want.astype(np.float64) ** 2).sum())
            mx = max(mx, float(np.abs(d).max()))
        rel = (se / max(sw, 1e-30)) ** 0.5
        ok = rel <= EXACT_Q80
        bad += not ok
        print("head:")
        print(f"  {'lm_head.weight':44s} {t.tensor_type.name:5s} rel L2 {rel:.3e}  max abs {mx:.3e}  {'ok' if ok else 'FAIL'}")
    print("ALL OK" if not bad else f"{bad} FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
