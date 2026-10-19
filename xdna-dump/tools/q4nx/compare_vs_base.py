"""Validate a converted q4nx against a known-good q4nx of the SAME architecture,
tensor by tensor, by cosine similarity in logical (dequantized) order.

Why this works even for a different checkpoint: a pruned + LoRA-healed model's
weights stay ~0.99 correlated with the base layer they came from, so a wrong
tensor mapping / permutation in the converter shows up as ~0-0.5 while a right
one reads ~0.99+ (quant noise). This is how the linear-attention head-pairing
bug and the double -exp on ssm_a were found on 2026-09-01 (see README).

Usage:
  python compare_vs_base.py CONVERTED.q4nx BASE.q4nx [--pairs 0:0,1:2,2:3,...]
  (default pairs assume the 27B prune: kept base layers 0,2,3 | 4,6,7 | ...)
Prints the mapping per converted layer; anything below ~0.95 on a non-expert
tensor is a converter bug. Stacked expert tensors are skipped (too large).
"""
import argparse, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kernel-interp"))
sys.path.insert(0, r"C:/code/FastFlowLM/tools/kernel-interp")
from q4nx import Q4NX  # noqa: E402


def load(m, name):
    t = m.tensors[name]
    if t["dtype"] == "I8":
        if name == "lm_head.weight":
            return None
        s = t["shape"]
        if len(s) == 3:
            out, inn = s[0] * 32, s[1] * 256
            if out * inn > 40_000_000:  # stacked experts
                return None
            return m.matmul_w(name, out, inn)
        return None
    if t["dtype"] == "BF16":
        return m.bf16(name)
    if t["dtype"] == "F32":
        return m.f32(name)
    return None


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.shape != b.shape:
        return float("nan")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def default_pairs(n):
    """27B prune: each [L,L,L,F] base block keeps L0, L2, F -> base ids 0,2,3 / 4,6,7 / ..."""
    out = []
    for blk in range(n // 3):
        b = 4 * blk
        out += [(3 * blk, b), (3 * blk + 1, b + 2), (3 * blk + 2, b + 3)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("converted")
    ap.add_argument("base")
    ap.add_argument("--pairs", default=None, help="conv:base,conv:base,...")
    ap.add_argument("--layers", type=int, default=6, help="how many converted layers to check")
    a = ap.parse_args()
    C, B = Q4NX(a.converted), Q4NX(a.base)
    nC = len({k.split(".")[2] for k in C.tensors if k.startswith("model.layer.")})
    pairs = [tuple(map(int, p.split(":"))) for p in a.pairs.split(",")] if a.pairs else default_pairs(nC)
    pairs = pairs[: a.layers]
    print(f"converted fmt={C.fmt} layers={nC}; base fmt={B.fmt}")
    for g in ["model.embed_tokens.weight", "model.norm.weight"]:
        print(f"{g:44s} cos={cos(load(C, g), load(B, g)):.4f}")
    worst = 1.0
    for cl, bl in pairs:
        p = f"model.layer.{cl}."
        print(f"\n=== converted layer {cl} vs base layer {bl}")
        for k in sorted(C.tensors):
            if not k.startswith(p):
                continue
            suf = k[len(p):]
            bn = f"model.layer.{bl}.{suf}"
            if bn not in B.tensors:
                print(f"  {suf:45s} (no base tensor)")
                continue
            x = load(C, k)
            if x is None:
                continue
            c = cos(x, load(B, bn))
            worst = min(worst, c)
            flag = "" if c > 0.95 else "   <-- MISMATCH"
            print(f"  {suf:45s} {c:7.4f}{flag}")
    print(f"\nworst non-expert cos: {worst:.4f}  ({'OK' if worst > 0.95 else 'CONVERTER BUG'})")


if __name__ == "__main__":
    main()
