r"""Check a dit_ew run (make_test.py's Y.out / Z.out) against the float64 reference.

    python compare.py <testdir>

Prints rel_fro per output next to the floor (the reference itself rounded to bf16) and
PASSes at GATE_REL_FRO: a few bf16 roundings of fp32 math, plus the tanh approximation in
SiLU (see ew.cc).

The model-specific q/k streams (qk_flux3, qk_moss, qk_qwen21) write zeros to the rows
past the stream's real tokens (n_txt + grid_h * grid_w, or two image blocks for a Qwen
edit's ``cond_row`` layout) by design. Those rows are
checked to be exactly zero, separately from the error of the live rows; the reference a
fixture stores for them (older qk_flux3 fixtures fill them) is not compared. qk_flux3
also reports the text, image and reference-image row segments on their own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

GATE_REL_FRO = 1e-2
ZERO_PAD_OPS = ("qk_flux3", "qk_moss", "qk_qwen21")


def live_rows(spec: dict) -> int | None:
    """Rows the kernel computes, or None when every row is live."""
    if spec.get("op") not in ZERO_PAD_OPS:
        return None
    blocks = 2 if spec.get("op") == "qk_qwen21" and spec.get("cond_row") else 1
    return spec.get("n_txt", 0) + blocks * spec.get("grid_h", 0) * spec.get("grid_w", 0)


def segments(spec: dict, valid: int) -> dict[str, slice]:
    """qk_flux3's text / image / reference-image row ranges (real rows only)."""
    if spec.get("op") != "qk_flux3":
        return {}
    n_txt, w = spec.get("n_txt", 0), spec.get("grid_w", 1)
    ref_row = spec.get("ref_row", 64)
    split = min(valid, n_txt + ref_row * w)
    out = {"text": slice(0, n_txt), "image": slice(n_txt, split)}
    if split < valid:
        out["reference"] = slice(split, valid)
    return {k: v for k, v in out.items() if v.stop > v.start}


def stats(got: np.ndarray, ref: np.ndarray) -> tuple[float, float, float, float]:
    n = np.linalg.norm(ref)
    floor = np.linalg.norm(ref.astype(np.float32).astype(bfloat16).astype(np.float64) - ref)
    rel = np.linalg.norm(got - ref) / n
    cos = float(np.vdot(got, ref) / (np.linalg.norm(got) * n)) if n else float("nan")
    worst = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    return float(rel), float(floor / n), cos, float(worst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("testdir")
    a = ap.parse_args()
    d = Path(a.testdir)
    r = np.load(d / "ref.npz")
    spec_path = d / "spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {"op": str(r["op"])}
    ok = True
    for k in ("Y", "Z"):
        if f"ref_{k}" not in r:
            continue
        ref = r[f"ref_{k}"]
        got = np.fromfile(d / f"{k}.out", dtype=bfloat16).astype(np.float64)[:ref.size]
        got = got.reshape(ref.shape)
        fin = bool(np.isfinite(got).all())
        valid = live_rows(spec)
        pad_note = ""
        pad_ok = True
        if valid is not None and valid < ref.shape[0]:
            pad = got[valid:]
            pad_ok = bool(np.all(pad == 0))
            pad_note = (f", {ref.shape[0] - valid} padded rows "
                        f"{'all zero' if pad_ok else f'NONZERO (max |x| {np.abs(pad).max():.3e})'}")
            got, ref = got[:valid], ref[:valid]
        rf, floor, cos, worst = stats(got, ref)
        passed = fin and rf <= GATE_REL_FRO and pad_ok
        ok &= passed
        rows = f"{got.shape[0]} live rows" if valid is not None else f"{got.shape[0]} rows"
        print(f"  {spec['op']} {k}: {rows}: rel_fro {rf:.3e} (bf16 floor {floor:.3e}), "
              f"cos {cos:.6f}, max err / max |ref| {worst:.3e}, finite {fin}{pad_note}  "
              f"{'PASS' if passed else 'FAIL'}")
        for name, rows_ in segments(spec, got.shape[0]).items():
            srf, _, scos, _ = stats(got[rows_], ref[rows_])
            sok = srf <= GATE_REL_FRO
            ok &= sok
            print(f"    {name:9s} rows {rows_.start}-{rows_.stop - 1}: rel_fro {srf:.3e}, "
                  f"cos {scos:.6f}  {'PASS' if sok else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
