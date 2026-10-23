#!/usr/bin/env python3
"""Gate two CLI logit dumps on identical position coverage, finite values and fidelity.

Run each CLI with the same --ids-file and --prefill-logits --dump-logits PREFIX.
This compares *all* _p<position>.bin files, never just their intersection.
For an allocation/layout-only change use --exact. Timing runs must omit dumps.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np


def positions(prefix):
    prefix = Path(prefix)
    pattern = re.compile(re.escape(prefix.name) + r"_p(\d+)\.bin$")
    return {int(m[1]): p for p in prefix.parent.iterdir() if (m := pattern.fullmatch(p.name))}


def compare(reference, candidate, vocab=0, min_cosine=0.999, min_argmax=0.99, exact=False):
    a, b = positions(reference), positions(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("position coverage differs or is empty")
    cosines, correlations, margins = [], [], []
    argmax = identical = top5 = 0
    for pos in sorted(a):
        x, y = (np.fromfile(p[pos], dtype="<f4").astype(np.float64) for p in (a, b))
        if x.shape != y.shape or not len(x) or vocab < 0 or vocab > len(x):
            raise ValueError(f"position {pos}: logit dimensions disagree")
        if vocab:
            x, y = x[:vocab], y[:vocab]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(f"position {pos}: nonfinite logits")
        identical += int(np.array_equal(x.view(np.uint64), y.view(np.uint64)))
        argmax += int(x.argmax() == y.argmax())
        k = min(5, len(x))
        top5 += int(set(np.argsort(-x, kind="stable")[:k]) == set(np.argsort(-y, kind="stable")[:k]))
        norms = np.linalg.norm(x) * np.linalg.norm(y)
        cosines.append(float(x @ y / norms) if norms else float(np.array_equal(x, y)))
        xc, yc = x - x.mean(), y - y.mean()
        centered = np.linalg.norm(xc) * np.linalg.norm(yc)
        correlations.append(float(xc @ yc / centered) if centered else float(np.array_equal(x, y)))
        if x.argmax() != y.argmax():
            best = np.sort(x)[-2:]
            margins.append({"position": pos, "reference_margin": float(best[-1] - best[-2]) if len(best)>1 else 0})
    n = len(a)
    result = {"positions": n, "first": min(a), "last": max(a), "argmax_equal": argmax,
              "top5_equal": top5, "identical_positions": identical,
              "min_cosine": min(cosines), "min_correlation": min(correlations), "flips": margins}
    result["passed"] = ((identical == n) if exact else
                         (min(cosines) >= min_cosine and min(correlations) >= min_cosine and argmax / n >= min_argmax))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--min-argmax", type=float, default=0.99)
    ap.add_argument("--exact", action="store_true")
    a = ap.parse_args()
    try:
        result = compare(a.reference, a.candidate, a.vocab, a.min_cosine, a.min_argmax, a.exact)
    except ValueError as e:
        ap.error(str(e))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
