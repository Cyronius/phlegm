"""Check the 12-head, 64-lane Gemma 4 vision attention image on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def run(driver: Path, build: Path, out: Path, tokens: int = 256,
        valid_len: int | None = None, qk_scale: float = 1.0) -> dict:
    heads, width = 12, 64
    valid_len = tokens if valid_len is None else valid_len
    if not 0 < valid_len <= tokens:
        raise ValueError("valid length must fit padded token count")
    rng = np.random.default_rng(20261027)
    q, k, v = (rng.normal(0, scale, (heads, tokens, width))
               .astype(np.float32).astype(bfloat16)
               for scale in (qk_scale, qk_scale, 0.5))
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, array in (("q", q), ("k", k), ("v", v)):
        paths[name] = out / f"{name}.bin"
        paths[name].write_bytes(array.tobytes())
    output_bytes = heads * tokens * width * 2
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx fa G {build / 'insts.bin'}",
        *(f"buf {name} {paths[name].stat().st_size} {paths[name]}" for name in ("q", "k", "v")),
        f"buf result {output_bytes}", "run fa q k v result",
        f"dump result {out / 'result.bin'} {output_bytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", dtype=bfloat16).astype(np.float32)
    actual = actual.reshape(heads, tokens, width)
    q64, k64, v64 = (a.astype(np.float64) for a in (q, k, v))
    # Gemma4VisionAttention.scaling is 1.0 after per-head Q/K RMSNorm.
    logits = q64 @ k64.transpose(0, 2, 1)
    logits[:, :, valid_len:] = -np.inf
    logits -= logits.max(axis=-1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=-1, keepdims=True)
    expected = (prob @ v64).astype(np.float32)
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    abs_error = float(np.max(np.abs(difference)))
    relative = abs_error / max(float(np.max(np.abs(expected))), 1e-20)
    flat_a, flat_e = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    cosine = float(flat_a @ flat_e / (np.linalg.norm(flat_a) * np.linalg.norm(flat_e)))
    report = dict(tokens=tokens, valid_len=valid_len, heads=heads, head_dim=width,
                  qk_scale=qk_scale,
                  max_absolute_error=abs_error, max_relative_error=relative,
                  cosine=cosine,
                  passed=bool(np.isfinite(actual).all() and relative < 0.04 and cosine > 0.999))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--valid-len", type=int)
    parser.add_argument("--qk-scale", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve(),
                         args.tokens, args.valid_len, args.qk_scale), indent=2))
