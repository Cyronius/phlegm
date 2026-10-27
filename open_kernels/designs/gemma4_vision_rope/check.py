"""Validate axial 2D vision RoPE against BF16-rounded Gemma 4 arithmetic."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def table_for(pos0: int, pos1: int) -> np.ndarray:
    inv = 100.0 ** (-np.arange(0, 32, 2, dtype=np.float32) / 32)
    parts = []
    for pos in (pos0, pos1):
        angle = np.float32(pos) * inv
        parts.extend((np.cos(angle), np.sin(angle)))
    return np.concatenate(parts).astype(bfloat16).astype(np.float32)


def expected(qk: np.ndarray, table: np.ndarray) -> np.ndarray:
    x = qk.astype(np.float64)
    t = table.astype(np.float64)
    out = np.empty_like(x)
    for axis in range(2):
        off = axis * 32
        out[:, off:off+16] = x[:, off:off+16] * t[off:off+16] - x[:, off+16:off+32] * t[off+16:off+32]
        out[:, off+16:off+32] = x[:, off+16:off+32] * t[off:off+16] + x[:, off:off+16] * t[off+16:off+32]
    return out.astype(bfloat16)


def run(driver: Path, build: Path, out: Path, pos0: int, pos1: int) -> dict:
    rng = np.random.default_rng(20261027 + pos0 * 97 + pos1)
    qk = rng.normal(0, 0.7, (24, 64)).astype(np.float32).astype(bfloat16)
    table = table_for(pos0, pos1)
    reference = expected(qk, table)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qk.bin").write_bytes(qk.tobytes())
    (out / "table.bin").write_bytes(table.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf qk {qk.nbytes} {out / 'qk.bin'}",
        f"buf table {table.nbytes} {out / 'table.bin'}",
        f"buf result {qk.nbytes}", "run k qk table result",
        f"dump result {out / 'result.bin'} {qk.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16)
    if actual.size != reference.size or not np.isfinite(actual.astype(np.float32)).all():
        raise AssertionError("invalid vision RoPE result")
    delta = np.abs(actual.astype(np.float32) - reference.ravel().astype(np.float32))
    report = dict(pos0=pos0, pos1=pos1, max_absolute_error=float(delta.max()),
                  exact_bf16_fraction=float(np.mean(delta == 0)),
                  passed=bool(delta.max() <= 0.015))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pos0", type=int, required=True)
    parser.add_argument("--pos1", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve(),
                         args.pos0, args.pos1), indent=2))
