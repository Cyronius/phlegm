"""Check full 2520-key vision attention through 1024+1024+472 NPU chunks."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def run(driver: Path, full: Path, tail: Path, out: Path) -> dict:
    width = 64
    rng = np.random.default_rng(20261027)
    q = rng.normal(0, 1, width).astype(np.float32).astype(bfloat16)
    k = rng.normal(0, 1, (2520, width)).astype(np.float32).astype(bfloat16)
    v = rng.normal(0, 0.5, (2520, width)).astype(np.float32).astype(bfloat16)
    kv = np.stack((k, v), axis=1)
    scores = k.astype(np.float64) @ q.astype(np.float64)
    weights = np.exp(scores - scores.max())
    expected = weights @ v.astype(np.float64) / weights.sum()
    out.mkdir(parents=True, exist_ok=True)
    state = np.zeros(2 * width + 16, np.float32)
    reports = []
    start = 0
    for i, (rows, build) in enumerate(((1024, full), (1024, full), (472, tail))):
        stage = out / f"chunk-{i}"
        stage.mkdir(exist_ok=True)
        (stage / "meta.bin").write_bytes(q.tobytes() + state.tobytes())
        block = kv[start:start + rows]
        (stage / "kv.bin").write_bytes(block.tobytes())
        cfg = stage / "run.cfg"
        cfg.write_text("\n".join([
            "device", f"xclbin G {build / 'final.xclbin'}",
            f"kernelx k G {build / 'insts.bin'}",
            f"buf meta {q.nbytes + state.nbytes} {stage / 'meta.bin'}",
            f"buf kv {block.nbytes} {stage / 'kv.bin'}",
            f"buf result {state.nbytes}", "run k meta kv result",
            f"dump result {stage / 'result.bin'} {state.nbytes}", "",
        ]))
        proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
        state = np.fromfile(stage / "result.bin", np.float32)
        if state.size != 2 * width + 16 or not np.isfinite(state).all():
            raise AssertionError("invalid online attention state")
        start += rows
        report = dict(chunk=i, prefix_rows=start, state_mass=float(state[width + 1]))
        reports.append(report)
    actual = state[width + 16:].astype(np.float64)
    error = float(np.max(np.abs(actual - expected)))
    report = dict(keys=2520, head_dim=width, chunks=reports,
                  max_absolute_error=error,
                  passed=bool(error < 0.001 and state[width + 1] > 0))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--full-build", type=Path, required=True)
    parser.add_argument("--tail-build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.full_build.resolve(),
                         args.tail_build.resolve(), args.out.resolve()), indent=2))
