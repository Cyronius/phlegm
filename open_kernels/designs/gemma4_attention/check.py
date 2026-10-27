"""Check bounded Gemma 4 attention and its carry state against FP64 softmax."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def fixture(width: int, rows: int, chunks: int):
    rng = np.random.default_rng(20261027 + width + rows * chunks)
    q = rng.normal(0, 0.7, width).astype(np.float32).astype(bfloat16)
    kv = rng.normal(0, 0.7, (rows * chunks, 2, width)).astype(np.float32).astype(bfloat16)
    return q, kv


def reference(q: np.ndarray, kv: np.ndarray) -> tuple[np.ndarray, float]:
    x = q.astype(np.float64)
    k = kv[:, 0].astype(np.float64)
    v = kv[:, 1].astype(np.float64)
    scores = k @ x  # Gemma 4 text attention scaling is exactly 1.0.
    maximum = float(np.max(scores))
    weights = np.exp(scores - maximum)
    total = float(np.sum(weights))
    return (weights @ v) / total, maximum + np.log(total)


def run(driver: Path, build: Path, out: Path, width: int, rows: int, chunks: int) -> dict:
    q, kv = fixture(width, rows, chunks)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    state = np.zeros(2 * width + 16, np.float32)
    reports = []
    for i in range(chunks):
        block = kv[i * rows:(i + 1) * rows]
        stage = out / f"chunk-{i:03d}"
        stage.mkdir(exist_ok=True)
        (stage / "meta.bin").write_bytes(q.tobytes() + state.tobytes())
        (stage / "kv.bin").write_bytes(block.tobytes())
        config = stage / "run.cfg"
        config.write_text("\n".join([
            "device", f"xclbin G {build / 'final.xclbin'}",
            f"kernelx k G {build / 'insts.bin'}",
            f"buf meta {q.nbytes + state.nbytes} {stage / 'meta.bin'}",
            f"buf kv {block.nbytes} {stage / 'kv.bin'}",
            f"buf result {state.nbytes}", "run k meta kv result",
            f"dump result {stage / 'result.bin'} {state.nbytes}", "",
        ]))
        proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
        state = np.fromfile(stage / "result.bin", np.float32)
        if state.size != 2 * width + 16 or not np.isfinite(state).all() or state[width + 1] <= 0:
            raise AssertionError("invalid attention state")
        expected, logsumexp = reference(q, kv[:(i + 1) * rows])
        actual = state[width + 16:].astype(np.float64)
        max_abs = float(np.max(np.abs(actual - expected)))
        cosine = float(actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        logmass = float(state[width] + np.log(state[width + 1]))
        row = {"chunk": i, "prefix_rows": (i + 1) * rows,
               "max_absolute_error": max_abs, "cosine": cosine,
               "logsumexp_error": abs(logmass - logsumexp),
               "passed": bool(max_abs <= 0.001 and cosine > 0.99999 and
                              abs(logmass - logsumexp) < 0.001)}
        reports.append(row)
        if not row["passed"]:
            raise AssertionError(row)
    report = {"head_dim": width, "rows_per_chunk": rows, "chunks": chunks,
              "passed": all(r["passed"] for r in reports), "checks": reports}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--head-dim", required=True, type=int, choices=(64, 256, 512))
    p.add_argument("--rows", required=True, type=int, choices=(1, 128, 472, 512, 1024))
    p.add_argument("--chunks", type=int, default=1)
    a = p.parse_args()
    if a.chunks < 1:
        p.error("--chunks must be positive")
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.head_dim, a.rows, a.chunks), indent=2))


if __name__ == "__main__":
    main()
