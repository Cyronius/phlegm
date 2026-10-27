"""Compare GPT-OSS 20B NPU sink attention with the checked FP64 layer replica."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from replica_gptoss import sink_attention  # noqa: E402

HEADS = 64
KV_HEADS = 8
HD = 64


def fixture(rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261027 + rows)
    q = rng.normal(0, 0.4, (HEADS, HD)).astype(np.float32).astype(bfloat16)
    sinks = rng.normal(0, 1.5, HEADS).astype(np.float32)
    # The local case provides a 256-row cache, then sends only its last 128
    # rows to the NPU. Its oracle applies the model's window rule to all 256.
    cache_rows = 256 if rows == 128 else rows
    K = rng.normal(0, 0.4, (cache_rows, KV_HEADS, HD)).astype(np.float32).astype(bfloat16)
    V = rng.normal(0, 0.6, (cache_rows, KV_HEADS, HD)).astype(np.float32).astype(bfloat16)
    selected_k = K[-rows:]
    selected_v = V[-rows:]
    q_sink = np.frombuffer(q.tobytes() + sinks.tobytes(), dtype=np.uint8).copy()
    kv = np.ascontiguousarray(np.concatenate([selected_k, selected_v], axis=1))
    expected = sink_attention(q, K, V, sinks,
                              sliding_window=128 if rows == 128 else None,
                              pos=cache_rows - 1).astype(np.float32)
    return q_sink, kv, expected


def run(driver: Path, build: Path, out: Path, rows: int) -> dict:
    if rows not in (1, 128, 256):
        raise ValueError("rows must match a compiled 1, 128 or 256 row image")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    q_sink, kv, expected = fixture(rows)
    (out / "q_sink.bin").write_bytes(q_sink.tobytes())
    (out / "kv.bin").write_bytes(kv.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf q_sink {q_sink.nbytes} {out / 'q_sink.bin'}",
        f"buf kv {kv.nbytes} {out / 'kv.bin'}",
        f"buf result {expected.nbytes}", "run k q_sink kv result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"NPU harness failed: {completed.stdout}\n{completed.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32)
    if actual.shape != (HEADS * HD,) or not np.isfinite(actual).all():
        raise AssertionError("invalid NPU output shape or nonfinite value")
    a = actual.astype(np.float64)
    e = expected.ravel().astype(np.float64)
    err = np.abs(a - e)
    max_abs = float(err.max())
    max_rel = float(max_abs / max(np.max(np.abs(e)), 1e-30))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-30))
    passed = max_rel < 0.005 and cosine > 0.9999
    report = {"passed": bool(passed), "rows": rows,
              "mode": "sliding-128 from 256-row cache" if rows == 128 else "full",
              "max_absolute_error": max_abs, "max_relative_error": max_rel,
              "cosine": cosine,
              "reference": "FP64 GQA attention with learned sink and no sink value"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rows", required=True, type=int, choices=[1, 128, 256])
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.rows), indent=2))


if __name__ == "__main__":
    main()
