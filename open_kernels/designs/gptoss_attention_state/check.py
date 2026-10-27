"""Check full GPT-OSS sink attention over bounded NPU calls, including a tail."""
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
STATE_FLOATS = 68
CHUNK_ROWS = 512


def fixture(total_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261027 + total_rows)
    q = rng.normal(0, 0.4, (HEADS, HD)).astype(np.float32).astype(bfloat16)
    sinks = rng.normal(0, 1.5, HEADS).astype(np.float32)
    K = rng.normal(0, 0.4, (total_rows, KV_HEADS, HD)).astype(np.float32).astype(bfloat16)
    V = rng.normal(0, 0.6, (total_rows, KV_HEADS, HD)).astype(np.float32).astype(bfloat16)
    kv = np.ascontiguousarray(np.concatenate([K, V], axis=1))
    expected = sink_attention(q, K, V, sinks, pos=total_rows - 1).astype(np.float32)
    return q, sinks, kv, expected


def invoke(driver: Path, build: Path, directory: Path,
           q_state: np.ndarray, kv: np.ndarray) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "q_state.bin").write_bytes(q_state.tobytes())
    (directory / "kv.bin").write_bytes(kv.tobytes())
    config = directory / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf q_state {q_state.nbytes} {directory / 'q_state.bin'}",
        f"buf kv {kv.nbytes} {directory / 'kv.bin'}",
        f"buf result {HEADS * STATE_FLOATS * 4}",
        "run k q_state kv result",
        f"dump result {directory / 'state.bin'} {HEADS * STATE_FLOATS * 4}", "",
    ]))
    completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{directory.name}: NPU harness failed: "
                           f"{completed.stdout}\n{completed.stderr}")
    state = np.fromfile(directory / "state.bin", np.float32)
    if state.size != HEADS * STATE_FLOATS or not np.isfinite(state).all():
        raise AssertionError(f"{directory.name}: invalid state")
    state = state.reshape(HEADS, STATE_FLOATS)
    if np.any(state[:, 65] <= 0) or np.any(state[:, 66:] != 0):
        raise AssertionError(f"{directory.name}: invalid online softmax denominator or padding")
    return state


def run(driver: Path, build: Path, out: Path, total_rows: int) -> dict:
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    q, sinks, kv, expected = fixture(total_rows)
    state = np.zeros((HEADS, STATE_FLOATS), np.float32)
    state[:, 64] = sinks
    state[:, 65] = 1
    chunks = []
    for start in range(0, total_rows, CHUNK_ROWS):
        valid = min(CHUNK_ROWS, total_rows - start)
        meta = np.array([valid, 0, 0, 0], dtype=np.int32)
        q_state = np.frombuffer(q.tobytes() + state.tobytes() + meta.tobytes(), np.uint8).copy()
        rows = np.zeros((CHUNK_ROWS, 2 * KV_HEADS, HD), dtype=bfloat16)
        rows[:valid] = kv[start:start + valid]
        state = invoke(driver, build, out / f"chunk-{start // CHUNK_ROWS:02d}",
                       q_state, rows)
        chunks.append({"start_row": start, "end_row": start + valid,
                       "valid_rows": valid,
                       "denominator_min": float(np.min(state[:, 65]))})
    actual = state[:, :HD] / state[:, 65, None]
    a = actual.ravel().astype(np.float64)
    e = expected.ravel().astype(np.float64)
    err = np.abs(a - e)
    max_abs = float(err.max())
    max_rel = float(max_abs / max(np.max(np.abs(e)), 1e-30))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-30))
    passed = max_rel < 0.005 and cosine > 0.9999
    report = {"passed": bool(passed), "rows": total_rows, "chunk_rows": CHUNK_ROWS,
              "chunks": chunks, "max_absolute_error": max_abs,
              "max_relative_error": max_rel, "cosine": cosine,
              "reference": "FP64 full GQA attention with learned per-head sink"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--total-rows", type=int, default=4097)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.total_rows), indent=2))


if __name__ == "__main__":
    main()
