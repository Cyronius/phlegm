"""Check generic q4 NPU LM heads at Gemma 4 text widths and vocabulary size."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
from q4nx import dq_chunks_q4_1  # noqa: E402

WIDTHS = {"e2b": 1536, "e4b": 2560, "12b": 3840}
VOCAB_BANDS = 262144 // 64
CHUNK = 5120


def fixture(width: int, seed: int, activation: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    k_chunks = width // 256
    raw = rng.integers(0, 256, (k_chunks, 2, CHUNK), dtype=np.uint8)
    scales = rng.uniform(0.003, 0.025, (k_chunks, 2, 256)).astype(np.float32).astype(bfloat16)
    mins = rng.uniform(-0.15, -0.02, (k_chunks, 2, 256)).astype(np.float32).astype(bfloat16)
    raw[:, :, :512] = scales.view(np.uint8).reshape(k_chunks, 2, 512)
    raw[:, :, 512:1024] = mins.view(np.uint8).reshape(k_chunks, 2, 512)
    x = (rng.normal(0, 0.5, width).astype(np.float32).astype(bfloat16)
         if activation is None else activation)
    weights = np.empty((64, width), np.float32)
    for rb in range(2):
        decoded = dq_chunks_q4_1(np.ascontiguousarray(raw[:, rb])).reshape(k_chunks, 32, 256)
        weights[rb * 32:(rb + 1) * 32] = decoded.transpose(1, 0, 2).reshape(32, width)
    expected = (weights.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    return raw.reshape(-1), x, expected


def run(driver: Path, build: Path, out: Path, model: str, bands: int) -> dict:
    if bands not in (1, VOCAB_BANDS):
        raise ValueError("bands must be one or the full Gemma 4 vocabulary")
    width = WIDTHS[model]
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    first, x, first_expected = fixture(width, 20261027)
    if bands == 1:
        variants, expected = (first,), first_expected
    else:
        second, _, second_expected = fixture(width, 20261028, x)
        variants = (first, second)
        expected = np.tile(np.concatenate((first_expected, second_expected)), bands // 2)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "weights.bin").open("wb") as fh:
        parts = tuple(v.tobytes() for v in variants)
        for i in range(bands):
            fh.write(parts[i % len(parts)])
    (out / "activation.bin").write_bytes(x.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf weights {first.nbytes * bands} {out / 'weights.bin'}",
        f"buf activation {x.nbytes} {out / 'activation.bin'}",
        f"buf result {expected.nbytes}", "run k weights activation result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("invalid head output")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    maximum = float(np.max(np.abs(a - e)))
    rel = maximum / float(np.max(np.abs(e)))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    report = {"model": model, "hidden": width, "vocab_logits": bands * 64,
              "max_absolute_error": maximum, "max_relative_error": rel,
              "cosine": cosine, "passed": bool(rel < 0.005 and cosine > 0.9999),
              "reference": "FP64 matvec of independently decoded synthetic q4_1 chunks"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", choices=WIDTHS, required=True)
    p.add_argument("--bands", type=int, choices=(1, VOCAB_BANDS), default=1)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.model, a.bands), indent=2))


if __name__ == "__main__":
    main()
