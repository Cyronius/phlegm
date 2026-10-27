"""Check one GPT-OSS q4 LM-head band through the generic 3072-wide NPU image."""
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
from recipes.gptoss_source import CHUNK, GptOssSource, HEAD_BANDS  # noqa: E402


def fixture(seed: int = 20261027, activation: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = rng.integers(0, 256, (23, 2, CHUNK), dtype=np.uint8)
    scales = rng.uniform(0.003, 0.025, (23, 2, 128)).astype(np.float32).astype(bfloat16)
    mins = rng.uniform(-0.15, -0.02, (23, 2, 128)).astype(np.float32).astype(bfloat16)
    half[:, :, :256] = scales.view(np.uint8).reshape(23, 2, 256)
    half[:, :, 256:512] = mins.view(np.uint8).reshape(23, 2, 256)

    class FixtureSource(GptOssSource):
        def _bytes(self, *args, **kwargs):
            return half.tobytes()

    packed = FixtureSource(None).head_band(0)
    if activation is None:
        x = np.zeros(3072, bfloat16)
        x[:2880] = rng.normal(0, 0.5, 2880).astype(np.float32).astype(bfloat16)
    else:
        x = np.asarray(activation)
        if x.shape != (3072,) or x.dtype != bfloat16 or np.any(x[2880:] != 0):
            raise ValueError("head activation must be padded BF16[3072]")
    weights = np.empty((64, 3072), np.float32)
    for rb in range(2):
        chunks = np.ascontiguousarray(packed.reshape(12, 2, 5120)[:, rb])
        tile = dq_chunks_q4_1(chunks).reshape(12, 32, 256)
        weights[rb * 32:(rb + 1) * 32] = tile.transpose(1, 0, 2).reshape(32, 3072)
    expected = (weights.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    return packed, x, expected


def run(driver: Path, build: Path, out: Path, bands: int = 1) -> dict:
    if bands not in (1, HEAD_BANDS):
        raise ValueError("head check supports one band or the full vocabulary")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    packed, x, one_expected = fixture()
    if bands == 1:
        variants = (packed,)
        expected = one_expected
    else:
        other, _, other_expected = fixture(20261028, x)
        variants = (packed, other)
        expected = np.tile(np.concatenate((one_expected, other_expected)), bands // 2)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "weights.bin").open("wb") as weights:
        chunks = tuple(v.tobytes() for v in variants)
        for i in range(bands):
            weights.write(chunks[i % len(chunks)])
    (out / "x.bin").write_bytes(x.tobytes())
    (out / "expected.bin").write_bytes(expected.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf weights {packed.nbytes * bands} {out / 'weights.bin'}",
        f"buf x {x.nbytes} {out / 'x.bin'}",
        f"buf result {expected.nbytes}", "run k weights x result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"NPU harness failed: {completed.stdout}\n{completed.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("invalid GPT-OSS head NPU output")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    error = np.abs(a - e)
    max_abs = float(error.max())
    max_rel = float(max_abs / max(np.max(np.abs(e)), 1e-30))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    report = {"passed": bool(max_rel < 0.005 and cosine > 0.9999),
              "logits": 64 * bands, "bands": bands,
              "hidden": 2880, "padded_hidden": 3072,
              "max_absolute_error": max_abs, "max_relative_error": max_rel,
              "cosine": cosine,
              "reference": "FP64 product of decoded GPT-OSS q4_1 half-chunk head band"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bands", type=int, choices=(1, HEAD_BANDS), default=1)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(), a.bands), indent=2))


if __name__ == "__main__":
    main()
