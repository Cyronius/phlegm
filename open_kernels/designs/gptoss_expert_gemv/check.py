"""Check GPT-OSS MXFP4 expert projection against independently decoded FP64 GEMV."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from mxfp4 import chunk_bias, decode_chunk  # noqa: E402

CHUNK = 2560
KBLOCKS = 23
REAL_CHUNKS = ROOT.parent / "specs/open-engine/tests/fixtures/gptoss_mxfp4_chunks.npz"


def fixture(bands: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(KBLOCKS * 128, dtype=bfloat16)
    x[:2880] = rng.normal(0, 0.5, 2880).astype(np.float32).astype(bfloat16)
    chunks = rng.integers(0, 256, (bands, KBLOCKS, CHUNK), dtype=np.uint8)
    chunks[:, :, :128] = rng.integers(121, 129, (bands, KBLOCKS, 128), dtype=np.uint8)
    chunks[:, :, 128:512] = 0
    bias = rng.normal(0, 0.2, (bands, 32)).astype(np.float32).astype(bfloat16)
    chunks[:, 0, 128:192] = bias.view(np.uint8).reshape(bands, 64)
    # Real bytes from the pinned GPT-OSS container exercise its scale and
    # nibble distributions at positions spread across the full DMA stream.
    with np.load(REAL_CHUNKS) as captured:
        real = captured["chunks"]
    positions = np.linspace(0, bands * KBLOCKS - 1, len(real), dtype=np.int64)
    chunks.reshape(-1, CHUNK)[positions] = real
    return x, chunks


def reference(x: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    out = np.empty((len(chunks), 32), np.float32)
    x64 = x.astype(np.float64).reshape(KBLOCKS, 128)
    for band in range(len(chunks)):
        acc = chunk_bias(chunks[band, 0]).astype(np.float64)
        for kb in range(KBLOCKS):
            acc += decode_chunk(chunks[band, kb]).astype(np.float64) @ x64[kb]
        out[band] = acc.astype(np.float32)
    return out


def run(driver: Path, build: Path, out: Path, bands: int, seed: int) -> dict:
    if bands not in (1, 90):
        raise ValueError("bands must match a 1 or 90 band build")
    out.mkdir(parents=True, exist_ok=True)
    x, chunks = fixture(bands, seed)
    expected = reference(x, chunks)
    (out / "x.bin").write_bytes(x.tobytes())
    (out / "chunks.bin").write_bytes(chunks.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf x {x.nbytes} {out / 'x.bin'}",
        f"buf chunks {chunks.nbytes} {out / 'chunks.bin'}",
        f"buf result {expected.nbytes}", "run k x chunks result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    completed = subprocess.run([str(driver), str(cfg)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"harness failed: {completed.stdout}\n{completed.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32)
    if actual.size != expected.size or not np.isfinite(actual).all():
        raise AssertionError("invalid NPU output size or nonfinite value")
    err = np.abs(actual.astype(np.float64) - expected.ravel().astype(np.float64))
    max_abs = float(err.max())
    max_rel = max_abs / max(float(np.max(np.abs(expected))), 1e-30)
    cosine = float(actual.astype(np.float64) @ expected.ravel().astype(np.float64) /
                   (np.linalg.norm(actual.astype(np.float64)) *
                    np.linalg.norm(expected.astype(np.float64)) + 1e-30))
    passed = max_rel < 0.005 and cosine > 0.99999
    report = {"passed": passed, "bands": bands, "max_absolute_error": max_abs,
              "max_relative_error": max_rel, "cosine": cosine,
              "real_chunk_fixture_sha256": hashlib.sha256(REAL_CHUNKS.read_bytes()).hexdigest(),
              "real_chunks": 8,
              "reference": "FP64 sum of independently decoded MXFP4 chunks plus BF16 bias"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bands", required=True, type=int, choices=[1, 90])
    p.add_argument("--seed", type=int, default=20261027)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.bands, a.seed), indent=2))


if __name__ == "__main__":
    main()
