"""Compare the GPT-OSS q4_1 Q/K/V NPU projection with decoded source chunks."""
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
from q4nx import dq_chunks_q4_1  # noqa: E402
from pack import CHUNK, K_BLOCKS, ROLE_ROWS, projection  # noqa: E402


def fixture(role: str, start_band: int, bands: int,
            activation: np.ndarray | None = None):
    rng = np.random.default_rng(20261027 + start_band + {"q": 0, "k": 1, "v": 2}[role])
    total = ROLE_ROWS[role] // 32
    source = rng.integers(0, 256, (total * K_BLOCKS, CHUNK), dtype=np.uint8)
    scales = rng.uniform(0.003, 0.025, (len(source), 128)).astype(np.float32).astype(bfloat16)
    mins = rng.uniform(-0.15, -0.02, (len(source), 128)).astype(np.float32).astype(bfloat16)
    source[:, :256] = scales.view(np.uint8).reshape(-1, 256)
    source[:, 256:512] = mins.view(np.uint8).reshape(-1, 256)
    selected = projection(source, role, start_band, bands)
    oracle_index = np.array([((band // 4) * K_BLOCKS + kb) * 4 + band % 4
                             for band in range(start_band, start_band + bands)
                             for kb in range(K_BLOCKS)])
    oracle_chunks = source[oracle_index].reshape(bands, K_BLOCKS, CHUNK)
    if not np.array_equal(selected, oracle_chunks):
        raise AssertionError("container supertile selection differs from the source raster")
    if activation is None:
        x = rng.normal(0, 0.5, 2880).astype(np.float32).astype(bfloat16)
    else:
        x = np.asarray(activation)
        if x.shape != (2880,) or x.dtype != bfloat16:
            raise ValueError("shared QKV activation must be BF16[2880]")
    bias = rng.normal(0, 0.2, (bands, 32)).astype(np.float32).astype(bfloat16)
    x_bias = np.concatenate((x, np.zeros(64, bfloat16), bias.ravel()))

    # Pair adjacent half-width chunks into the packer's ordinary 5120-byte
    # q4_1 layout; the final high half is the container's absent 128 columns.
    from recipes.pack import fuse_chunks
    padded = np.zeros((bands, 24, CHUNK), np.uint8)
    padded[:, :K_BLOCKS] = oracle_chunks
    full = fuse_chunks(padded[:, 0::2].reshape(-1, CHUNK),
                       padded[:, 1::2].reshape(-1, CHUNK))
    weights = dq_chunks_q4_1(full).reshape(bands, 12, 32, 256)
    weights = weights.transpose(0, 2, 1, 3).reshape(bands * 32, 3072)[:, :2880]
    expected = weights.astype(np.float64) @ x.astype(np.float64)
    expected += bias.ravel().astype(np.float64)
    return x_bias, selected, expected.astype(np.float32)


def run(driver: Path, build: Path, out: Path, role: str, start_band: int, bands: int,
        activation: np.ndarray | None = None) -> dict:
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    x_bias, chunks, expected = fixture(role, start_band, bands, activation)
    out.mkdir(parents=True, exist_ok=True)
    (out / "x_bias.bin").write_bytes(x_bias.tobytes())
    (out / "chunks.bin").write_bytes(chunks.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf x_bias {x_bias.nbytes} {out / 'x_bias.bin'}",
        f"buf chunks {chunks.nbytes} {out / 'chunks.bin'}",
        f"buf result {expected.nbytes}", "run k x_bias chunks result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"NPU harness failed: {completed.stdout}\n{completed.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("invalid NPU output")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    max_abs = float(np.max(np.abs(a - e)))
    max_rel = float(max_abs / max(np.max(np.abs(e)), 1e-30))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    report = {"passed": bool(max_rel < 0.001 and cosine > 0.99999),
              "role": role, "start_band": start_band, "bands": bands,
              "max_absolute_error": max_abs, "max_relative_error": max_rel,
              "cosine": cosine,
              "reference": "q4nx decoded 2560-byte source chunks with BF16 bias"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--role", required=True, choices=ROLE_ROWS)
    p.add_argument("--start-band", type=int, default=0)
    p.add_argument("--bands", type=int, choices=(1, 16), default=16)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.role, a.start_band, a.bands), indent=2))


if __name__ == "__main__":
    main()
