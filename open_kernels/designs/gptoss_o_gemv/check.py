"""Check GPT-OSS 4096-to-3072 padded attention output projection on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from q4nx import dq_chunks_q4_1  # noqa: E402
from designs.gptoss_o_gemv.pack import BANDS, CHUNK, K_BLOCKS, projection  # noqa: E402
from recipes.pack import fuse_chunks  # noqa: E402


def fixture(start_band: int, activation: np.ndarray | None = None):
    rng = np.random.default_rng(20261028 + start_band)
    source = rng.integers(0, 256, (BANDS * K_BLOCKS, CHUNK), dtype=np.uint8)
    scales = rng.uniform(0.003, 0.025, (len(source), 128)).astype(np.float32).astype(bfloat16)
    mins = rng.uniform(-0.15, -0.02, (len(source), 128)).astype(np.float32).astype(bfloat16)
    source[:, :256] = scales.view(np.uint8).reshape(-1, 256)
    source[:, 256:512] = mins.view(np.uint8).reshape(-1, 256)
    selected = projection(source, start_band)
    oracle_index = np.array([((band // 4) * K_BLOCKS + kb) * 4 + band % 4
                             for band in range(start_band, start_band + 16)
                             for kb in range(K_BLOCKS)])
    oracle = source[oracle_index].reshape(16, K_BLOCKS, CHUNK)
    if not np.array_equal(selected, oracle):
        raise AssertionError("o_proj pack differs from the container supertile raster")
    if activation is None:
        x = rng.normal(0, 0.5, 4096).astype(np.float32).astype(bfloat16)
    else:
        x = np.asarray(activation)
        if x.shape != (4096,) or x.dtype != bfloat16:
            raise ValueError("attention output must be BF16[4096]")
    full = fuse_chunks(oracle[:, 0::2].reshape(-1, CHUNK),
                       oracle[:, 1::2].reshape(-1, CHUNK))
    weights = dq_chunks_q4_1(full).reshape(16, 16, 32, 256)
    weights = weights.transpose(0, 2, 1, 3).reshape(512, 4096)
    expected = weights.astype(np.float64) @ x.astype(np.float64)
    return x, selected, expected.astype(np.float32)


def run(driver: Path, build: Path, out: Path, start_band: int,
        activation: np.ndarray | None = None) -> dict:
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    x, chunks, expected = fixture(start_band, activation)
    out.mkdir(parents=True, exist_ok=True)
    (out / "x.bin").write_bytes(x.tobytes())
    (out / "chunks.bin").write_bytes(chunks.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf x {x.nbytes} {out / 'x.bin'}",
        f"buf chunks {chunks.nbytes} {out / 'chunks.bin'}",
        f"buf result {expected.nbytes}", "run k x chunks result",
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
              "start_band": start_band, "bands": 16, "real_rows": 2880,
              "max_absolute_error": max_abs, "max_relative_error": max_rel,
              "cosine": cosine,
              "reference": "q4nx decoded 2560-byte o_proj chunks"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start-band", required=True, type=int)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.start_band), indent=2))


if __name__ == "__main__":
    main()
