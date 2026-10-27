"""Check the 12B q8 LM head arithmetic on one 128-logit band."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_kernels.model.q4nx import dq_chunks_q8  # noqa: E402


def fixture():
    rng = np.random.default_rng(20261027)
    k, rows = 3840, 128
    chunks = np.empty((k // 256, 4, 8704), np.uint8)
    scales = rng.uniform(0.001, 0.014, (k // 256, 4, 256)).astype(np.float32).astype(bfloat16)
    chunks[..., :512] = scales.view(np.uint8).reshape(k // 256, 4, 512)
    chunks[..., 512:] = rng.integers(-127, 128, (k // 256, 4, 8192),
                                     dtype=np.int8).view(np.uint8)
    x = rng.normal(0, 0.5, k).astype(np.float32).astype(bfloat16)
    w = np.empty((rows, k), np.float32)
    for kt in range(k // 256):
        decoded = dq_chunks_q8(np.ascontiguousarray(chunks[kt])).reshape(4, 32, 256)
        w[:, kt * 256:(kt + 1) * 256] = decoded.reshape(rows, 256)
    expected = (w.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    return chunks.reshape(-1), x, expected


def run(driver: Path, build: Path, out: Path) -> dict:
    weight, activation, expected = fixture()
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(weight.tobytes())
    (out / "activation.bin").write_bytes(activation.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf weights {weight.nbytes} {out / 'weights.bin'}",
        f"buf activation {activation.nbytes} {out / 'activation.bin'}",
        f"buf result {expected.nbytes}", "run k weights activation result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("invalid q8 head output")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    maximum = float(np.max(np.abs(a - e)))
    relative = maximum / max(float(np.max(np.abs(e))), 1e-20)
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    report = {"rows": 128, "cols": 3840, "max_absolute_error": maximum,
              "max_relative_error": relative, "cosine": cosine,
              "passed": bool(relative < 0.001 and cosine > 0.999999),
              "reference": "FP64 matvec of independently decoded synthetic q8 chunks"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve()), indent=2))
