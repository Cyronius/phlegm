"""Check a Gemma 4 Q/K rotation image against half-split RoPE arithmetic."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

GEOMETRIES = {
    "e2b-local": (9, 256), "e2b-full": (9, 512),
    "e4b-local": (10, 256), "e4b-full": (10, 512),
    "12b-local": (24, 256), "12b-full": (17, 512),
}


def table_for(position: int, head_dim: int) -> np.ndarray:
    if position < 0:
        raise ValueError("position must be nonnegative")
    half = head_dim // 2
    inv = np.zeros(half, np.float64)
    if head_dim == 256:
        inv[:] = 10000.0 ** (-2 * np.arange(half, dtype=np.float64) / head_dim)
    elif head_dim == 512:
        # proportional RoPE uses 25% of the 256 frequency pairs.
        count = head_dim // 8
        inv[:count] = 1000000.0 ** (-2 * np.arange(count, dtype=np.float64) / head_dim)
    else:
        raise ValueError(head_dim)
    # Gemma4TextRotaryEmbedding forms angles in FP32 and casts cos/sin to
    # the BF16 query dtype before apply_rotary_pos_emb. The NPU table remains
    # FP32 for arithmetic, carrying those already-rounded BF16 values.
    angle = np.float32(position) * inv.astype(np.float32)
    return np.concatenate((np.cos(angle), np.sin(angle))).astype(bfloat16).astype(np.float32)


def fixture(geometry: str, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heads, width = GEOMETRIES[geometry]
    rng = np.random.default_rng(20261027 + position + heads * width)
    qk = rng.normal(0, 0.7, (heads, width)).astype(np.float32).astype(bfloat16)
    table = table_for(position, width)
    half = width // 2
    x = qk.astype(np.float64)
    t = table.astype(np.float64)
    expected = np.empty_like(x)
    expected[:, :half] = x[:, :half] * t[:half] - x[:, half:] * t[half:]
    expected[:, half:] = x[:, half:] * t[:half] + x[:, :half] * t[half:]
    return qk, table, expected.astype(bfloat16)


def run(driver: Path, build: Path, out: Path, geometry: str, position: int) -> dict:
    qk, table, expected = fixture(geometry, position)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qk.bin").write_bytes(qk.tobytes())
    (out / "table.bin").write_bytes(table.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf qk {qk.nbytes} {out / 'qk.bin'}",
        f"buf table {table.nbytes} {out / 'table.bin'}",
        f"buf result {qk.nbytes}", "run k qk table result",
        f"dump result {out / 'result.bin'} {qk.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", dtype=bfloat16)
    if actual.size != expected.size or not np.isfinite(actual.astype(np.float32)).all():
        raise AssertionError("invalid NPU output")
    if GEOMETRIES[geometry][1] == 512:
        # Proportional RoPE rotates only the first 64 half-split pairs.
        unchanged = actual.reshape(qk.shape)[:, 64:256]
        if not np.array_equal(unchanged.view(np.uint16), qk[:, 64:256].view(np.uint16)):
            raise AssertionError("inactive proportional RoPE pairs changed")
        unchanged = actual.reshape(qk.shape)[:, 320:512]
        if not np.array_equal(unchanged.view(np.uint16), qk[:, 320:512].view(np.uint16)):
            raise AssertionError("inactive proportional RoPE pairs changed")
    a, e = actual.astype(np.float32), expected.ravel().astype(np.float32)
    delta = np.abs(a - e)
    cosine = float(np.dot(a.astype(np.float64), e.astype(np.float64)) /
                   (np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(e.astype(np.float64))))
    report = {"geometry": geometry, "position": position,
              "max_absolute_error": float(delta.max()),
              "exact_bf16_fraction": float(np.mean(a == e)), "cosine": cosine,
              "passed": bool(delta.max() <= 0.015 and cosine > 0.99999)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--pos", required=True, type=int)
    a = parser.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(),
                         a.geometry, a.pos), indent=2))


if __name__ == "__main__":
    main()
