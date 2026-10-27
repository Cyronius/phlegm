"""Check one Gemma 4 q4 text projection image at its complete output width.

Build `designs/lm_head_q4/lm_head_q4.py` with LMHEAD_N=rows and LMHEAD_K=cols.
The design is a general pool-order q4 GEMV despite its historical name.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from open_kernels.gemma4_lm_head_check import fixture  # noqa: E402


def band_stripe(raw: np.ndarray, full_cols: int, image_cols: int,
                stripe: int) -> np.ndarray:
    """One 64-row q4_1 band, selected by consecutive 256-column tiles."""
    chunks = image_cols // 256
    if (full_cols % image_cols or raw.size != full_cols // 256 * 2 * 5120
            or stripe < 0 or stripe >= full_cols // image_cols):
        raise ValueError("invalid q4 band stripe")
    full = raw.reshape(full_cols // 256, 2, 5120)
    return np.ascontiguousarray(full[stripe * chunks:(stripe + 1) * chunks]).reshape(-1)


def run(driver: Path, build: Path, out: Path, rows: int, cols: int,
        image_cols: int | None = None) -> dict:
    if rows <= 0 or rows % 64 or cols <= 0 or cols % 256:
        raise ValueError("q4 projection requires rows divisible by 64 and cols by 256")
    image_cols = image_cols or cols
    if image_cols > 8192 or image_cols <= 0 or image_cols % 256 or cols % image_cols:
        raise ValueError("image_cols must evenly divide cols, be a 256 multiple, and fit 8192")
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    bands = rows // 64
    first, x, first_expected = fixture(cols, 20261027)
    second, _, second_expected = fixture(cols, 20261028, x)
    expected = np.tile(np.concatenate((first_expected, second_expected)),
                       (bands + 1) // 2)[:rows]
    out.mkdir(parents=True, exist_ok=True)
    partials = []
    for stripe in range(cols // image_cols):
        wd = out / f"weights_{stripe}.bin"
        with wd.open("wb") as stream:
            for band in range(bands):
                stream.write(band_stripe((first, second)[band % 2], cols,
                                         image_cols, stripe).tobytes())
        xd = out / f"activation_{stripe}.bin"
        xs = np.ascontiguousarray(x[stripe * image_cols:(stripe + 1) * image_cols])
        xd.write_bytes(xs.tobytes())
        result = out / f"result_{stripe}.bin"
        cfg = out / f"run_{stripe}.cfg"
        cfg.write_text("\n".join([
            "device", f"xclbin G {build / 'final.xclbin'}",
            f"kernelx k G {build / 'insts.bin'}",
            f"buf weights {wd.stat().st_size} {wd}",
            f"buf activation {xs.nbytes} {xd}",
            f"buf result {expected.nbytes}", "run k weights activation result",
            f"dump result {result} {expected.nbytes}", "",
        ]))
        proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
        partials.append(np.fromfile(result, np.float32).astype(np.float64))
    actual = np.sum(partials, axis=0)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("invalid projection output")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    maximum = float(np.max(np.abs(a - e)))
    relative = maximum / max(float(np.max(np.abs(e))), 1e-20)
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    report = {"rows": rows, "cols": cols, "image_cols": image_cols,
              "npu_calls": len(partials), "max_absolute_error": maximum,
              "max_relative_error": relative, "cosine": cosine,
              "passed": bool(relative < 0.005 and cosine > 0.9999),
              "reference": "FP64 matvec of independently decoded synthetic q4_1 chunks"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--image-cols", type=int,
                   help="K width compiled into the image; wide reductions use several calls")
    args = p.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve(),
                         args.rows, args.cols, args.image_cols), indent=2))
