"""NPU check of both Gemma 4 audio 3x3 stride-2 subsampling convolutions."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_kernels.gemma4_audio_subsample import pack_weights, pack_window, unpack  # noqa: E402


def run(driver: Path, build: Path, out: Path, cin: int, cout: int) -> dict:
    rng = np.random.default_rng(20261027 + cin)
    x = rng.normal(0, 0.2, (32, 32, cin)).astype(np.float32).astype(bfloat16)
    w = rng.normal(0, 0.02, (cout, cin, 3, 3)).astype(np.float32).astype(bfloat16)
    a, shape = pack_window(x)
    b = pack_weights(w)
    out.mkdir(parents=True, exist_ok=True)
    (out / "activation.bin").write_bytes(a.tobytes())
    (out / "weight.bin").write_bytes(b.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx conv G {build / 'insts.bin'}",
        f"buf activation {a.nbytes} {out / 'activation.bin'}",
        f"buf weight {b.nbytes} {out / 'weight.bin'}",
        f"buf result {256 * 256 * 4}", "run conv activation weight result",
        f"dump result {out / 'result.bin'} {256 * 256 * 4}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = unpack(np.fromfile(out / "result.bin", np.float32), shape, cout)
    padded = np.pad(x.astype(np.float64), ((1, 1), (1, 1), (0, 0)))
    w64 = w.astype(np.float64)
    expected = np.empty((16, 16, cout), np.float64)
    for y in range(16):
        for col in range(16):
            patch = padded[2*y:2*y+3, 2*col:2*col+3].transpose(2, 0, 1)
            expected[y, col] = np.sum(w64 * patch[None], axis=(1, 2, 3))
    error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
    relative = error / max(float(np.max(np.abs(expected))), 1e-20)
    report = dict(cin=cin, cout=cout, input_shape=list(x.shape),
                  output_shape=list(actual.shape), max_absolute_error=error,
                  max_relative_error=relative,
                  passed=bool(np.isfinite(actual).all() and relative < 0.005))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cin", type=int, required=True)
    parser.add_argument("--cout", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve(),
                         args.cin, args.cout), indent=2))
