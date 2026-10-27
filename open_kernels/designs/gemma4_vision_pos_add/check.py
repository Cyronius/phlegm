"""NPU check for Gemma 4 vision projected-plus-position BF16 addition."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from open_kernels.gemma4_vision_position_pack import pack_positions, reference  # noqa: E402


def run(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261027)
    projected = rng.normal(0, 0.4, (256, 768)).astype(np.float32).astype(bfloat16)
    table = rng.normal(0, 0.2, (2, 10240, 768)).astype(np.float32).astype(bfloat16)
    positions = np.array([(row % 42, row // 42) for row in range(256)], np.int32)
    positions[-40:] = -1
    projected[-40:] = bfloat16(0)
    x, y = pack_positions(table, positions)
    xy = np.stack((x, y), axis=1)
    expected = reference(projected, x, y)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("projected", projected), ("xy", xy)):
        (out / f"{name}.bin").write_bytes(data.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf projected {projected.nbytes} {out / 'projected.bin'}",
        f"buf xy {xy.nbytes} {out / 'xy.bin'}",
        f"buf result {expected.nbytes}", "run k projected xy result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", bfloat16).reshape(expected.shape)
    error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
    report = {"rows": 256, "hidden": 768, "valid_rows": 216,
              "max_absolute_error": error,
              "exact_bf16_fraction": float(np.mean(actual == expected)),
              "passed": bool(np.isfinite(actual.astype(np.float32)).all()
                             and error == 0 and np.all(actual[-40:] == 0))}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve()), indent=2))
