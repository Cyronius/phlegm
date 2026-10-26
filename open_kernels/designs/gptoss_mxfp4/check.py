"""Check the standalone MXFP4 chunk decoder on NPU2 against the Python oracle."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from mxfp4 import decode_chunk  # noqa: E402


def cases() -> list[tuple[str, bytes]]:
    rng = np.random.default_rng(20261026)
    rows = []
    for i in range(3):
        chunk = rng.integers(0, 256, 2560, dtype=np.uint8)
        chunk[:128] = rng.integers(120, 137, 128, dtype=np.uint8)
        rows.append((f"random-{i}", chunk.tobytes()))
    # Every nibble and both row parities occur across all four scale groups.
    chunk = np.zeros(2560, np.uint8)
    chunk[:128] = np.resize(np.array([0, 1, 2, 120, 127, 128, 129, 136, 252], np.uint8), 128)
    chunk[512:] = np.resize(np.array([0x10, 0x32, 0x54, 0x76,
                                      0x98, 0xba, 0xdc, 0xfe], np.uint8), 2048)
    rows.append(("all-codes-and-scale-edges", chunk.tobytes()))
    return rows


def run(driver: Path, build: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, chunk in cases():
        d = out / name
        d.mkdir(exist_ok=True)
        (d / "chunk.bin").write_bytes(chunk)
        config = d / "run.cfg"
        config.write_text("\n".join([
            "device", f"xclbin G {build / 'final.xclbin'}",
            f"kernelx k G {build / 'insts.bin'}",
            f"buf chunk 2560 {d / 'chunk.bin'}", "buf values 16384",
            "run k chunk values", f"dump values {d / 'values.bin'} 16384", "",
        ]))
        completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"{name}: harness failed: {completed.stdout}\n{completed.stderr}")
        actual = np.fromfile(d / "values.bin", dtype=np.float32)
        expected = decode_chunk(chunk).ravel()
        if actual.shape != expected.shape:
            raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise AssertionError(f"{name}: nonfinite output in edge fixture")
        max_error = float(np.max(np.abs(actual - expected)))
        passed = np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
        results.append({"case": name, "passed": bool(passed), "max_absolute_error": max_error})
        if not passed:
            raise AssertionError(results[-1])
    report = {"passed": True, "comparison": "bit-exact fp32 MXFP4 chunk decode",
              "cases": results}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
