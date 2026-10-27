"""NPU check for learned Gemma 4 audio Q and fixed K scaling."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from open_kernels.gemma4_audio_qk_scale_pack import pack, reference  # noqa: E402


def run(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261027)
    q = rng.normal(0, 0.5, (8, 12, 128)).astype(np.float32).astype(bfloat16)
    k = rng.normal(0, 0.7, (8, 24, 128)).astype(np.float32).astype(bfloat16)
    learned = rng.normal(0, 0.6, 128).astype(np.float32).astype(bfloat16)
    packet = pack(q, k, learned)
    eq, ek = reference(q, k, learned)
    expected = np.concatenate((eq, ek), axis=1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "packet.bin").write_bytes(packet.tobytes())
    (out / "learned.bin").write_bytes(learned.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf packet {packet.nbytes} {out / 'packet.bin'}",
        f"buf learned {learned.nbytes} {out / 'learned.bin'}",
        f"buf result {expected.nbytes}", "run k packet learned result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32).reshape(expected.shape)
    error = float(np.max(np.abs(actual - expected)))
    relative = error / max(float(np.max(np.abs(expected))), 1e-20)
    report = {"heads": 8, "queries": 12, "context": 24, "head_dim": 128,
              "max_absolute_error": error, "max_relative_error": relative,
              "passed": bool(np.isfinite(actual).all() and error == 0)}
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
