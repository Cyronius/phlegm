"""NPU check for Gemma 4 audio softcapped relative local attention."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from open_kernels.gemma4_audio_attention_pack import pack, reference  # noqa: E402


def run(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261027)
    q = rng.normal(0, 0.2, (8, 12, 128)).astype(np.float32)
    k = rng.normal(0, 0.3, (8, 24, 128)).astype(np.float32)
    v = rng.normal(0, 0.4, (8, 24, 128)).astype(np.float32)
    rel_key = rng.normal(0, 0.3, (8, 13, 128)).astype(np.float32)
    mask = np.zeros((8, 12, 24), np.uint8)
    mask[:, :, :3] = 1
    mask[:, 0, 4:7] = 1
    data = pack(q, k, v, rel_key, mask)
    expected = reference(q, k, v, rel_key, mask)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.bin").write_bytes(data.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf input {data.nbytes} {out / 'input.bin'}",
        f"buf result {expected.nbytes}", "run k input result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32).reshape(expected.shape)
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    relative = error / max(float(np.max(np.abs(expected))), 1e-20)
    report = dict(heads=8, queries=12, context=24, head_dim=128,
                  max_absolute_error=error, max_relative_error=relative,
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
    args = parser.parse_args()
    print(json.dumps(run(args.driver.resolve(), args.build.resolve(), args.out.resolve()), indent=2))
