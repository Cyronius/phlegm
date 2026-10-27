"""Check the full Gemma 4 vocabulary softcap on NPU against FP64 tanh."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

VOCAB = 262144


def run(driver: Path, build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261027)
    x = rng.normal(0, 35, VOCAB).astype(np.float32)
    x[:12] = np.array([0, -0.0, 1e-5, -1e-5, 1, -1, 30, -30,
                        300, -300, 1000, -1000], np.float32)
    expected = (30 * np.tanh(x.astype(np.float64) / 30)).astype(np.float32)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logits.bin").write_bytes(x.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf logits {x.nbytes} {out / 'logits.bin'}",
        f"buf capped {x.nbytes}", "run k logits capped",
        f"dump capped {out / 'capped.bin'} {x.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "capped.bin", np.float32)
    if actual.size != VOCAB or not np.isfinite(actual).all():
        raise AssertionError("invalid NPU output")
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    cosine = float(np.dot(actual.astype(np.float64), expected.astype(np.float64)) /
                   (np.linalg.norm(actual.astype(np.float64)) * np.linalg.norm(expected.astype(np.float64))))
    report = {"vocab": VOCAB, "max_absolute_error": float(diff.max()),
              "cosine": cosine,
              "edge_errors": [float(v) for v in diff[:12]],
              "passed": bool(diff.max() < 0.0002 and cosine > 0.999999999)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
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
