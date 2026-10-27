"""Check the GPT-OSS NPU mixer using the compiled biased router's actual bytes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gptoss_router.check import cases, validate  # noqa: E402


def invoke(driver: Path, build: Path, out: Path, inputs: list[np.ndarray],
           output_bytes: int) -> bytes:
    out.mkdir(parents=True, exist_ok=True)
    lines = ["device", f"xclbin G {build / 'final.xclbin'}",
             f"kernelx k G {build / 'insts.bin'}"]
    names = []
    for index, array in enumerate(inputs):
        name = f"input{index}"
        path = out / f"{name}.bin"
        path.write_bytes(array.tobytes())
        lines.append(f"buf {name} {array.nbytes} {path}")
        names.append(name)
    lines += [f"buf result {output_bytes}", f"run k {' '.join(names)} result",
              f"dump result {out / 'result.bin'} {output_bytes}", ""]
    config = out / "run.cfg"
    config.write_text("\n".join(lines))
    completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{out.name}: harness failed: {completed.stdout}\n{completed.stderr}")
    raw = (out / "result.bin").read_bytes()
    if len(raw) != output_bytes:
        raise AssertionError(f"{out.name}: wrong result size")
    return raw


def run(driver: Path, router_build: Path, mixer_build: Path, out: Path) -> dict:
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for build in (router_build, mixer_build):
        for name in ("final.xclbin", "insts.bin"):
            if not (build / name).is_file():
                raise FileNotFoundError(build / name)
    rng = np.random.default_rng(20261027)
    report_cases = []
    for name, x, router_matrix, bias in cases():
        d = out / name
        x_bias = np.concatenate([x, bias])
        route_bytes = invoke(driver, router_build, d / "router",
                             [x_bias, router_matrix], 256)
        ids = np.frombuffer(route_bytes, dtype=np.int32, count=4).copy()
        weights = np.frombuffer(route_bytes, dtype=np.float32, count=4, offset=16).copy()
        logits = np.frombuffer(route_bytes, dtype=np.float32, count=32, offset=32).copy()
        expected_logits = x.astype(np.float64) @ router_matrix.astype(np.float64)
        expected_logits += bias.astype(np.float64)
        validate(ids, weights, logits, expected_logits)
        expert_bank = rng.normal(0, 1, (32, 2880)).astype(np.float32)
        expert_bank += (np.arange(32, dtype=np.float32) * 0.03)[:, None]
        selected = expert_bank[ids]
        packed = np.ascontiguousarray(selected.reshape(4, 45, 64).transpose(1, 0, 2))
        route = np.frombuffer(route_bytes, dtype=np.uint8).copy()
        raw = invoke(driver, mixer_build, d / "mixer", [packed, route], 2880 * 4)
        actual = np.frombuffer(raw, dtype=np.float32)
        expected = weights.astype(np.float64) @ selected.astype(np.float64)
        if not np.isfinite(actual).all():
            raise AssertionError(f"{name}: nonfinite mixer output")
        error = np.abs(actual.astype(np.float64) - expected)
        max_abs = float(error.max())
        max_rel = float(max_abs / max(np.max(np.abs(expected)), 1e-30))
        cosine = float(actual.astype(np.float64) @ expected /
                       (np.linalg.norm(actual.astype(np.float64)) *
                        np.linalg.norm(expected) + 1e-30))
        if max_rel >= 1e-4 or cosine <= 0.999999:
            raise AssertionError(f"{name}: mixer error {max_rel}, cosine {cosine}")
        report_cases.append({"case": name, "ids": ids.tolist(),
                             "max_absolute_error": max_abs,
                             "max_relative_error": max_rel, "cosine": cosine})
    out.mkdir(parents=True, exist_ok=True)
    report = {"passed": True, "reference": "FP64 weighted sum of four router-selected expert vectors",
              "cases": report_cases}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--router-build", required=True, type=Path)
    p.add_argument("--mixer-build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.router_build.resolve(),
                         a.mixer_build.resolve(), a.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
