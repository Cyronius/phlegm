"""Check the standalone GPT-OSS activation kernel against fp64 arithmetic.

Build the design first. Generated fixtures and reports belong under target/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def reference(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    for label, values in (("gate", gate), ("up", up)):
        if values.ndim != 1 or values.size not in (64, 2880) or not np.isfinite(values).all():
            raise AssertionError(f"{label}: expected 64 or 2880 finite input values")
    if gate.shape != up.shape:
        raise AssertionError("gate and up projections must have the same shape")
    g = np.minimum(gate.astype(np.float64), 7.0)
    u = np.clip(up.astype(np.float64), -7.0, 7.0)
    return ((u + 1) * g / (1 + np.exp(-1.702 * g))).astype(np.float32)


def validate(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    for label, values in (("output", actual), ("reference", expected)):
        if values.shape != expected.shape or not np.isfinite(values).all():
            raise AssertionError(f"{label}: expected {expected.size} finite values")
    absolute = float(np.max(np.abs(actual - expected)))
    scaled = float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0)))
    if not np.isfinite(absolute) or not np.isfinite(scaled) or scaled > 0.012:
        raise AssertionError(f"max scaled error {scaled}, max abs {absolute}")
    return absolute, scaled


def cases(blocks: int = 1) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if blocks not in (1, 45):
        raise ValueError("blocks must match a 1 or 45 block build")
    rng = np.random.default_rng(20261026)
    n = 64 * blocks
    random = [(rng.normal(0, 8, n).astype(np.float32),
               rng.normal(0, 8, n).astype(np.float32)) for _ in range(3)]
    edge_g = np.resize(np.array([-100, -88, -7, -1, 0, 0.5, 7, 20], np.float32), n)
    edge_u = np.resize(np.array([-20, -7, -1, 0, 0.25, 1, 7, 20], np.float32), n)
    return [(f"random-{i}", g, u) for i, (g, u) in enumerate(random)] + \
           [("clamp-edges", edge_g, edge_u)]


def run(driver: Path, build: Path, out: Path, blocks: int = 1) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, gate, up in cases(blocks):
        d = out / name
        d.mkdir(exist_ok=True)
        (d / "gate.bin").write_bytes(gate.tobytes())
        (d / "up.bin").write_bytes(up.tobytes())
        cfg = "\n".join([
            "device", f"xclbin G {build / 'final.xclbin'}",
            f"kernelx k G {build / 'insts.bin'}",
            f"buf gate {gate.nbytes} {d / 'gate.bin'}",
            f"buf up {up.nbytes} {d / 'up.bin'}",
            f"buf hidden {128 * blocks}", "run k gate up hidden",
            f"dump hidden {d / 'hidden.bin'} {128 * blocks}", "",
        ])
        config = d / "run.cfg"
        config.write_text(cfg)
        completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"{name}: harness failed: {completed.stdout}\n{completed.stderr}")
        raw = (d / "hidden.bin").read_bytes()
        if len(raw) != 128 * blocks:
            raise AssertionError(f"{name}: expected {128 * blocks} output bytes, got {len(raw)}")
        actual = np.frombuffer(raw, dtype=bfloat16).astype(np.float32)
        expected = reference(gate, up).astype(bfloat16).astype(np.float32)
        absolute, scaled = validate(actual, expected)
        results.append({"case": name, "max_absolute_error": absolute,
                        "max_scaled_error": scaled})
    report = {"passed": True, "blocks": blocks,
              "comparison": "bf16-rounded fp64 GPT-OSS reference",
              "threshold": 0.012, "cases": results}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--blocks", type=int, choices=[1, 45], default=1)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(), a.blocks), indent=2))


if __name__ == "__main__":
    main()
