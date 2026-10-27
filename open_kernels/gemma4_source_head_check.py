"""Run one 12B signed-source q8 vocabulary band through its NPU head."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_text_plan import from_hf_config
from open_kernels.gemma4_text_source import Gemma4TextSource


class _Reader:
    def __init__(self, shape, payload):
        self.tensors = {"lm_head.weight": {"dtype": "I8", "shape": shape,
                                           "data_offsets": [0, len(payload)]}}
        self.data_base = 0
        self.mm = payload


def run(driver: Path, image: Path, config: dict, out: Path) -> dict:
    plan = from_hf_config(config)
    if plan.variant != "12B":
        raise ValueError("expected the Gemma 4 12B text configuration")
    for path in (driver, image / "final.xclbin", image / "insts.bin"):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows, cols = 128, plan.hidden
    rng = np.random.default_rng(20261027)
    scales = rng.uniform(0.001, 0.014, (rows // 16, cols // 256, 8, 16)) \
        .astype(np.float32).astype(bfloat16)
    codes = rng.integers(-127, 128, (rows // 16, cols // 256, 256, 16), dtype=np.int8)
    source = np.empty((rows // 16, cols // 256, 4352), np.uint8)
    source[..., :256] = scales.view(np.uint8).reshape(rows // 16, cols // 256, 256)
    source[..., 256:] = codes.view(np.uint8).reshape(rows // 16, cols // 256, 4096)
    x = rng.normal(0, 0.5, cols).astype(np.float32).astype(bfloat16)
    expected = np.zeros(rows, np.float64)
    for rb in range(rows // 16):
        for kt in range(cols // 256):
            x_tile = x[kt * 256:(kt + 1) * 256].astype(np.float64)
            for row in range(16):
                weight = codes[rb, kt, :, row].astype(np.float64) * \
                    scales[rb, kt, :, row].astype(np.float64).repeat(32)
                expected[rb * 16 + row] += weight @ x_tile
    expected = expected.astype(np.float32)
    compact = replace(plan, vocab=rows)
    source_reader = _Reader(source.shape, source.tobytes())
    packed = Gemma4TextSource(source_reader, compact).head_band(0)
    out.mkdir(parents=True, exist_ok=False)
    (out / "weights.bin").write_bytes(packed.tobytes())
    (out / "activation.bin").write_bytes(x.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {image / 'final.xclbin'}",
        f"kernelx k G {image / 'insts.bin'}",
        f"buf weights {packed.nbytes} {out / 'weights.bin'}",
        f"buf activation {x.nbytes} {out / 'activation.bin'}",
        f"buf result {expected.nbytes}", "run k weights activation result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU q8 head failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("invalid Gemma 4 NPU head output")
    maximum = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    relative = maximum / max(float(np.max(np.abs(expected))), 1e-20)
    report = {"variant": plan.variant, "logits": rows, "hidden": cols,
              "max_absolute_error": maximum, "max_relative_error": relative,
              "passed": bool(relative < 0.001),
              "reference": "FP64 matvec from independently decoded 16-row q8 source tiles"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.image.resolve(),
                         json.loads(a.config.read_text()), a.out.resolve()), indent=2))
