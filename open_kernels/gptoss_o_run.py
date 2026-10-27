"""Run GPT-OSS attention output projection, bias, and residual from q4nx on NPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from designs.gptoss_attention.check_chain import invoke  # noqa: E402
from model.q4nx import Q4NX  # noqa: E402
from recipes.gptoss_source import GptOssSource, validate_header  # noqa: E402

STARTS = tuple(range(0, 96, 16))


def run(driver: Path, o_build: Path, residual_build: Path, source: GptOssSource,
        layer: int, activation: np.ndarray, residual: np.ndarray, out: Path) -> dict:
    source._prefix(layer)
    x = np.asarray(activation)
    skip = np.asarray(residual)
    if x.shape != (4096,) or x.dtype != bfloat16 or not np.isfinite(x.astype(np.float32)).all():
        raise ValueError("GPT-OSS output projection input must be finite BF16[4096]")
    if skip.shape != (2880,) or skip.dtype != np.float32 or not np.isfinite(skip).all():
        raise ValueError("GPT-OSS residual input must be finite FP32[2880]")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for build in (o_build, residual_build):
        for name in ("final.xclbin", "insts.bin"):
            if not (build / name).is_file():
                raise FileNotFoundError(build / name)
    bias = source.constant(layer, "o_bias")
    out.mkdir(parents=True, exist_ok=False)
    pieces = []
    for start in STARTS:
        chunks = source.attention(layer, "o", start)
        pieces.append(invoke(driver, o_build, out / f"o-{start:03d}",
                             [("x", x), ("chunks", chunks)], np.dtype(np.float32), 512))
    projected = np.concatenate(pieces)[:2880]
    pair = np.ascontiguousarray(np.stack((skip.reshape(45, 64),
                                          projected.reshape(45, 64)), axis=1))
    result = invoke(driver, residual_build, out / "residual",
                    [("pair", pair), ("bias", bias)], np.dtype(np.float32), 2880)
    (out / "output.f32").write_bytes(result.tobytes())
    report = {"layer": layer, "projection_calls": len(STARTS),
              "residual_calls": 1, "output_channels": result.size, "passed": True}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "o-build", "residual-build", "model", "activation",
                 "residual", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--layer", required=True, type=int)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        validate_header(reader.tensors)
        x = np.fromfile(a.activation, bfloat16)
        skip = np.fromfile(a.residual, np.float32)
        print(json.dumps(run(a.driver.resolve(), a.o_build.resolve(),
                             a.residual_build.resolve(), GptOssSource(reader),
                             a.layer, x, skip, a.out.resolve()), indent=2))
    finally:
        reader.mm.close()
        reader.f.close()
