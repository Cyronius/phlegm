"""Run a GPT-OSS source top-four MoE and residual through NPU images."""
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

HIDDEN = 2880
PADDED = 2944


def run(driver: Path, kernels: Path, source: GptOssSource, layer: int,
        activation: np.ndarray, residual: np.ndarray, out: Path) -> dict:
    source._prefix(layer)
    x = np.asarray(activation)
    skip = np.asarray(residual)
    if x.shape != (HIDDEN,) or x.dtype != bfloat16 or not np.isfinite(x.astype(np.float32)).all():
        raise ValueError("GPT-OSS MoE activation must be finite BF16[2880]")
    if skip.shape != (HIDDEN,) or skip.dtype != np.float32 or not np.isfinite(skip).all():
        raise ValueError("GPT-OSS MoE residual must be finite FP32[2880]")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    builds = {name: kernels / name for name in (
        "router_top4", "expert_projection_90_bands", "expert_gate_45_blocks",
        "expert_mix_top4", "residual_bias")}
    for build in builds.values():
        for name in ("final.xclbin", "insts.bin"):
            if not (build / name).is_file():
                raise FileNotFoundError(build / name)
    router_x = source.router_input(layer, x)
    router_w = source.router_weight(layer)
    if router_w.shape != (2880, 32) or router_w.dtype != bfloat16:
        raise ValueError("GPT-OSS router weight must be BF16[2880,32]")
    out.mkdir(parents=True, exist_ok=False)
    route = invoke(driver, builds["router_top4"], out / "router",
                   [("x_bias", router_x), ("w", router_w)], np.dtype(np.uint8), 256)
    ids = np.frombuffer(route, np.int32, count=4).copy()
    weights = np.frombuffer(route, np.float32, count=4, offset=16).copy()
    if (np.any(ids < 0) or np.any(ids >= 32) or np.unique(ids).size != 4 or
            np.any(weights < 0) or abs(float(weights.sum()) - 1) > 0.002):
        raise AssertionError("invalid GPT-OSS NPU top-four route")
    padded = np.zeros(PADDED, bfloat16)
    padded[:HIDDEN] = x
    down_outputs = []
    for expert_id in ids:
        stage = out / f"expert-{int(expert_id):02d}"
        gate = invoke(driver, builds["expert_projection_90_bands"], stage / "gate",
                      [("x", padded), ("chunks", source.expert_input(layer, int(expert_id), "gate"))],
                      np.dtype(np.float32), HIDDEN)
        up = invoke(driver, builds["expert_projection_90_bands"], stage / "up",
                    [("x", padded), ("chunks", source.expert_input(layer, int(expert_id), "up"))],
                    np.dtype(np.float32), HIDDEN)
        hidden = invoke(driver, builds["expert_gate_45_blocks"], stage / "activation",
                        [("gate", gate), ("up", up)], np.dtype(bfloat16), HIDDEN)
        down_x = np.zeros(PADDED, bfloat16)
        down_x[:HIDDEN] = hidden
        down = invoke(driver, builds["expert_projection_90_bands"], stage / "down",
                      [("x", down_x), ("chunks", source.expert_input(layer, int(expert_id), "down"))],
                      np.dtype(np.float32), HIDDEN)
        down_outputs.append(down)
    packed = np.ascontiguousarray(np.stack(down_outputs).reshape(4, 45, 64).transpose(1, 0, 2))
    mixed = invoke(driver, builds["expert_mix_top4"], out / "mix",
                   [("expert", packed), ("route", route)], np.dtype(np.float32), HIDDEN)
    pair = np.ascontiguousarray(np.stack((skip.reshape(45, 64),
                                          mixed.reshape(45, 64)), axis=1))
    final = invoke(driver, builds["residual_bias"], out / "residual",
                   [("pair", pair), ("bias", np.zeros(HIDDEN, bfloat16))],
                   np.dtype(np.float32), HIDDEN)
    (out / "output.f32").write_bytes(final.tobytes())
    report = {"layer": layer, "expert_ids": ids.tolist(),
              "router_calls": 1, "expert_projection_calls": 12,
              "gate_calls": 4, "mix_calls": 1, "residual_calls": 1,
              "kernel_calls": 19, "passed": True}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "kernels", "model", "activation", "residual", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--layer", required=True, type=int)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        validate_header(reader.tensors)
        print(json.dumps(run(a.driver.resolve(), a.kernels.resolve(), GptOssSource(reader),
                             a.layer, np.fromfile(a.activation, bfloat16),
                             np.fromfile(a.residual, np.float32), a.out.resolve()), indent=2))
    finally:
        reader.mm.close()
        reader.f.close()
