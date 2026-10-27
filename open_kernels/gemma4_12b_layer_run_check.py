"""Smoke-check local and K=V full Gemma 4 12B layer schedules on NPU.

Zero projection weights make the expected residual and K/V states exact;
this checks image wiring and shape routing, not real-model accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_12b_layer_run import run
from open_kernels.gemma4_text_plan import from_hf_config
from open_kernels.gemma4_text_source import Gemma4TextSource


class ZeroSource(Gemma4TextSource):
    def __init__(self, plan):
        self.plan = plan
        self.roles = []

    def projection_band(self, projection, band, stripe=0, image_cols=None):
        self.roles.append(projection.name)
        width = image_cols or projection.cols
        return np.zeros(width // 256 * 2 * 5120, np.uint8)

    def vector(self, name, width):
        return np.ones(width, bfloat16)


def check(driver: Path, bundle: Path, config: dict, out: Path) -> dict:
    plan = from_hf_config(config)
    if plan.variant != "12B":
        raise ValueError("expected Gemma 4 12B config")
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261027)
    hidden = rng.normal(0, 0.5, plan.hidden).astype(np.float32).astype(bfloat16)
    results = []
    for layer_index, position in ((0, 0), (5, 0), (0, 1), (5, 1)):
        layer = plan.layers[layer_index]
        source = ZeroSource(plan)
        previous = np.zeros((position, layer.kv_heads, layer.head_dim), bfloat16)
        report = run(driver, bundle, source, layer_index, position, hidden, previous, previous,
                     out / f"layer-{layer_index:02d}-position-{position}")
        if not np.array_equal(report["output"].view(np.uint16), hidden.view(np.uint16)):
            raise AssertionError(f"layer {layer_index} changed the zero-weight residual")
        if np.any(report["key"] != 0) or np.any(report["value"] != 0):
            raise AssertionError(f"layer {layer_index} produced nonzero K/V")
        if ("v" in source.roles) == layer.alternative_k_eq_v:
            raise AssertionError("Gemma 4 K=V projection routing is wrong")
        results.append({"layer": layer_index, "position": position,
                        "attention": layer.attention,
                        "alternative_k_eq_v": layer.alternative_k_eq_v,
                        "npu_calls": report["npu_calls"], "passed": True})
    summary = {"variant": plan.variant, "layers": results,
               "npu_calls": sum(x["npu_calls"] for x in results), "passed": True,
               "scope": "zero-weight local/full layer scheduler and K=V routing"}
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.driver.resolve(), a.bundle.resolve(),
                           json.loads(a.config.read_text()), a.out.resolve()), indent=2))
