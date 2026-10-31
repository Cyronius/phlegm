"""Smoke-check 48 Gemma 4 12B NPU layers, full q8 head, and logit cap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_12b_layer_run_check import ZeroSource
from open_kernels.gemma4_12b_token_run import run
from open_kernels.gemma4_text_plan import from_hf_config


class ZeroModel(ZeroSource):
    def __init__(self, plan):
        super().__init__(plan)
        rng = np.random.default_rng(20261027)
        self.raw_embedding = rng.normal(0, 0.01, plan.hidden).astype(np.float32).astype(bfloat16)
        self.zero_head_band = np.zeros(4 * (plan.hidden // 256) * 8704, np.uint8)

    def embedding(self, token):
        return self.raw_embedding.astype(np.float32)

    def head_band(self, band):
        return self.zero_head_band


def check(driver: Path, bundle: Path, config: dict, out: Path) -> dict:
    plan = from_hf_config(config)
    source = ZeroModel(plan)
    report = run(driver, bundle, source, 0, 0, {}, out)
    scale = np.asarray(np.sqrt(plan.hidden), np.float32).astype(bfloat16)
    expected = (source.raw_embedding * scale).astype(bfloat16)
    if not np.array_equal(report["output"].view(np.uint16), expected.view(np.uint16)):
        raise AssertionError("zero-weight Gemma 4 stack changed the embedding residual")
    if not np.array_equal(report["logits"], np.zeros(plan.vocab, np.float32)):
        raise AssertionError("zero-weight Gemma 4 head produced nonzero logits")
    if len(report["caches"]) != 48 or any(k.shape[0] != 1 or v.shape != k.shape
                                              for k, v in report["caches"].values()):
        raise AssertionError("Gemma 4 layer cache inventory is incomplete")
    if report["next_token"] != 0:
        raise AssertionError("zero logits must select token zero")
    summary = {"variant": plan.variant, "layers": report["layers"],
               "npu_calls": report["npu_calls"], "cache_layers": len(report["caches"]),
               "logits": plan.vocab, "next_token": report["next_token"],
               "passed": True, "scope": "zero-weight first-token schedule"}
    (out / "check-report.json").write_text(json.dumps(summary, indent=2) + "\n")
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
