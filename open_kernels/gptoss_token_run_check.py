"""Smoke-check a 24-layer GPT-OSS NPU token and cache using zero fixture weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from designs.gptoss_rope.check import CONFIG  # noqa: E402
from gptoss_token_run import run, cache_rows  # noqa: E402
from recipes.gptoss_source import GptOssSource  # noqa: E402


def check(driver: Path, kernels: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261031)
    embedding = rng.normal(0, 0.5, 2880).astype(np.float32).astype(bfloat16)
    router_bias = np.zeros(32, bfloat16)
    router_bias[[17, 9, 3, 26]] = np.array([12, 10, 8, 6], bfloat16)
    zeros_q4 = np.zeros((16, 23, 2560), np.uint8)
    zeros_expert = np.zeros((90, 23, 2560), np.uint8)

    class FixtureSource(GptOssSource):
        def __init__(self):
            pass

        def embedding(self, token):
            return embedding

        def constant(self, layer, name):
            if name in ("input_norm", "post_attention_norm"):
                return np.ones(2880, bfloat16)
            if name in ("q_bias",):
                return np.zeros(4096, bfloat16)
            if name in ("k_bias", "v_bias"):
                return np.zeros(512, bfloat16)
            if name == "o_bias":
                return np.zeros(2880, bfloat16)
            if name == "sinks":
                return np.zeros(64, bfloat16)
            if name == "router_bias":
                return router_bias
            raise ValueError(name)

        def attention(self, layer, role, start):
            return zeros_q4

        def router_weight(self, layer):
            return np.zeros((2880, 32), bfloat16)

        def expert(self, layer, expert_id, role):
            return zeros_expert.copy()

        def expert_bias(self, layer, expert_id, role):
            return np.zeros(2880, bfloat16)

        def final_norm(self):
            return np.ones(2880, bfloat16)

        def head_band(self, band):
            return np.zeros((24, 5120), np.uint8)

    report = run(driver, kernels, FixtureSource(), CONFIG, 0, 0, None, out)
    logits = np.fromfile(out / "logits.f32", np.float32)
    if logits.shape != (201088,) or not np.array_equal(logits, np.zeros_like(logits)):
        raise AssertionError("zero GPT-OSS head fixture should emit zero logits")
    if report["layers"] != 24 or report["next_token"] != 0:
        raise AssertionError("incomplete GPT-OSS token schedule")
    for layer in range(24):
        got = np.fromfile(out / f"layer-{layer:02d}" / "output.f32", np.float32)
        if not np.array_equal(got, embedding.astype(np.float32)):
            raise AssertionError(f"GPT-OSS zero-weight layer {layer} changed residual")
        key, value = cache_rows(out / "cache", layer, 1,
                                "sliding" if layer % 2 == 0 else "full")
        if key.shape != (1, 8, 64) or np.any(key) or np.any(value):
            raise AssertionError(f"GPT-OSS layer {layer} cache handoff is invalid")
    result = {"layers": report["layers"], "kernel_calls": report["kernel_calls"],
              "logits": logits.size, "cache_layers": 24, "passed": True}
    (out / "check.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "kernels", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.driver.resolve(), a.kernels.resolve(),
                           a.out.resolve()), indent=2))
