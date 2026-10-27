"""Run a GPT-OSS token through source attention norm, attention, and residual."""
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
from gptoss_attention_run import run as attention_run, validate_config  # noqa: E402
from gptoss_o_run import run as output_run  # noqa: E402
from model.q4nx import Q4NX  # noqa: E402
from recipes.gptoss_source import GptOssSource, validate_header  # noqa: E402


def run(driver: Path, kernels: Path, source: GptOssSource, config: dict,
        layer: int, pos: int, residual: np.ndarray, previous_k: np.ndarray,
        previous_v: np.ndarray, out: Path) -> dict:
    source._prefix(layer)
    validate_config(config)
    skip = np.asarray(residual)
    if skip.shape != (2880,) or skip.dtype != np.float32 or not np.isfinite(skip).all():
        raise ValueError("GPT-OSS block residual must be finite FP32[2880]")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    builds = {name: kernels / name for name in (
        "input_norm", "qkv_projection_16_bands", "yarn_rope",
        "full_sink_attention_state_512", "attention_finalize_bf16",
        "o_projection_16_bands", "residual_bias")}
    for build in builds.values():
        for name in ("final.xclbin", "insts.bin"):
            if not (build / name).is_file():
                raise FileNotFoundError(build / name)
    norm_weight = source.constant(layer, "input_norm")
    if norm_weight.shape != (2880,) or norm_weight.dtype != bfloat16:
        raise ValueError("GPT-OSS input norm must be BF16[2880]")
    out.mkdir(parents=True, exist_ok=False)
    x = invoke(driver, builds["input_norm"], out / "norm",
               [("residual", skip), ("weight", norm_weight)],
               np.dtype(bfloat16), 2880)
    attention = attention_run(driver, builds["qkv_projection_16_bands"],
                              builds["yarn_rope"], builds["full_sink_attention_state_512"],
                              builds["attention_finalize_bf16"], source, config, layer, pos,
                              x, previous_k, previous_v, out / "attention")
    attn_value = np.fromfile(out / "attention" / "attention.bf16", bfloat16)
    output = output_run(driver, builds["o_projection_16_bands"],
                        builds["residual_bias"], source, layer, attn_value,
                        skip, out / "output")
    report = {"layer": layer, "position": pos, "mode": attention["mode"],
              "kernel_calls": 1 + attention["qkv_calls"] + attention["rope_calls"] +
              attention["attention_state_calls"] + attention["finalize_calls"] +
              output["projection_calls"] + output["residual_calls"],
              "output": str(out / "output" / "output.f32"), "passed": True}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "kernels", "model", "config", "residual", "previous-k",
                 "previous-v", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--pos", required=True, type=int)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        validate_header(reader.tensors)
        print(json.dumps(run(a.driver.resolve(), a.kernels.resolve(), GptOssSource(reader),
                             json.loads(a.config.read_text()), a.layer, a.pos,
                             np.fromfile(a.residual, np.float32),
                             np.fromfile(a.previous_k, bfloat16).reshape(-1, 8, 64),
                             np.fromfile(a.previous_v, bfloat16).reshape(-1, 8, 64),
                             a.out.resolve()), indent=2))
    finally:
        reader.mm.close()
        reader.f.close()
