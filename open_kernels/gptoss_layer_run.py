"""Run one GPT-OSS q4nx token through attention and MoE on NPU."""
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
from gptoss_attention_block_run import run as attention_block_run  # noqa: E402
from gptoss_attention_run import validate_config  # noqa: E402
from gptoss_moe_run import run as moe_run  # noqa: E402
from model.q4nx import Q4NX  # noqa: E402
from recipes.gptoss_source import GptOssSource, validate_header  # noqa: E402


def run(driver: Path, kernels: Path, source: GptOssSource, config: dict,
        layer: int, pos: int, residual: np.ndarray, previous_k: np.ndarray,
        previous_v: np.ndarray, out: Path) -> dict:
    source._prefix(layer)
    validate_config(config)
    post_weight = source.constant(layer, "post_attention_norm")
    if post_weight.shape != (2880,) or post_weight.dtype != bfloat16:
        raise ValueError("GPT-OSS post-attention norm must be BF16[2880]")
    norm_build = kernels / "input_norm"
    if not driver.is_file() or not all((norm_build / name).is_file() for name in
                                       ("final.xclbin", "insts.bin")):
        raise FileNotFoundError("GPT-OSS norm harness or image absent")
    out.mkdir(parents=True, exist_ok=False)
    attention = attention_block_run(driver, kernels, source, config, layer, pos,
                                    residual, previous_k, previous_v, out / "attention_block")
    skip = np.fromfile(out / "attention_block" / "output" / "output.f32", np.float32)
    x = invoke(driver, norm_build, out / "post_attention_norm",
               [("residual", skip), ("weight", post_weight)],
               np.dtype(bfloat16), 2880)
    moe = moe_run(driver, kernels, source, layer, x, skip, out / "moe")
    final = np.fromfile(out / "moe" / "output.f32", np.float32)
    (out / "output.f32").write_bytes(final.tobytes())
    report = {"layer": layer, "position": pos, "mode": attention["mode"],
              "expert_ids": moe["expert_ids"],
              "kernel_calls": attention["kernel_calls"] + 1 + moe["kernel_calls"],
              "passed": True}
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
