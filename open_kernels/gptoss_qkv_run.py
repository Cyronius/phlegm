"""Run all GPT-OSS 20B Q/K/V projection bands from a q4nx source on NPU."""
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

CALLS = {"q": tuple(range(0, 128, 16)), "k": (0,), "v": (0,)}


def run(driver: Path, build: Path, source: GptOssSource, layer: int,
        activation: np.ndarray, out: Path) -> dict:
    source._prefix(layer)
    x = np.asarray(activation)
    if x.shape != (2880,) or x.dtype != bfloat16 or not np.isfinite(x.astype(np.float32)).all():
        raise ValueError("GPT-OSS QKV input must be finite BF16[2880]")
    if not driver.is_file() or not all((build / part).is_file() for part in
                                       ("final.xclbin", "insts.bin")):
        raise FileNotFoundError("missing GPT-OSS QKV harness or image")
    out.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for role, starts in CALLS.items():
        pieces = []
        for start in starts:
            packed = source.qkv_input(layer, role, start, x)
            chunks = source.attention(layer, role, start)
            result = invoke(driver, build, out / f"{role}-{start:03d}",
                            [("x_bias", packed), ("chunks", chunks)],
                            np.dtype(np.float32), 512)
            pieces.append(result)
        outputs[role] = np.concatenate(pieces)
    if tuple(outputs[role].size for role in "qkv") != (4096, 512, 512):
        raise AssertionError("incomplete GPT-OSS Q/K/V projection bands")
    qk = np.concatenate((outputs["q"], outputs["k"])).astype(bfloat16)
    qk_path = out / "qk.bf16"
    qk_path.write_bytes(qk.tobytes())
    (out / "v.bf16").write_bytes(outputs["v"].astype(bfloat16).tobytes())
    report = {"layer": layer, "kernel_calls": 10, "q_channels": 4096,
              "k_channels": 512, "v_channels": 512,
              "qk_for_rope": str(qk_path), "passed": True}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "build", "model", "activation", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--layer", required=True, type=int)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        validate_header(reader.tensors)
        x = np.fromfile(a.activation, bfloat16)
        print(json.dumps(run(a.driver.resolve(), a.build.resolve(), GptOssSource(reader),
                             a.layer, x, a.out.resolve()), indent=2))
    finally:
        reader.mm.close()
        reader.f.close()
