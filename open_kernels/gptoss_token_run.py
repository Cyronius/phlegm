"""Run one GPT-OSS q4nx token through 24 NPU layers and the full LM head."""
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
from gptoss_attention_run import validate_config  # noqa: E402
from gptoss_layer_run import run as layer_run  # noqa: E402
from model.q4nx import Q4NX  # noqa: E402
from recipes.gptoss_source import GptOssSource, HEAD_BANDS, validate_header  # noqa: E402


def cache_rows(cache_in: Path | None, layer: int, pos: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    rows = pos if mode == "full" else min(pos, 127)
    if rows == 0:
        return np.empty((0, 8, 64), bfloat16), np.empty((0, 8, 64), bfloat16)
    if cache_in is None:
        raise ValueError("GPT-OSS decode after position zero needs a cache directory")
    pair = []
    for role in ("key", "value"):
        path = cache_in / f"layer-{layer:02d}" / f"{role}.bf16"
        array = np.fromfile(path, bfloat16)
        if array.size != rows * 8 * 64:
            raise ValueError(f"GPT-OSS layer {layer} {role} cache needs {rows} rows")
        pair.append(array.reshape(rows, 8, 64))
    return pair[0], pair[1]


def run(driver: Path, kernels: Path, source: GptOssSource, config: dict,
        token: int, pos: int, cache_in: Path | None, out: Path) -> dict:
    spec = validate_config(config)
    if not 0 <= pos < config["max_position_embeddings"]:
        raise ValueError("GPT-OSS token position outside model context")
    if not isinstance(token, int) or not 0 <= token < spec.vocab:
        raise ValueError("GPT-OSS token outside model vocabulary")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for name in ("input_norm", "lm_head_201088"):
        for part in ("final.xclbin", "insts.bin"):
            if not (kernels / name / part).is_file():
                raise FileNotFoundError(kernels / name / part)
    caches = []
    for layer in range(spec.num_layers):
        mode = "sliding" if spec.layer_types[layer] == "dense_local" else "full"
        caches.append(cache_rows(cache_in, layer, pos, mode))
    embedding = source.embedding(token)
    if embedding.shape != (2880,) or embedding.dtype != bfloat16:
        raise ValueError("GPT-OSS embedding must be BF16[2880]")
    residual = embedding.astype(np.float32)
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    for layer, (previous_k, previous_v) in enumerate(caches):
        layer_out = out / f"layer-{layer:02d}"
        report = layer_run(driver, kernels, source, config, layer, pos, residual,
                           previous_k, previous_v, layer_out)
        residual = np.fromfile(layer_out / "output.f32", np.float32)
        if residual.size != 2880 or not np.isfinite(residual).all():
            raise AssertionError(f"invalid GPT-OSS layer {layer} residual")
        cache_out = out / "cache" / f"layer-{layer:02d}"
        cache_out.mkdir(parents=True)
        for role, previous in (("key", previous_k), ("value", previous_v)):
            current = np.fromfile(layer_out / "attention_block" / "attention" /
                                  f"{role}.bf16", bfloat16).reshape(1, 8, 64)
            retained = np.concatenate((previous, current), axis=0)
            if report["mode"] == "sliding":
                retained = retained[-127:]
            (cache_out / f"{role}.bf16").write_bytes(retained.tobytes())
        reports.append(report)
    norm_weight = source.final_norm()
    final = invoke(driver, kernels / "input_norm", out / "final_norm",
                   [("residual", residual), ("weight", norm_weight)],
                   np.dtype(bfloat16), 2880)
    padded = np.zeros(3072, bfloat16)
    padded[:2880] = final
    packed = np.empty((HEAD_BANDS, 24, 5120), np.uint8)
    for band in range(HEAD_BANDS):
        packed[band] = source.head_band(band)
    logits = invoke(driver, kernels / "lm_head_201088", out / "lm_head",
                    [("weights", packed), ("x", padded)],
                    np.dtype(np.float32), spec.vocab)
    (out / "logits.f32").write_bytes(logits.tobytes())
    report = {"token": token, "position": pos, "layers": len(reports),
              "kernel_calls": sum(r["kernel_calls"] for r in reports) + 2,
              "logits": logits.size, "next_token": int(np.argmax(logits)),
              "cache": str(out / "cache"), "passed": True}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "kernels", "model", "config", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--token", required=True, type=int)
    p.add_argument("--pos", required=True, type=int)
    p.add_argument("--cache-in", type=Path)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        validate_header(reader.tensors)
        print(json.dumps(run(a.driver.resolve(), a.kernels.resolve(), GptOssSource(reader),
                             json.loads(a.config.read_text()), a.token, a.pos,
                             a.cache_in.resolve() if a.cache_in else None,
                             a.out.resolve()), indent=2))
    finally:
        reader.mm.close()
        reader.f.close()
