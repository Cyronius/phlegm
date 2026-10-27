"""Run one GPT-OSS source QKV, YaRN, sink attention, and BF16 finalization."""
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
from designs.gptoss_rope.check import CONFIG  # noqa: E402
from gptoss_qkv_run import run as qkv_run  # noqa: E402
from model.q4nx import Q4NX  # noqa: E402
from recipes.gptoss_source import GptOssSource, validate_header  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402


def validate_config(config: dict) -> ModelSpec:
    """Require the fixed image geometry and FLM GPT-OSS YaRN/layer schedule."""
    spec = ModelSpec.from_hf_config(config)
    baseline = ModelSpec.from_hf_config(CONFIG)
    fields = ("hidden", "num_layers", "num_heads", "num_kv_heads", "head_dim",
              "rotary_dim", "sliding_window", "rope_theta", "rope_scaling",
              "layer_types")
    if any(getattr(spec, key) != getattr(baseline, key) for key in fields):
        raise ValueError("GPT-OSS config differs from the checked NPU image geometry")
    if config.get("max_position_embeddings") != CONFIG["max_position_embeddings"]:
        raise ValueError("GPT-OSS context length differs from the checked YaRN table")
    return spec


def run(driver: Path, qkv_build: Path, rope_build: Path, state_build: Path,
        finalize_build: Path, source: GptOssSource, config: dict, layer: int,
        pos: int, activation: np.ndarray, previous_k: np.ndarray,
        previous_v: np.ndarray, out: Path) -> dict:
    source._prefix(layer)
    spec = validate_config(config)
    mode = "sliding" if spec.layer_types[layer] == "dense_local" else "full"
    if not 0 <= pos < CONFIG["max_position_embeddings"]:
        raise ValueError("invalid GPT-OSS position")
    rows = pos if mode == "full" else min(pos, 127)
    for name, cache in (("key", previous_k), ("value", previous_v)):
        if (cache.shape != (rows, 8, 64) or cache.dtype != bfloat16 or
                not np.isfinite(cache.astype(np.float32)).all()):
            raise ValueError(f"{name} cache must be finite BF16[{rows},8,64]")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for build in (qkv_build, rope_build, state_build, finalize_build):
        for name in ("final.xclbin", "insts.bin"):
            if not (build / name).is_file():
                raise FileNotFoundError(build / name)
    sinks = source.constant(layer, "sinks")
    if sinks.shape != (64,) or sinks.dtype != bfloat16:
        raise ValueError("GPT-OSS attention sinks must be BF16[64]")
    out.mkdir(parents=True, exist_ok=False)
    qkv_run(driver, qkv_build, source, layer, activation, out / "qkv")
    qk = np.fromfile(out / "qkv" / "qk.bf16", bfloat16)
    v = np.fromfile(out / "qkv" / "v.bf16", bfloat16).reshape(8, 64)
    inv = np.asarray(spec.rope_inv_freq(ctx=pos + 1), np.float64)
    angles = pos * inv
    scale = spec.rope_scale()
    table = np.concatenate((scale * np.cos(angles), scale * np.sin(angles))).astype(np.float32)
    rotated = invoke(driver, rope_build, out / "rope",
                     [("qk", qk), ("table", table)], np.dtype(bfloat16), 72 * 64)
    rotated = rotated.reshape(72, 64)
    key = rotated[64:]
    (out / "key.bf16").write_bytes(key.tobytes())
    (out / "value.bf16").write_bytes(v.tobytes())
    cached_k = np.concatenate((previous_k, key[None]), axis=0)
    cached_v = np.concatenate((previous_v, v[None]), axis=0)
    kv = np.ascontiguousarray(np.concatenate((cached_k, cached_v), axis=1))
    state = np.zeros((64, 68), np.float32)
    state[:, 64] = sinks.astype(np.float32)
    state[:, 65] = 1
    chunks = 0
    for start in range(0, len(kv), 512):
        valid = min(512, len(kv) - start)
        meta = np.array([valid, 0, 0, 0], np.int32)
        q_state = np.frombuffer(rotated[:64].tobytes() + state.tobytes() +
                                meta.tobytes(), np.uint8).copy()
        padded = np.zeros((512, 16, 64), bfloat16)
        padded[:valid] = kv[start:start + valid]
        state = invoke(driver, state_build, out / f"state-{chunks:03d}",
                       [("q_state", q_state), ("kv_rows", padded)],
                       np.dtype(np.float32), 64 * 68).reshape(64, 68)
        if np.any(state[:, 65] <= 0):
            raise AssertionError("invalid carried GPT-OSS attention denominator")
        chunks += 1
    result = invoke(driver, finalize_build, out / "finalize", [("state", state)],
                    np.dtype(bfloat16), 4096)
    (out / "attention.bf16").write_bytes(result.tobytes())
    report = {"layer": layer, "position": pos, "mode": mode,
              "attention_rows": len(kv), "qkv_calls": 10, "rope_calls": 1,
              "attention_state_calls": chunks, "finalize_calls": 1, "passed": True}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "qkv-build", "rope-build", "state-build", "finalize-build",
                 "model", "config", "activation", "previous-k", "previous-v", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--pos", required=True, type=int)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        validate_header(reader.tensors)
        print(json.dumps(run(a.driver.resolve(), a.qkv_build.resolve(),
                             a.rope_build.resolve(), a.state_build.resolve(),
                             a.finalize_build.resolve(), GptOssSource(reader),
                             json.loads(a.config.read_text()), a.layer, a.pos,
                             np.fromfile(a.activation, bfloat16),
                             np.fromfile(a.previous_k, bfloat16).reshape(-1, 8, 64),
                             np.fromfile(a.previous_v, bfloat16).reshape(-1, 8, 64),
                             a.out.resolve()), indent=2))
    finally:
        reader.mm.close()
        reader.f.close()
