"""Schedule one Gemma 4 12B token through 48 NPU layers and the q8 head.

The caller supplies per-layer BF16 decode caches. This path deliberately
streams vocabulary weights by 128-logit source band instead of materializing
the 262144-row head matrix in host memory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.designs.gemma4_attention.check_chain import invoke
from open_kernels.gemma4_12b_layer_run import run as layer_run
from open_kernels.gemma4_text_plan import from_hf_config
from open_kernels.gemma4_text_source import Gemma4TextSource
from open_kernels.model.q4nx import Q4NX


def _head(driver: Path, bundle: Path, source: Gemma4TextSource,
          hidden: np.ndarray, out: Path) -> np.ndarray:
    plan = source.plan
    if hidden.shape != (plan.hidden,) or hidden.dtype != bfloat16:
        raise ValueError("Gemma 4 head requires a BF16 hidden vector")
    image = bundle / "lm_head"
    out.mkdir(parents=True, exist_ok=False)
    weight_path = out / "weights.bin"
    bands = plan.vocab // 128
    one_bytes = 4 * (plan.hidden // 256) * 8704
    with weight_path.open("wb") as stream:
        for band in range(bands):
            packed = source.head_band(band)
            if packed.dtype != np.uint8 or packed.nbytes != one_bytes:
                raise ValueError(f"Gemma 4 head band {band} has unexpected size")
            stream.write(packed.tobytes())
    x_path = out / "activation.bin"
    x_path.write_bytes(hidden.tobytes())
    result_path = out / "logits.f32"
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {image / 'final.xclbin'}",
        f"kernelx k G {image / 'insts.bin'}",
        f"buf weights {one_bytes * bands} {weight_path}",
        f"buf activation {hidden.nbytes} {x_path}",
        f"buf result {plan.vocab * 4}", "run k weights activation result",
        f"dump result {result_path} {plan.vocab * 4}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"Gemma 4 NPU head failed: {proc.stdout}\n{proc.stderr}")
    logits = np.fromfile(result_path, np.float32)
    if logits.shape != (plan.vocab,) or not np.isfinite(logits).all():
        raise AssertionError("invalid Gemma 4 q8 head output")
    return logits


def run(driver: Path, bundle: Path, source: Gemma4TextSource, token: int,
        position: int, caches: dict[int, tuple[np.ndarray, np.ndarray]],
        out: Path) -> dict:
    plan = source.plan
    if plan.variant != "12B" or len(plan.layers) != 48:
        raise ValueError("expected the pinned Gemma 4 12B text plan")
    if not isinstance(position, int) or position < 0 or position >= plan.max_context:
        raise ValueError("Gemma 4 token position out of range")
    out.mkdir(parents=True, exist_ok=False)
    raw = source.embedding(token).astype(bfloat16)
    embed_scale = np.asarray(np.sqrt(plan.hidden), np.float32).astype(bfloat16)
    hidden = (raw * embed_scale).astype(bfloat16)
    if hidden.shape != (plan.hidden,) or not np.isfinite(hidden.astype(np.float32)).all():
        raise ValueError("invalid Gemma 4 source embedding")
    next_caches = {}
    calls = 0
    layer_reports = []
    for layer in plan.layers:
        if layer.index in caches:
            previous_k, previous_v = caches[layer.index]
        elif position == 0:
            previous_k = previous_v = np.empty((0, layer.kv_heads, layer.head_dim), bfloat16)
        else:
            raise ValueError(f"missing Gemma 4 cache for layer {layer.index}")
        report = layer_run(driver, bundle, source, layer.index, position, hidden,
                           previous_k, previous_v, out / f"layer-{layer.index:02d}")
        hidden = report["output"]
        k, v = report["key"], report["value"]
        if layer.sliding_window:
            k, v = k[-(layer.sliding_window - 1):], v[-(layer.sliding_window - 1):]
        next_caches[layer.index] = (k.copy(), v.copy())
        calls += report["npu_calls"]
        layer_reports.append({key: report[key] for key in
                              ("layer", "attention", "alternative_k_eq_v", "npu_calls")})
    scale = source.vector("model.norm.weight", plan.hidden)
    pair = np.concatenate((hidden, np.zeros(plan.hidden, bfloat16)))
    final_hidden = invoke(driver, bundle / "hidden_norm", out / "final_norm",
                          [pair, scale], plan.hidden * 2, bfloat16)
    calls += 1
    logits = _head(driver, bundle, source, final_hidden, out / "lm_head")
    calls += 1
    capped = invoke(driver, bundle / "logit_cap", out / "logit_cap",
                    [logits], plan.vocab * 4, np.float32)
    calls += 1
    if capped.shape != (plan.vocab,) or not np.isfinite(capped).all():
        raise AssertionError("invalid Gemma 4 capped logits")
    summary = {"variant": plan.variant, "position": position, "token": token,
               "layers": len(layer_reports), "npu_calls": calls,
               "next_token": int(np.argmax(capped)), "passed": True}
    (out / "report.json").write_text(json.dumps({**summary, "layer_reports": layer_reports}, indent=2) + "\n")
    return {**summary, "output": hidden, "final_hidden": final_hidden,
            "logits": capped, "caches": next_caches}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "bundle", "config", "model", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--token", required=True, type=int)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        source = Gemma4TextSource(reader, from_hf_config(json.loads(a.config.read_text())))
        source.validate()
        report = run(a.driver.resolve(), a.bundle.resolve(), source, a.token, 0, {}, a.out.resolve())
        print(json.dumps({k: v for k, v in report.items()
                          if not isinstance(v, (np.ndarray, dict))}, indent=2))
    finally:
        reader.mm.close()
        reader.f.close()
