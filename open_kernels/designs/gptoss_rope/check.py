"""Check GPT-OSS 20B YaRN RoPE on NPU against the FP64 layer replica."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from recipes.spec import ModelSpec  # noqa: E402
from replica_dense import rope  # noqa: E402

CONFIG = {
    "model_type": "gpt_oss", "hidden_size": 2880, "intermediate_size": 2880,
    "num_hidden_layers": 24, "num_attention_heads": 64, "num_key_value_heads": 8,
    "head_dim": 64, "num_local_experts": 32, "num_experts_per_tok": 4,
    "vocab_size": 201088, "sliding_window": 128, "rope_theta": 150000.0,
    "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0,
                     "beta_slow": 1.0, "truncate": False,
                     "original_max_position_embeddings": 4096},
    "rms_norm_eps": 1e-5, "hidden_act": "silu", "attention_bias": True,
    "swiglu_limit": 7.0, "max_position_embeddings": 131072,
}


def fixture(pos: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pos < 0 or pos >= CONFIG["max_position_embeddings"]:
        raise ValueError("position must be within GPT-OSS's context")
    spec = ModelSpec.from_hf_config(CONFIG)
    rng = np.random.default_rng(20261027 + pos)
    qk = rng.normal(0, 0.7, (72, 64)).astype(np.float32).astype(bfloat16)
    inv = np.asarray(spec.rope_inv_freq(ctx=pos + 1), np.float64)
    ang = pos * inv
    scale = spec.rope_scale()
    table = np.concatenate((scale * np.cos(ang), scale * np.sin(ang))).astype(np.float32)
    expected = rope(qk, pos, 64, spec.rope_theta, inv, scale).astype(bfloat16)
    return qk, table, expected


def run(driver: Path, build: Path, out: Path, pos: int) -> dict:
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for name in ("final.xclbin", "insts.bin"):
        if not (build / name).is_file():
            raise FileNotFoundError(build / name)
    qk, table, expected = fixture(pos)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qk.bin").write_bytes(qk.tobytes())
    (out / "table.bin").write_bytes(table.tobytes())
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {build / 'final.xclbin'}",
        f"kernelx k G {build / 'insts.bin'}",
        f"buf qk {qk.nbytes} {out / 'qk.bin'}",
        f"buf table {table.nbytes} {out / 'table.bin'}",
        f"buf result {qk.nbytes}", "run k qk table result",
        f"dump result {out / 'result.bin'} {qk.nbytes}", "",
    ]))
    completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"NPU harness failed: {completed.stdout}\n{completed.stderr}")
    actual = np.fromfile(out / "result.bin", dtype=bfloat16)
    if actual.size != expected.size or not np.isfinite(actual.astype(np.float32)).all():
        raise AssertionError("invalid NPU output")
    a = actual.astype(np.float32)
    e = expected.ravel().astype(np.float32)
    max_abs = float(np.max(np.abs(a - e)))
    a64, e64 = a.astype(np.float64), e.astype(np.float64)
    cosine = float(a64 @ e64 / (np.linalg.norm(a64) * np.linalg.norm(e64)))
    report = {"passed": bool(max_abs < 0.015 and cosine > 0.99999),
              "position": pos, "max_absolute_error": max_abs,
              "cosine": cosine, "exact_bf16_fraction": float(np.mean(a == e)),
              "reference": "ModelSpec YaRN table and replica_dense.rope"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--pos", required=True, type=int)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.build.resolve(), a.out.resolve(), a.pos), indent=2))


if __name__ == "__main__":
    main()
