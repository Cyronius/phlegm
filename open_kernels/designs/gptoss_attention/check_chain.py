"""Check GPT-OSS QKV projection -> YaRN RoPE -> sliding sink attention on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "designs" / "gptoss_qkv_gemv"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
from check import fixture as qkv_fixture, run as qkv_run  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402
from replica_dense import rope  # noqa: E402
from replica_gptoss import sink_attention  # noqa: E402
from designs.gptoss_rope.check import CONFIG  # noqa: E402


def invoke(driver: Path, build: Path, out: Path,
           inputs: list[tuple[str, np.ndarray]], dtype: np.dtype, count: int) -> np.ndarray:
    out.mkdir(parents=True, exist_ok=True)
    lines = ["device", f"xclbin G {build / 'final.xclbin'}",
             f"kernelx k G {build / 'insts.bin'}"]
    for name, value in inputs:
        path = out / f"{name}.bin"
        path.write_bytes(value.tobytes())
        lines.append(f"buf {name} {value.nbytes} {path}")
    lines += [f"buf result {count * np.dtype(dtype).itemsize}",
              "run k " + " ".join(name for name, _ in inputs) + " result",
              f"dump result {out / 'result.bin'} {count * np.dtype(dtype).itemsize}", ""]
    config = out / "run.cfg"
    config.write_text("\n".join(lines))
    completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{out.name}: {completed.stdout}\n{completed.stderr}")
    result = np.fromfile(out / "result.bin", dtype=dtype)
    if result.size != count or not np.isfinite(result.astype(np.float32)).all():
        raise AssertionError(f"{out.name}: invalid NPU output")
    return result


def run(driver: Path, qkv_build: Path, rope_build: Path, attn_build: Path,
        out: Path, pos: int) -> dict:
    if pos < 127 or pos >= CONFIG["max_position_embeddings"]:
        raise ValueError("position must be from 127 to 131071 for the 128-row window")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for build in (qkv_build, rope_build, attn_build):
        for name in ("final.xclbin", "insts.bin"):
            if not (build / name).is_file():
                raise FileNotFoundError(build / name)
    rng = np.random.default_rng(20261028 + pos)
    x = rng.normal(0, 0.5, 2880).astype(np.float32).astype(bfloat16)
    outputs = {}
    references = {}
    projection_reports = []
    for role in ("q", "k", "v"):
        starts = range(0, 128, 16) if role == "q" else (0,)
        chunks = []
        gold = []
        for start in starts:
            directory = out / f"{role}-{start:03d}"
            projection_reports.append(qkv_run(driver, qkv_build, directory,
                                              role, start, 16, x))
            chunks.append(np.fromfile(directory / "result.bin", np.float32))
            gold.append(qkv_fixture(role, start, 16, x)[2])
        outputs[role] = np.concatenate(chunks)
        references[role] = np.concatenate(gold)

    spec = ModelSpec.from_hf_config(CONFIG)
    inv = np.asarray(spec.rope_inv_freq(ctx=pos + 1), np.float64)
    angles = pos * inv
    scale = spec.rope_scale()
    table = np.concatenate((scale * np.cos(angles), scale * np.sin(angles))).astype(np.float32)
    qk = np.concatenate((outputs["q"], outputs["k"])).astype(bfloat16)
    rotated = invoke(driver, rope_build, out / "rope", [("qk", qk), ("table", table)],
                     np.dtype(bfloat16), 72 * 64).reshape(72, 64)
    q_gold = references["q"].astype(bfloat16).reshape(64, 64)
    k_gold = references["k"].astype(bfloat16).reshape(8, 64)
    q_gold = rope(q_gold, pos, 64, spec.rope_theta, inv, scale).astype(bfloat16)
    k_gold = rope(k_gold, pos, 64, spec.rope_theta, inv, scale).astype(bfloat16)

    previous_k = rng.normal(0, 0.4, (127, 8, 64)).astype(np.float32).astype(bfloat16)
    previous_v = rng.normal(0, 0.5, (127, 8, 64)).astype(np.float32).astype(bfloat16)
    v = outputs["v"].astype(bfloat16).reshape(8, 64)
    v_gold = references["v"].astype(bfloat16).reshape(8, 64)
    K = np.concatenate((previous_k, rotated[64:][None]), axis=0)
    V = np.concatenate((previous_v, v[None]), axis=0)
    K_gold = np.concatenate((previous_k, k_gold[None]), axis=0)
    V_gold = np.concatenate((previous_v, v_gold[None]), axis=0)
    sinks = rng.normal(0, 1.5, 64).astype(np.float32)
    q_sink = np.frombuffer(rotated[:64].tobytes() + sinks.tobytes(), np.uint8).copy()
    kv = np.ascontiguousarray(np.concatenate((K, V), axis=1))
    actual = invoke(driver, attn_build, out / "attention",
                    [("q_sink", q_sink), ("kv", kv)], np.dtype(np.float32), 64 * 64)
    # The fixture holds only the last 128 rows. Its local position is 127;
    # absolute pos is used solely to form the YaRN table for the new token.
    expected = sink_attention(q_gold, K_gold, V_gold, sinks,
                              sliding_window=128, pos=127).ravel().astype(np.float32)
    (out / "attention" / "expected.bin").write_bytes(expected.tobytes())
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    max_abs = float(np.max(np.abs(a - e)))
    max_rel = float(max_abs / max(np.max(np.abs(e)), 1e-30))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    report = {"passed": bool(max_rel < 0.005 and cosine > 0.9999),
              "position": pos, "projection_calls": len(projection_reports),
              "projection_max_relative_error": max(r["max_relative_error"] for r in projection_reports),
              "attention_max_absolute_error": max_abs,
              "attention_max_relative_error": max_rel,
              "attention_cosine": cosine,
              "reference": "q4nx decoded QKV, ModelSpec YaRN, FP64 sink attention"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--qkv-build", required=True, type=Path)
    p.add_argument("--rope-build", required=True, type=Path)
    p.add_argument("--attn-build", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--pos", type=int, default=127)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.qkv_build.resolve(), a.rope_build.resolve(),
                         a.attn_build.resolve(), a.out.resolve(), a.pos), indent=2))


if __name__ == "__main__":
    main()
