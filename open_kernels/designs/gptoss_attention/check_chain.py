"""Check GPT-OSS QKV, YaRN and sliding or full sink attention on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
from designs.gptoss_qkv_gemv.check import fixture as qkv_fixture, run as qkv_run  # noqa: E402
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
    result_path = out / "result.bin"
    # An occasional long sequence of XRT launches has returned a short or
    # nonfinite dump for a pure GEMV. Bound the recovery and record any replay.
    for launch in range(3):
        completed = subprocess.run([str(driver), str(config)], capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"{out.name}: {completed.stdout}\n{completed.stderr}")
        for attempt in range(20):
            result = np.fromfile(result_path, dtype=dtype)
            if result.size == count and np.isfinite(result.astype(np.float32)).all():
                if launch:
                    (out / "retries.json").write_text(json.dumps({"replays": launch}) + "\n")
                return result
            if result.size == count:
                break
            if attempt < 19:
                time.sleep(0.1)
    raise AssertionError(f"{out.name}: invalid NPU output after 3 launches "
                         f"({result.size} of {count})")


def run(driver: Path, qkv_build: Path, rope_build: Path, attn_build: Path,
        out: Path, pos: int, activation: np.ndarray | None = None,
        oracle_activation: np.ndarray | None = None,
        full_state_build: Path | None = None,
        finalize_build: Path | None = None) -> dict:
    if finalize_build is not None and full_state_build is None:
        raise ValueError("attention finalizer requires the full-state image")
    if full_state_build is None and (pos < 127 or pos >= CONFIG["max_position_embeddings"]):
        raise ValueError("sliding position must be from 127 to 131071")
    if full_state_build is not None and (pos < 0 or pos > 4096):
        raise ValueError("full-layer fixture supports positions 0 through 4096")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    builds = (qkv_build, rope_build, attn_build)
    for build in builds + ((full_state_build,) if full_state_build is not None else ()) + \
            ((finalize_build,) if finalize_build is not None else ()):
        for name in ("final.xclbin", "insts.bin"):
            if not (build / name).is_file():
                raise FileNotFoundError(build / name)
    rng = np.random.default_rng(20261028 + pos)
    x = (rng.normal(0, 0.5, 2880).astype(np.float32).astype(bfloat16)
         if activation is None else np.asarray(activation))
    if x.shape != (2880,) or x.dtype != bfloat16:
        raise ValueError("QKV input must be BF16[2880]")
    golden_x = x if oracle_activation is None else np.asarray(oracle_activation)
    if golden_x.shape != (2880,) or golden_x.dtype != bfloat16:
        raise ValueError("oracle QKV input must be BF16[2880]")
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
            gold.append(qkv_fixture(role, start, 16, golden_x)[2])
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

    previous_rows = pos if full_state_build is not None else 127
    previous_k = rng.normal(0, 0.4, (previous_rows, 8, 64)).astype(np.float32).astype(bfloat16)
    previous_v = rng.normal(0, 0.5, (previous_rows, 8, 64)).astype(np.float32).astype(bfloat16)
    v = outputs["v"].astype(bfloat16).reshape(8, 64)
    v_gold = references["v"].astype(bfloat16).reshape(8, 64)
    K = np.concatenate((previous_k, rotated[64:][None]), axis=0)
    V = np.concatenate((previous_v, v[None]), axis=0)
    K_gold = np.concatenate((previous_k, k_gold[None]), axis=0)
    V_gold = np.concatenate((previous_v, v_gold[None]), axis=0)
    sinks = rng.normal(0, 1.5, 64).astype(np.float32)
    q_sink = np.frombuffer(rotated[:64].tobytes() + sinks.tobytes(), np.uint8).copy()
    kv = np.ascontiguousarray(np.concatenate((K, V), axis=1))
    finalize_exact = None
    if full_state_build is None:
        actual = invoke(driver, attn_build, out / "attention",
                        [("q_sink", q_sink), ("kv", kv)], np.dtype(np.float32), 64 * 64)
        # The fixture holds only the last 128 rows. Its local position is 127;
        # absolute pos is used solely to form the YaRN table for the new token.
        expected = sink_attention(q_gold, K_gold, V_gold, sinks,
                                  sliding_window=128, pos=127).ravel().astype(np.float32)
    else:
        state = np.zeros((64, 68), np.float32)
        state[:, 64] = sinks
        state[:, 65] = 1
        for start in range(0, len(kv), 512):
            valid = min(512, len(kv) - start)
            meta = np.array([valid, 0, 0, 0], np.int32)
            q_state = np.frombuffer(rotated[:64].tobytes() + state.tobytes() +
                                    meta.tobytes(), np.uint8).copy()
            rows = np.zeros((512, 16, 64), bfloat16)
            rows[:valid] = kv[start:start + valid]
            state = invoke(driver, full_state_build,
                           out / "attention" / f"chunk-{start // 512:02d}",
                           [("q_state", q_state), ("kv_rows", rows)],
                           np.dtype(np.float32), 64 * 68).reshape(64, 68)
            if np.any(state[:, 65] <= 0) or not np.isfinite(state).all():
                raise AssertionError("invalid carried full-attention state")
        actual = (state[:, :64] / state[:, 65, None]).ravel().astype(np.float32)
        (out / "attention").mkdir(parents=True, exist_ok=True)
        (out / "attention" / "result.bin").write_bytes(actual.tobytes())
        if finalize_build is not None:
            projection_input = invoke(driver, finalize_build,
                                      out / "attention" / "finalize",
                                      [("state", state)], np.dtype(bfloat16), 4096)
            golden_input = (state[:, :64].astype(np.float64) /
                            state[:, 65, None].astype(np.float64)).astype(bfloat16).ravel()
            finalize_exact = float(np.mean(projection_input == golden_input))
            if finalize_exact != 1.0:
                raise AssertionError("NPU attention finalizer differs from BF16 reference")
            (out / "attention" / "projection_input.bin").write_bytes(
                projection_input.tobytes())
        expected = sink_attention(q_gold, K_gold, V_gold, sinks,
                                  pos=pos).ravel().astype(np.float32)
    (out / "attention" / "expected.bin").write_bytes(expected.tobytes())
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    max_abs = float(np.max(np.abs(a - e)))
    max_rel = float(max_abs / max(np.max(np.abs(e)), 1e-30))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    report = {"passed": bool(max_rel < 0.005 and cosine > 0.9999),
              "position": pos, "mode": "full-state" if full_state_build else "sliding-128",
              "attention_rows": len(kv), "projection_calls": len(projection_reports),
              "projection_max_relative_error": max(r["max_relative_error"] for r in projection_reports),
              "attention_max_absolute_error": max_abs,
              "attention_max_relative_error": max_rel,
              "attention_cosine": cosine,
              "finalize_exact_bf16_fraction": finalize_exact,
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
    p.add_argument("--full-state-build", type=Path)
    p.add_argument("--finalize-build", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--pos", type=int, default=127)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.qkv_build.resolve(), a.rope_build.resolve(),
                         a.attn_build.resolve(), a.out.resolve(), a.pos,
                         full_state_build=a.full_state_build.resolve()
                         if a.full_state_build else None,
                         finalize_build=a.finalize_build.resolve()
                         if a.finalize_build else None), indent=2))


if __name__ == "__main__":
    main()
