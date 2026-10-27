"""Check GPT-OSS source attention against decoded q4 and FP64 sink attention."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from designs.gptoss_qkv_gemv.check import fixture  # noqa: E402
from designs.gptoss_rope.check import CONFIG  # noqa: E402
from gptoss_attention_run import run  # noqa: E402
from gptoss_qkv_run import CALLS  # noqa: E402
from model.replica_dense import rope  # noqa: E402
from model.replica_gptoss import sink_attention  # noqa: E402
from recipes.gptoss_source import GptOssSource  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402


def check(driver: Path, qkv_build: Path, rope_build: Path, state_build: Path,
          finalize_build: Path, out: Path) -> dict:
    rng = np.random.default_rng(20261029)
    activation = rng.normal(0, 0.5, 2880).astype(np.float32).astype(bfloat16)
    sinks = rng.normal(0, 1, 64).astype(np.float32).astype(bfloat16)
    chunks = {}
    biases = {}
    expected = {}
    for role, starts in CALLS.items():
        biases[role] = np.empty(4096 if role == "q" else 512, bfloat16)
        pieces = []
        for start in starts:
            packed, selected, reference = fixture(role, start, 16, activation)
            chunks[role, start] = selected
            biases[role][start * 32:(start + 16) * 32] = packed[2944:]
            pieces.append(reference)
        expected[role] = np.concatenate(pieces).astype(bfloat16)

    class FixtureSource(GptOssSource):
        def __init__(self):
            pass

        def constant(self, layer, name):
            return sinks if name == "sinks" else biases[name[0]]

        def attention(self, layer, role, start):
            return chunks[role, start]

    spec = ModelSpec.from_hf_config(CONFIG)
    cases = []
    for layer, pos, mode in ((1, 0, "full"), (0, 128, "sliding")):
        rows = pos if mode == "full" else min(pos, 127)
        previous_k = rng.normal(0, 0.4, (rows, 8, 64)).astype(np.float32).astype(bfloat16)
        previous_v = rng.normal(0, 0.5, (rows, 8, 64)).astype(np.float32).astype(bfloat16)
        case_out = out / f"{mode}-{pos}"
        run_report = run(driver, qkv_build, rope_build, state_build,
                         finalize_build, FixtureSource(), CONFIG, layer, pos,
                         activation, previous_k, previous_v, case_out)
        inv = np.asarray(spec.rope_inv_freq(ctx=pos + 1), np.float64)
        scale = spec.rope_scale()
        q = rope(expected["q"].reshape(64, 64), pos, 64, spec.rope_theta,
                 inv, scale).astype(bfloat16)
        k = rope(expected["k"].reshape(8, 64), pos, 64, spec.rope_theta,
                 inv, scale).astype(bfloat16)
        v = expected["v"].reshape(8, 64)
        K = np.concatenate((previous_k, k[None]), axis=0)
        V = np.concatenate((previous_v, v[None]), axis=0)
        reference = sink_attention(q, K, V, sinks,
                                   sliding_window=128 if mode == "sliding" else None,
                                   pos=len(K) - 1).ravel().astype(bfloat16).astype(np.float32)
        actual = np.fromfile(case_out / "attention.bf16", bfloat16).astype(np.float32)
        key = np.fromfile(case_out / "key.bf16", bfloat16)
        value = np.fromfile(case_out / "value.bf16", bfloat16)
        if actual.size != 4096 or key.size != 512 or value.size != 512:
            raise AssertionError("incomplete GPT-OSS attention output")
        error = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))
        scale_ref = max(float(np.max(np.abs(reference))), 1e-20)
        case = {"mode": mode, "position": pos,
                "kernel_calls": sum(run_report[name] for name in
                                    ("qkv_calls", "rope_calls", "attention_state_calls",
                                     "finalize_calls")),
                "attention_max_relative_error": error / scale_ref,
                "key_exact_bf16_fraction": float(np.mean(key == k.ravel())),
                "value_exact_bf16_fraction": float(np.mean(value == v.ravel())),
                "passed": bool(error / scale_ref < 0.005 and
                               np.mean(key == k.ravel()) > 0.99 and
                               np.array_equal(value, v.ravel()))}
        cases.append(case)
    report = {"cases": cases, "passed": all(case["passed"] for case in cases)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "qkv-build", "rope-build", "state-build", "finalize-build", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.driver.resolve(), a.qkv_build.resolve(), a.rope_build.resolve(),
                           a.state_build.resolve(), a.finalize_build.resolve(),
                           a.out.resolve()), indent=2))
