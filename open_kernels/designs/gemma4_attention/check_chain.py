"""Check composed Gemma 4 text Q/K/V norm, RoPE and decode attention on NPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from open_kernels.designs.gemma4_rope.check import table_for  # noqa: E402

GEOMETRIES = {
    "e2b-local": (8, 1, 256), "e2b-full": (8, 1, 512),
    "e4b-local": (8, 2, 256), "e4b-full": (8, 2, 512),
    "12b-local": (16, 8, 256), "12b-full": (16, 1, 512),
}
ROWS = 128
POSITION = 127


def invoke(driver: Path, image: Path, stage: Path,
           inputs: list[np.ndarray], out_nbytes: int, dtype) -> np.ndarray:
    stage.mkdir(parents=True, exist_ok=False)
    lines = ["device", f"xclbin G {image / 'final.xclbin'}",
             f"kernelx k G {image / 'insts.bin'}"]
    names = []
    for i, arr in enumerate(inputs):
        name = f"in{i}"
        path = stage / f"{name}.bin"
        path.write_bytes(np.ascontiguousarray(arr).tobytes())
        lines.append(f"buf {name} {path.stat().st_size} {path}")
        names.append(name)
    lines += [f"buf result {out_nbytes}", f"run k {' '.join(names)} result",
              f"dump result {stage / 'result.bin'} {out_nbytes}", ""]
    config = stage / "run.cfg"
    config.write_text("\n".join(lines))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{stage.name}: {proc.stdout}\n{proc.stderr}")
    result = np.fromfile(stage / "result.bin", dtype=dtype)
    if result.nbytes != out_nbytes or not np.isfinite(result.astype(np.float32)).all():
        raise AssertionError(f"{stage.name}: invalid output")
    return result


def norm_reference(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64)
    inv = np.power(np.mean(z * z, axis=-1, keepdims=True) + 1e-6, -0.5)
    return (z * inv * scale.astype(np.float64)).astype(bfloat16)


def rope_reference(qk: np.ndarray, table: np.ndarray) -> np.ndarray:
    z, t = qk.astype(np.float64), table.astype(np.float64)
    half = z.shape[-1] // 2
    out = np.empty_like(z)
    out[:, :half] = z[:, :half] * t[:half] - z[:, half:] * t[half:]
    out[:, half:] = z[:, half:] * t[:half] + z[:, :half] * t[half:]
    return out.astype(bfloat16)


def metric(actual: np.ndarray, expected: np.ndarray, threshold: float) -> dict:
    a, e = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    if a.shape != e.shape:
        raise AssertionError(f"shape mismatch: {a.shape} versus {e.shape}")
    maximum = float(np.max(np.abs(a - e)))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    return {"max_absolute_error": maximum, "cosine": cosine,
            "passed": bool(maximum <= threshold and cosine > 0.99999)}


def run(driver: Path, kernels: Path, out: Path, geometry: str) -> dict:
    nh, kvh, width = GEOMETRIES[geometry]
    rng = np.random.default_rng(20261027 + nh * width + kvh)
    projected_q = rng.normal(0, 0.7, (nh, width)).astype(np.float32).astype(bfloat16)
    projected_k = rng.normal(0, 0.7, (kvh, width)).astype(np.float32).astype(bfloat16)
    projected_v = (projected_k.copy() if geometry == "12b-full" else
                   rng.normal(0, 0.7, (kvh, width)).astype(np.float32).astype(bfloat16))
    q_scale = rng.uniform(0.7, 1.3, width).astype(np.float32).astype(bfloat16)
    k_scale = rng.uniform(0.7, 1.3, width).astype(np.float32).astype(bfloat16)
    v_scale = np.ones(width, bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    image = lambda heads: kernels / f"gemma4-head-norm-{heads}-{width}"  # noqa: E731
    q = invoke(driver, image(nh), out / "01-q-norm", [projected_q, q_scale],
               projected_q.nbytes, bfloat16).reshape(nh, width)
    k = invoke(driver, image(kvh), out / "02-k-norm", [projected_k, k_scale],
               projected_k.nbytes, bfloat16).reshape(kvh, width)
    v = invoke(driver, image(kvh), out / "03-v-norm", [projected_v, v_scale],
               projected_v.nbytes, bfloat16).reshape(kvh, width)
    q_ref = norm_reference(projected_q, q_scale)
    k_ref = norm_reference(projected_k, k_scale)
    v_ref = norm_reference(projected_v, v_scale)
    stages = {"q_norm": metric(q, q_ref, 0.015),
              "k_norm": metric(k, k_ref, 0.015),
              "v_norm": metric(v, v_ref, 0.015)}
    table = table_for(POSITION, width)
    qk = np.concatenate((q, k))
    rotated = invoke(driver, kernels / f"gemma4-rope-{geometry}", out / "04-rope",
                     [qk, table], qk.nbytes, bfloat16).reshape(nh + kvh, width)
    rotated_ref = rope_reference(np.concatenate((q_ref, k_ref)), table)
    stages["rope"] = metric(rotated, rotated_ref, 0.015)
    old_k = rng.normal(0, 0.7, (ROWS - 1, kvh, width)).astype(np.float32).astype(bfloat16)
    old_v = rng.normal(0, 0.7, (ROWS - 1, kvh, width)).astype(np.float32).astype(bfloat16)
    cache_k = np.concatenate((old_k, rotated[nh:][None]), axis=0)
    cache_v = np.concatenate((old_v, v[None]), axis=0)
    cache_k_ref = np.concatenate((old_k, rotated_ref[nh:][None]), axis=0)
    cache_v_ref = np.concatenate((old_v, v_ref[None]), axis=0)
    output = np.empty((nh, width), np.float32)
    expected = np.empty((nh, width), np.float64)
    conditioned = np.empty((nh, width), np.float64)
    group = nh // kvh
    for h in range(nh):
        selected = h // group
        kv_rows = np.stack((cache_k[:, selected], cache_v[:, selected]), axis=1)
        blank = np.zeros(2 * width + 16, np.float32)
        meta = np.frombuffer(rotated[h].tobytes() + blank.tobytes(), np.uint8)
        state = invoke(driver, kernels / f"gemma4-attention-{width}-{ROWS}",
                       out / f"05-attention-h{h:02d}", [meta, kv_rows], blank.nbytes,
                       np.float32)
        output[h] = state[width + 16:]
        scores = cache_k_ref[:, selected].astype(np.float64) @ rotated_ref[h].astype(np.float64)
        weights = np.exp(scores - scores.max())
        expected[h] = weights @ cache_v_ref[:, selected].astype(np.float64) / weights.sum()
        scores = cache_k[:, selected].astype(np.float64) @ rotated[h].astype(np.float64)
        weights = np.exp(scores - scores.max())
        conditioned[h] = weights @ cache_v[:, selected].astype(np.float64) / weights.sum()
    stages["attention_from_npu_inputs"] = metric(output, conditioned.astype(np.float32), 0.001)
    stages["attention_end_to_end"] = metric(output, expected.astype(np.float32), 0.015)
    report = {"geometry": geometry, "position": POSITION, "rows": ROWS,
              "query_heads": nh, "kv_heads": kvh, "head_dim": width,
              "passed": all(s["passed"] for s in stages.values()), "stages": stages}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--kernels", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--geometry", choices=GEOMETRIES, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.kernels.resolve(), a.out.resolve(),
                         a.geometry), indent=2))


if __name__ == "__main__":
    main()
