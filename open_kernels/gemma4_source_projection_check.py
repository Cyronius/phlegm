"""Check Gemma 4 q4nx source-band packing against a full NPU projection.

Uses deterministic source-format weights, so no model payload is needed.
The 12B reference decodes signed nibbles directly from its 16-row source.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma4_lm_head_check import fixture
from open_kernels.gemma4_source import source_shape, tensor_name
from open_kernels.gemma4_text_plan import from_hf_config, projections
from open_kernels.gemma4_text_source import Gemma4TextSource


class _Reader:
    def __init__(self, name, dtype, shape, payload):
        self.tensors = {name: {"dtype": dtype, "shape": shape,
                               "data_offsets": [0, len(payload)]}}
        self.data_base = 0
        self.mm = payload


def _signed_band(rng, cols, activation):
    kt = cols // 256
    source = np.empty((4, kt, 2304), np.uint8)
    scales = rng.uniform(0.002, 0.035, (4, kt, 8, 16)).astype(np.float32).astype(bfloat16)
    source[..., :256] = scales.view(np.uint8).reshape(4, kt, 256)
    source[..., 256:] = rng.integers(0, 256, (4, kt, 2048), dtype=np.uint8)
    expected = np.zeros(64, np.float64)
    x = activation.astype(np.float64).reshape(kt, 256)
    for half in range(4):
        for tile in range(kt):
            code = source[half, tile, 256:].reshape(256, 8)
            for row in range(16):
                nib = code[:, row // 2] & 15 if row % 2 == 0 else code[:, row // 2] >> 4
                signed = ((nib.astype(np.int16) ^ 8) - 8).astype(np.float64)
                scale = scales[half, tile, :, row].astype(np.float64).repeat(32)
                expected[half * 16 + row] += (signed * scale) @ x[tile]
    return source, expected.astype(np.float32)


def run(driver: Path, bundle: Path, config: dict, out: Path) -> dict:
    plan = from_hf_config(config)
    projection = next(p for p in projections(plan) if p.layer == 0 and p.name == "q")
    manifest = json.loads((bundle / "projections" / "manifest.json").read_text())
    if manifest["variant"] != plan.variant:
        raise ValueError("Gemma 4 bundle variant differs from config")
    entry = next(e for e in manifest["projections"] if e["layer"] == 0 and e["name"] == "q")
    if (entry["rows"], entry["cols"], entry["calls"]) != (projection.rows, projection.cols, 1):
        raise ValueError("expected an unsplit layer-zero Q projection")
    image = bundle / "projections" / entry["image"]
    for path in (driver, image / "final.xclbin", image / "insts.bin"):
        if not path.is_file():
            raise FileNotFoundError(path)
    rng = np.random.default_rng(20261027)
    x = rng.normal(0, 0.5, projection.cols).astype(np.float32).astype(bfloat16)
    source_bands = []
    expected_bands = []
    pool_bands = []
    if plan.variant == "12B":
        for _ in range(projection.rows // 64):
            source, expected = _signed_band(rng, projection.cols, x)
            source_bands.append(source)
            expected_bands.append(expected)
    else:
        for band in range(projection.rows // 64):
            pool, _, expected = fixture(projection.cols, 20261027 + band, x)
            source_bands.append(pool.reshape(projection.cols // 256, 2, 5120)
                                .transpose(1, 0, 2).copy())
            pool_bands.append(pool)
            expected_bands.append(expected)
    dtype, shape = source_shape(plan, projection)
    source = np.concatenate(source_bands, axis=0)
    if source.size != int(np.prod(shape)):
        raise AssertionError(f"source byte count differs from {shape}")
    source = source.reshape(shape)
    adapter = Gemma4TextSource(_Reader(tensor_name(projection), dtype, shape,
                                       source.tobytes()), plan)
    packed = np.concatenate([adapter.projection_band(projection, band)
                             for band in range(projection.rows // 64)])
    if pool_bands and not np.array_equal(packed, np.concatenate(pool_bands)):
        raise AssertionError("source adapter changed a q4_1 chunk")
    expected = np.concatenate(expected_bands)
    out.mkdir(parents=True, exist_ok=False)
    (out / "weights.bin").write_bytes(packed.tobytes())
    (out / "activation.bin").write_bytes(x.tobytes())
    cfg = out / "run.cfg"
    cfg.write_text("\n".join([
        "device", f"xclbin G {image / 'final.xclbin'}",
        f"kernelx k G {image / 'insts.bin'}",
        f"buf weights {packed.nbytes} {out / 'weights.bin'}",
        f"buf activation {x.nbytes} {out / 'activation.bin'}",
        f"buf result {expected.nbytes}", "run k weights activation result",
        f"dump result {out / 'result.bin'} {expected.nbytes}", "",
    ]))
    proc = subprocess.run((str(driver), str(cfg)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU projection failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "result.bin", np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("invalid Gemma 4 NPU projection output")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    relative = error / max(float(np.max(np.abs(expected))), 1e-20)
    report = {"variant": plan.variant, "projection": tensor_name(projection),
              "rows": projection.rows, "cols": projection.cols,
              "max_absolute_error": error, "max_relative_error": relative,
              "passed": bool(relative < 0.005),
              "reference": "FP64 matvec from independently decoded source chunks"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.bundle.resolve(),
                         json.loads(a.config.read_text()), a.out.resolve()), indent=2))
