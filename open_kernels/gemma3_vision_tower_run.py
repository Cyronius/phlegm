"""Run the Gemma 3 4B SigLIP tower and projector on Phlegm's NPU images.

Input is FLM's preprocessed BF16 CHW[3,896,896] image and vision_weight.q4nx.
The runner keeps one layer's scratch files and two alternating hidden-state
files, so 27 layers do not multiply the scratch storage requirement.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.gemma3_vision_attention_check import maxrel, run_image
from open_kernels.gemma3_vision_layer_run import layernorm_ref, run as run_layer
from open_kernels.gemma3_vision_patch_pack import pack_pixels, pad_weight
from open_kernels.gemma3_vision_pool_pack import pack as pack_groups
from open_kernels.gemma3_vision_source import (patch_embedding, post_layernorm,
                                                projector, validate_header)
from open_kernels.gemma4_mm_source import npu_weight_row_major
from open_kernels.model.q4nx import Q4NX

VISION_CONFIG = {
    "vision_model_weight": "vision_weight.q4nx",
    "vision_conv2d_stride": 14,
    "vision_conv2d_padding": 0,
    "vision_conv2d_kernel": 14,
    "vision_conv2d_Cin": 3,
    "vision_conv2d_Cout": 1152,
    "vision_average_pooling_kernel": 4,
    "vision_average_pooling_stride": 4,
    "vision_average_pooling_padding": 0,
    "vision_layer_norm_eps": 1e-6,
    "vision_rms_norm_eps": 1e-6,
    "vision_intermediate_size": 4304,
    "vision_hidden_size": 1152,
    "vision_head_dim": 72,
    "vision_num_attention_heads": 16,
    "vision_num_key_value_heads": 16,
    "vision_num_hidden_layers": 27,
}


def validate_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    for key, expected in VISION_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"{key}: expected {expected!r}, got {config.get(key)!r}")
    return {key: config[key] for key in VISION_CONFIG}


def read_bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(path, bfloat16)
    if value.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected BF16{shape}")
    return value.reshape(shape)


def validate_images(patch: Path, tower: Path, post: Path, attention: Path,
                    residual: Path, pool: Path, projector_image: Path) -> int:
    images = ([patch / name for name in ("patch_mm_m256_k640_n1280",
                                         "patch_bias_position_256x1152")] +
              [tower / name for name in ("layernorm_256x1152", "qkv_m256_k1152_n3584",
                                         "attention_out_m256_k1152_n1280",
                                         "mlp_fc1_m256_k1152_n4352",
                                         "mlp_fc2_m256_k4352_n1280")] +
              [post / name for name in ("qkv_bias_n3584", "attention_out_bias_n1280",
                                        "fc1_bias_gelu_n4352", "fc2_bias_n1280")] +
              [attention / name for name in ("qk_m256_k128_n4096",
                                             "softmax_256x4096", "pv_m256_k4096_n256")] +
              [residual, pool, projector_image])
    for image in images:
        if not all((image / part).is_file() for part in ("final.xclbin", "insts.bin")):
            raise FileNotFoundError(f"missing NPU image: {image}")
    return len(images)


def patch_run(driver: Path, bundle: Path, image: np.ndarray, weights: dict,
              out: Path) -> Path:
    pixels = pack_pixels(image)
    tiled_weight = npu_weight_row_major(pad_weight(weights["weight"]), 1280, 640)
    hidden = np.empty((4096, 1152), bfloat16)
    for start in range(0, 4096, 256):
        projected_path = run_image(driver, bundle / "patch_mm_m256_k640_n1280",
                                   out / "matrix", [("patches", pixels[start:start + 256]),
                                   ("weight", tiled_weight)], 256 * 1280 * 4)
        projected = np.fromfile(projected_path, np.float32).reshape(256, 1280)
        addends = np.empty((256, 2, 1152), bfloat16)
        addends[:, 0] = weights["bias"]
        addends[:, 1] = weights["position"][start:start + 256]
        result_path = run_image(driver, bundle / "patch_bias_position_256x1152",
                                out / "finish", [("projected", projected),
                                ("addends", addends)], 256 * 1152 * 2)
        hidden[start:start + 256] = read_bf16(result_path, (256, 1152))
    path = out / "result.bin"
    path.write_bytes(hidden.tobytes())
    return path


def finish_run(driver: Path, layernorm_image: Path, pool_image: Path,
               projector_image: Path, hidden_path: Path, weights: Q4NX,
               out: Path) -> dict:
    hidden = read_bf16(hidden_path, (4096, 1152))
    gamma, beta = post_layernorm(weights)
    params = np.stack((gamma, beta))
    normalized = np.empty_like(hidden)
    for start in range(0, 4096, 256):
        result_path = run_image(driver, layernorm_image, out / "post_layernorm",
                                [("input", hidden[start:start + 256]),
                                 ("params", params)], 256 * 1152 * 2)
        normalized[start:start + 256] = read_bf16(result_path, (256, 1152))
    norm_error = max(maxrel(normalized[row], layernorm_ref(hidden[row], gamma, beta))
                     for row in (0, 257, 4095))

    pool_weight, matrix = projector(weights)
    groups = pack_groups(normalized)
    pool_path = run_image(driver, pool_image, out / "pool",
                          [("groups", groups), ("weight", pool_weight)], 256 * 1152 * 2)
    pooled = read_bf16(pool_path, (256, 1152))
    mean = (groups.astype(np.float64).sum(axis=1) / 16).astype(bfloat16)
    x = mean.astype(np.float64)
    expected_pool = (x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6) *
                     (1 + pool_weight.astype(np.float64))).astype(bfloat16)
    pool_error = maxrel(pooled, expected_pool)

    tiled_matrix = npu_weight_row_major(matrix, 2560, 1152)
    projected_path = run_image(driver, projector_image, out / "projector",
                               [("activation", pooled), ("weight", tiled_matrix)],
                               256 * 2560 * 4)
    projected = np.fromfile(projected_path, np.float32).reshape(256, 2560)
    reference = (pooled[[0, 127, 255]].astype(np.float64) @
                 matrix.astype(np.float64).T).astype(np.float32)
    projector_error = maxrel(projected[[0, 127, 255]], reference)
    (out / "result.bf16").write_bytes(projected.astype(bfloat16).tobytes())
    return {"post_layernorm_max_relative_error": norm_error,
            "pool_max_relative_error": pool_error,
            "projector_sample_max_relative_error": projector_error,
            "finite": bool(np.isfinite(projected).all()),
            "passed": bool(np.isfinite(projected).all() and norm_error < 0.005 and
                           pool_error < 0.005 and projector_error < 0.005)}


def run(driver: Path, patch_bundle: Path, tower_bundle: Path, post_bundle: Path,
        attention_bundle: Path, residual_image: Path, pool_image: Path,
        projector_image: Path, vision_weights: Path, config_path: Path, image_path: Path,
        out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    config = validate_config(config_path)
    validate_images(patch_bundle, tower_bundle, post_bundle, attention_bundle,
                    residual_image, pool_image, projector_image)
    image = read_bf16(image_path, (3, 896, 896))
    weights = Q4NX(vision_weights)
    validate_header(weights)
    patch_weights = patch_embedding(weights)
    out.mkdir(parents=True)
    first = patch_run(driver, patch_bundle, image, patch_weights, out / "patch")
    current = out / "hidden_a.bin"
    next_hidden = out / "hidden_b.bin"
    shutil.copyfile(first, current)
    reports = []
    for index in range(27):
        report = run_layer(driver, tower_bundle, post_bundle, attention_bundle,
                           residual_image, out / "scratch", vision_weights,
                           current, index)
        reports.append({"layer": index, **report})
        (out / "layers.json").write_text(json.dumps(reports, indent=2) + "\n")
        shutil.copyfile(out / "scratch" / "result.bin", next_hidden)
        current, next_hidden = next_hidden, current
    finish = finish_run(driver, tower_bundle / "layernorm_256x1152",
                        pool_image, projector_image, current, weights,
                        out / "finish")
    report = {"model": "Gemma 3 4B SigLIP tower", "configuration": config,
              "layers": len(reports),
              "tokens": 256, "hidden": 2560, "finish": finish,
              "passed": bool(all(layer["passed"] for layer in reports) and finish["passed"])}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("driver", "patch-bundle", "tower-bundle", "post-bundle",
                "attention-bundle", "residual-image", "pool-image",
                "projector-image", "vision-weights", "config", "image", "out"):
        p.add_argument(f"--{key}", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(*(getattr(a, key.replace("-", "_")).resolve() for key in
                           ("driver", "patch-bundle", "tower-bundle", "post-bundle",
                            "attention-bundle", "residual-image", "pool-image",
                            "projector-image", "vision-weights", "config", "image", "out"))),
                     indent=2))
