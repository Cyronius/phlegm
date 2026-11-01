"""Run Qwen Image 2.1 text/image/timestep entry or final latent exit on NPU.

The entry consumes 15 Qwen3-VL embeddings, 1024 x 64 latents, and one flow
timestep. It emits embedded text/image rows, shared 4-way modulation for
t=0 and the requested timestep, and final-norm scales. The final stage takes
the image rows after all transformer blocks and emits 1024 x 64 velocities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.qwen21_block_run import HIDDEN, IMAGE, TEXT, HarnessStage, bf, tensor
from open_kernels.qwen21_image_edges import specs

CHUNK = 512


def _weight(weight_root: Path, manifest: dict, slot: str) -> Path:
    record = manifest["slots"][slot]
    path = weight_root / record["file"]
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"{slot}: packed weight byte count differs")
    return path


def _kernels(kernel_root: Path) -> dict[str, Path]:
    manifest = json.loads((kernel_root / "manifest.json").read_text())
    if not manifest.get("complete") or manifest["family"] != "Qwen/Qwen-Image-2.1":
        raise ValueError("Qwen Image 2.1 edge kernel set is incomplete")
    if manifest["streams"] != specs():
        raise ValueError("Qwen Image 2.1 edge specs changed since build")
    return {name: kernel_root / name for family in manifest["streams"].values()
            for name in family}


def _timestep(value: float) -> np.ndarray:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("flow timestep must lie in [0, 1]")
    freqs = np.exp(-np.log(10000.0) * np.arange(128, dtype=np.float32) / 128)
    phase = 1000 * np.array([value, 0.0], dtype=np.float32)[:, None] * freqs[None]
    encoded = np.concatenate((np.cos(phase), np.sin(phase)), -1)
    result = np.zeros((CHUNK, 512), bfloat16)
    result[:2, :256] = bf(encoded)
    return result


def entry(args) -> None:
    model, weights, out = args.model.resolve(), args.weights.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    packed = json.loads((weights / "packed_weights.json").read_text())
    if packed["model"] != "Qwen-Image-2.1":
        raise ValueError("wrong packed transformer family")
    rng = np.random.default_rng(args.seed)
    text = (np.fromfile(args.text_bf16, bfloat16).reshape(-1, HIDDEN) if args.text_bf16 else
            bf(rng.normal(0, 0.6, (TEXT, HIDDEN))))
    text_tokens = len(text)
    if not 1 <= text_tokens <= 64:
        raise ValueError("Qwen entry needs 1..64 text tokens")
    latent = (np.fromfile(args.latent_bf16, bfloat16).reshape(IMAGE, 64) if args.latent_bf16 else
              bf(rng.normal(0, 1.0, (IMAGE, 64))))
    text.tofile(out / "input-text.bf16")
    latent.tofile(out / "input-latent.bf16")
    text_scale = 1 + tensor(model, "txt_in.text_norm.weight").astype(np.float32)
    rms_param = np.concatenate((np.zeros(HIDDEN, bfloat16), bf(text_scale),
                                np.zeros(HIDDEN, bfloat16)))
    dirs = _kernels(args.kernels.resolve())
    s = HarnessStage(out, "01-entry", dirs, args.harness.resolve())
    for name, slot in (("w_img", "emb.image"), ("w_text_in", "text.in"),
                       ("w_text_out", "text.out"), ("w_time_in", "time.in"),
                       ("w_time_out", "time.out"), ("w_mod", "mod"),
                       ("w_final_mod", "final.mod")):
        s.buf(name, path=_weight(weights, packed, slot))
    padded_text = np.zeros((CHUNK, HIDDEN), bfloat16)
    padded_text[:text_tokens] = text
    s.buf("text", padded_text)
    s.buf("rms_param", rms_param)
    s.buf("dummy", shape=(HIDDEN,))
    for name in ("text_norm", "text_in", "text_gelu", "text_out"):
        s.buf(name, shape=(CHUNK, HIDDEN))
    s.run("text_rms", "text", "dummy", "rms_param", "text_norm", "dummy")
    s.run("hidden_to_hidden", "text_norm", "w_text_in", "text_in")
    s.run("gelu_tanh", "text_in", "dummy", "dummy", "text_gelu", "dummy")
    s.run("hidden_to_hidden", "text_gelu", "w_text_out", "text_out")
    s.dump("text_out", (CHUNK, HIDDEN))
    for i in range(2):
        source = np.zeros((CHUNK, 512), bfloat16)
        source[:, :64] = latent[i * CHUNK:(i + 1) * CHUNK]
        s.buf(f"latent{i}", source)
        s.buf(f"image{i}", shape=(CHUNK, HIDDEN))
        s.run("short_to_hidden", f"latent{i}", "w_img", f"image{i}")
        s.dump(f"image{i}", (CHUNK, HIDDEN))
    s.buf("time", _timestep(args.timestep))
    for name in ("time_in", "time_act", "time_out", "time_mod_act", "final_scale"):
        s.buf(name, shape=(CHUNK, HIDDEN))
    s.buf("mod", shape=(CHUNK, 4 * HIDDEN))
    s.run("short_to_hidden", "time", "w_time_in", "time_in")
    s.run("silu", "time_in", "dummy", "dummy", "time_act", "dummy")
    s.run("hidden_to_hidden", "time_act", "w_time_out", "time_out")
    s.run("silu", "time_out", "dummy", "dummy", "time_mod_act", "dummy")
    s.run("hidden_to_mod", "time_mod_act", "w_mod", "mod")
    s.run("hidden_to_hidden", "time_mod_act", "w_final_mod", "final_scale")
    s.dump("mod", (CHUNK, 4 * HIDDEN))
    s.dump("final_scale", (CHUNK, HIDDEN))
    result = s.execute()
    outputs = {
        "text": result["text_out"][:text_tokens],
        "image": np.concatenate((result["image0"], result["image1"])),
        "image-mod": result["mod"][0].reshape(4, HIDDEN),
        "text-mod": result["mod"][1].reshape(4, HIDDEN),
        "image-final-scale": result["final_scale"][0],
        "text-final-scale": result["final_scale"][1],
    }
    for name, value in outputs.items():
        if not np.isfinite(value.astype(np.float32)).all():
            raise ValueError(f"{name}: NPU entry returned NaN/Inf")
        value.tofile(out / f"native-{name}.bf16")
        print(f"{name}: {value.shape}, RMS {np.sqrt(np.mean(value.astype(np.float32)**2)):.6g}",
              flush=True)
    (out / "fixture.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "stage": "entry", "resolution": 512,
        "timestep": args.timestep, "seed": args.seed, "text_tokens": text_tokens,
        "operations": 12}, indent=2) + "\n")


def final(args) -> None:
    weights, out = args.weights.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    packed = json.loads((weights / "packed_weights.json").read_text())
    image = np.fromfile(args.image_bf16, bfloat16).reshape(IMAGE, HIDDEN)
    scale = np.fromfile(args.scale_bf16, bfloat16)
    if scale.size != HIDDEN:
        raise ValueError("final image scale must have 4096 elements")
    image.tofile(out / "input-image.bf16")
    scale.tofile(out / "input-scale.bf16")
    norm_param = np.concatenate((np.zeros(HIDDEN, bfloat16),
                                 np.zeros(HIDDEN, bfloat16), scale,
                                 np.zeros(HIDDEN, bfloat16)))
    dirs = _kernels(args.kernels.resolve())
    s = HarnessStage(out, "01-final", dirs, args.harness.resolve())
    s.buf("w_final", path=_weight(weights, packed, "final.image"))
    s.buf("norm_param", norm_param)
    s.buf("dummy", shape=(HIDDEN,))
    for i in range(2):
        s.buf(f"image{i}", image[i * CHUNK:(i + 1) * CHUNK])
        s.buf(f"norm{i}", shape=(CHUNK, HIDDEN))
        s.buf(f"output{i}", shape=(CHUNK, 1024))
        s.run("final_ln", f"image{i}", "dummy", "norm_param", f"norm{i}", "dummy")
        s.run("hidden_to_latent", f"norm{i}", "w_final", f"output{i}")
        s.dump(f"output{i}", (CHUNK, 1024))
    result = s.execute()
    velocity = np.concatenate((result["output0"][:, :64], result["output1"][:, :64]))
    if not np.isfinite(velocity.astype(np.float32)).all():
        raise ValueError("native final velocity contains NaN/Inf")
    velocity.tofile(out / "native-velocity.bf16")
    print(f"velocity: {velocity.shape}, RMS "
          f"{np.sqrt(np.mean(velocity.astype(np.float32)**2)):.6g}")
    (out / "fixture.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "stage": "final", "resolution": 512,
        "operations": 4}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("entry", "final"):
        p = sub.add_parser(name)
        p.add_argument("--model", type=Path, required=name == "entry")
        p.add_argument("--weights", type=Path, required=True)
        p.add_argument("--kernels", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--harness", type=Path, default=(
            Path(__file__).resolve().parents[1] / "target/npu-dev/harness-build/run_kernel"))
        if name == "entry":
            p.add_argument("--text-bf16", type=Path)
            p.add_argument("--latent-bf16", type=Path)
            p.add_argument("--timestep", type=float, default=1.0)
            p.add_argument("--seed", type=int, default=37)
        else:
            p.add_argument("--image-bf16", type=Path, required=True)
            p.add_argument("--scale-bf16", type=Path, required=True)
    args = parser.parse_args()
    entry(args) if args.stage == "entry" else final(args)


if __name__ == "__main__":
    main()
