"""NPU gate: capture native Klein conditioning, replay it, and check request isolation.

Requires the same full checkpoint, pyxrt environment and kernel set as
flux2_native_generate. Writes captures under a fresh --out directory. This is a
correctness gate, with no benchmark or generated-image quality claim.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flux2_native_generate as fg  # noqa: E402
import klein_conditioning as kc  # noqa: E402
import klein_pipeline as kp  # noqa: E402


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_capture(path: Path, bits: np.ndarray, ids: np.ndarray, real: int) -> None:
    from safetensors.numpy import save

    positions = np.zeros((kc.POSITIONS, 4), np.int64)
    positions[:, 3] = np.arange(kc.POSITIONS)
    mask = (np.arange(kc.POSITIONS) < real).astype(np.int64)
    tensors = {"projected_conditioning": np.ascontiguousarray(bits, np.uint16).view(bfloat16),
               "input_ids": np.ascontiguousarray(ids, np.int64), "attention_mask": mask,
               "text_ids": positions}
    data = save(tensors, metadata={"protocol": kc.PROTOCOL, "source": "native NPU context_embedder"})
    with path.open("xb") as writer:
        writer.write(data)


def capture_first_projection(runner, receive):
    """Wait/read immediately after ctx_emb, before later operations can mutate X."""
    from npu_host import COMPLETED

    for index, (kset, stream, bos, phase, op) in enumerate(runner.ops):
        if op["what"] != "ctx_emb":
            continue

        class Capture:
            name = stream.name

            def start(self, *args):
                handle = stream.start(*args)
                if handle.wait() != COMPLETED:
                    raise RuntimeError("native conditioning capture failed")
                bits = runner.bufs["X"].read(np.uint16, 0, kc.POSITIONS * kc.WIDTH)
                receive(bits.reshape(kc.POSITIONS, kc.WIDTH))
                return handle  # retain the original completed XRT handle in the queue

        original = runner.ops[index]
        runner.ops[index] = (kset, Capture(), bos, phase, op)
        return lambda: runner.ops.__setitem__(index, original)
    raise ValueError("normal runner has no context projection to capture")


def snapshot(runner, noise, out: Path, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    runner.set_noise(noise)
    phases = runner.run()
    latents, pixels = runner.latents(), runner.image()
    if not np.isfinite(latents.view(bfloat16)).all():
        raise ValueError(f"{name} produced nonfinite native latents")
    from PIL import Image

    np.save(out / (name + "-latents.npy"), latents)
    Image.fromarray(pixels).save(out / (name + ".png"))
    return latents, pixels, phases


def gate(args) -> dict:
    if args.out.exists():
        raise FileExistsError(args.out)
    if not 1 <= args.steps <= 50 or kp.check_resolution(args.size):
        raise ValueError("unsupported steps or resolution")
    geometry = fg.check_checkpoint(args.model)
    args.out.mkdir(parents=True)
    report = {"passed": False, "scope": "native capture/replay and input sensitivity; no image-quality gate",
              "model": str(args.model), "kernels": str(args.kernels), "checkpoint": geometry,
              "size": args.size, "steps": args.steps, "prompt": args.prompt, "seed": args.seed}
    try:
        sources = [Path(__file__), HERE / "flux2_native_generate.py", HERE / "klein_pipeline.py",
                   HERE / "klein_conditioning.py", HERE / "harness/npu_host.py"]
        report["source_sha256"] = {str(path): digest(path) for path in sources}
        weights = list(args.model.rglob("*.safetensors"))
        report["checkpoint_sha256"] = {str(path.relative_to(args.model)): digest(path) for path in weights}
        images = list(args.kernels.rglob("final.xclbin")) + list(args.kernels.rglob("insts*.bin"))
        report["kernel_sha256"] = {str(path.relative_to(args.kernels)): digest(path) for path in images}
        noise = np.random.default_rng(args.seed).standard_normal(
            (kp.LAT_CH, args.size // 16, args.size // 16), np.float32)
        np.save(args.out / "noise.npy", noise)
        report["noise_sha256"] = digest(args.out / "noise.npy")
        packed_noise = fg.pack_noise(noise)

        normal = fg.Runner(args.model, args.kernels, args.size, args.steps, False, args.pack_jobs)
        ids, real = normal.set_prompt(args.prompt)
        baseline, pixels, baseline_time = snapshot(normal, packed_noise, args.out, "baseline")
        captured = []
        restore = capture_first_projection(normal, lambda bits: captured.append(bits.copy()))
        try:
            normal.set_prompt(args.prompt)
            instrumented, instrumented_pixels, captured_time = snapshot(normal, packed_noise, args.out, "capture")
        finally:
            restore()
        report["capture_preserves_latents"] = bool(np.array_equal(baseline, instrumented))
        report["capture_preserves_pixels"] = bool(np.array_equal(pixels, instrumented_pixels))
        if len(captured) != 1 or not report["capture_preserves_latents"] or not report["capture_preserves_pixels"]:
            raise ValueError("native capture instrumentation changed outputs or capture count")
        capture_path = args.out / "native-conditioning.safetensors"
        save_capture(capture_path, captured[0], ids, real)
        conditioning = kc.load_projected_conditioning(capture_path)
        report["conditioning_sha256"] = conditioning.sha256
        report["native_phases_s"] = {"baseline": baseline_time, "capture": captured_time}
        del restore, normal
        gc.collect()  # release completed normal-route XRT contexts before the next runner

        candidate = fg.Runner(args.model, args.kernels, args.size, args.steps, False, args.pack_jobs, projected=True)
        candidate.set_projected_conditioning(conditioning, args.prompt)
        replay, replay_pixels, replay_time = snapshot(candidate, packed_noise, args.out, "replay")
        report["replay_latents_exact"] = bool(np.array_equal(baseline, replay))
        report["replay_pixels_exact"] = bool(np.array_equal(pixels, replay_pixels))
        report["native_phases_s"]["replay"] = replay_time

        zero_path = args.out / "zero-conditioning.safetensors"
        save_capture(zero_path, np.zeros_like(captured[0]), ids, real)
        candidate.set_projected_conditioning(kc.load_projected_conditioning(zero_path), args.prompt)
        zero, _, zero_time = snapshot(candidate, packed_noise, args.out, "zero-control")
        report["input_sensitivity"] = not np.array_equal(replay, zero)
        candidate.set_projected_conditioning(conditioning, args.prompt)
        repeated, repeated_pixels, repeated_time = snapshot(candidate, packed_noise, args.out, "restored")
        report["restored_latents_exact"] = bool(np.array_equal(baseline, repeated))
        report["restored_pixels_exact"] = bool(np.array_equal(pixels, repeated_pixels))
        report["native_phases_s"].update(zero_control=zero_time, restored=repeated_time)
        checks = [v for k, v in report.items() if k.endswith("_exact") or k.startswith("capture_preserves")
                  or k == "input_sensitivity"]
        report["passed"] = all(checks)
        report["output_sha256"] = {path.name: digest(path) for path in args.out.iterdir() if path.is_file()}
        return report
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "kernels", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prompt", default="A red fox in fresh snow")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pack-jobs", type=int, default=4)
    report = gate(parser.parse_args())
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
