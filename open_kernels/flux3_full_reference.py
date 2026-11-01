"""Compare one connected native FLUX 3 DiT pass with the published full model.

The text input is a deterministic synthetic 80-token conditioning tensor.
This is a DiT integration gate, not prompt encoding or sampled image quality.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def bf16(path: Path, rows: int, width: int) -> torch.Tensor:
    values = np.fromfile(path, "<u2")
    if values.size < rows * width:
        raise ValueError(f"{path}: insufficient BF16 input")
    return torch.from_numpy(values[:rows * width].reshape(rows, width).copy()).view(torch.bfloat16)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--time", type=float, default=0.8)
    parser.add_argument("--target", type=float, default=0.7)
    args = parser.parse_args()
    b = args.workspace.resolve()
    source = args.reference.resolve() / "src/flux3/dit.py"
    spec = importlib.util.spec_from_file_location("flux3_full_dit", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(8)
    model = module.DiT(flow_map=True).to(torch.bfloat16).eval()
    weights = load_file(str(args.model.resolve() / "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    del weights
    captures = {}
    def save(name):
        def hook(_module, _inputs, output):
            captures[name] = output.detach().float().cpu().numpy()
        return hook
    model.content_mode_blocks["image"][4].register_forward_hook(save("early_image"))
    model.txt_mode_blocks[4].register_forward_hook(save("early_text"))
    model.single_blocks[27].register_forward_hook(save("shared"))
    image = bf16(b / "flux3-image-entry-fixture-512/input.bf16", 1024, 128)
    text = bf16(b / "flux3-text-entry-fixture-512/input.bf16", 80, 20480)
    positions = torch.arange(1024, dtype=torch.float32)
    zeros = torch.zeros_like(positions)
    image_ids = torch.stack((zeros, (positions // 32) * 0.5,
                             (positions % 32) * 0.5, zeros), dim=-1)
    txt_pos = torch.arange(80, dtype=torch.float32)
    text_ids = torch.stack((torch.zeros_like(txt_pos), torch.zeros_like(txt_pos),
                            torch.zeros_like(txt_pos), txt_pos), dim=-1)
    t_image = torch.tensor(args.time, dtype=torch.bfloat16)
    packed = module.Packed(
        data={"image": image, "image_cond": torch.zeros((0, 128), dtype=torch.bfloat16),
              "text": text},
        ids={"image": image_ids, "image_cond": torch.zeros((0, 4)), "text": text_ids},
        timesteps={"image": torch.full((1024,), t_image, dtype=torch.bfloat16),
                   "image_cond": torch.zeros((0,), dtype=torch.bfloat16),
                   "text": torch.zeros((80,), dtype=torch.bfloat16)},
        mode_indices={"image": [(0, 1024)], "image_cond": [(0, 0)],
                      "text": [(0, 80)]},
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        velocity = model(packed, timesteps_ref=torch.tensor([args.target]))
    expected = velocity.float().numpy()
    if expected.shape != (1024, 128) or not np.isfinite(expected).all():
        raise ValueError("published FLUX 3 velocity has wrong shape or nonfinite values")
    path = b / "flux3-conditioned-final-fixture-512/reference-velocity.f32"
    expected.astype("<f4").tofile(path)
    native_raw = np.fromfile(b / "flux3-conditioned-final-fixture-512/native-output.bf16", "<u2")
    native = torch.from_numpy(native_raw.reshape(1024, 1024)[:, :128].copy()).view(torch.bfloat16)
    native = native.float().numpy()
    diff = native - expected
    print(f"connected FLUX 3 DiT pass: velocity RMS {np.sqrt(np.mean(native*native)):.6g}")
    print(f"native vs published BF16: relative L2 "
          f"{np.linalg.norm(diff)/np.linalg.norm(expected):.6g}, cosine "
          f"{np.vdot(native,expected)/(np.linalg.norm(native)*np.linalg.norm(expected)):.9g}, "
          f"max abs {np.max(np.abs(diff)):.6g}")
    stages = (
        ("early_image", b / "flux3-image-conditioned-five-block-fixture-512/native-x.bf16",
         1024, slice(0, 1024)),
        ("early_text", b / "flux3-text-conditioned-five-block-fixture-80/native-x.bf16",
         512, slice(0, 80)),
    )
    for name, path, padded, live in stages:
        got = bf16(path, padded, 3072).float().numpy()[live]
        ref = captures[name]
        delta = got - ref
        print(f"full DiT {name}: relative L2 {np.linalg.norm(delta)/np.linalg.norm(ref):.6g}, "
              f"cosine {np.vdot(got,ref)/(np.linalg.norm(got)*np.linalg.norm(ref)):.9g}")
    joint = bf16(b / "flux3-conditioned-shared-28-fixture-80-512/native-x.bf16",
                 1536, 3072).float().numpy()[:1104]
    reference_joint = np.concatenate((captures["shared"][1024:1104],
                                      captures["shared"][:1024]))
    delta = joint - reference_joint
    print(f"full DiT shared: relative L2 "
          f"{np.linalg.norm(delta)/np.linalg.norm(reference_joint):.6g}, cosine "
          f"{np.vdot(joint,reference_joint)/(np.linalg.norm(joint)*np.linalg.norm(reference_joint)):.9g}")


if __name__ == "__main__":
    main()
