"""Published BF16 FLUX 3 flow-map modulation oracle for the NPU fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    out = args.fixture.resolve()
    source = args.model.resolve() / "model.safetensors"

    def rows(name: str) -> torch.Tensor:
        raw = np.fromfile(out / f"{name}.bf16", "<u2")
        return torch.from_numpy(raw.reshape(512, 512)[:2].copy()).view(torch.bfloat16)

    current, target = rows("current"), rows("target")
    with safe_open(source, framework="pt", device="cpu") as tensors:
        w = {name: tensors.get_tensor(name) for name in (
            "time_in.in_layer.weight", "time_in.out_layer.weight",
            "ref_time_in.in_layer.weight", "ref_time_in.out_layer.weight",
            "early_stream_modulations.image.lin.weight",
            "early_stream_modulations.txt.lin.weight",
            "single_stream_modulations.image.lin.weight",
            "single_stream_modulations.txt.lin.weight",
            "final_layer.image.adaLN_modulation.1.weight")}

    def linear(x: torch.Tensor, name: str) -> torch.Tensor:
        return torch.nn.functional.linear(x, w[name])

    vec_time = linear(torch.nn.functional.silu(linear(current[:, :256],
                          "time_in.in_layer.weight")), "time_in.out_layer.weight")
    vec_ref = linear(torch.nn.functional.silu(linear(target[:, :256],
                         "ref_time_in.in_layer.weight")), "ref_time_in.out_layer.weight")
    vec = vec_time + vec_ref
    activated = torch.nn.functional.silu(vec)
    outputs = {
        "vec": vec,
        "early_image": linear(activated, "early_stream_modulations.image.lin.weight"),
        "early_text": linear(activated, "early_stream_modulations.txt.lin.weight"),
        "single_image": linear(activated, "single_stream_modulations.image.lin.weight"),
        "single_text": linear(activated, "single_stream_modulations.txt.lin.weight"),
        "final": linear(activated, "final_layer.image.adaLN_modulation.1.weight"),
    }
    for name, value in outputs.items():
        result = value.float().numpy()
        if not np.isfinite(result).all():
            raise ValueError(f"FLUX 3 {name} reference contains NaN/Inf")
        (out / f"reference-{name}.f32").write_bytes(result.astype("<f4").tobytes())
        print(f"published FLUX 3 {name}: image-row RMS "
              f"{np.sqrt(np.mean(result[0]**2)):.6g}")


if __name__ == "__main__":
    main()
