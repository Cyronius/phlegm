"""First-frame Qwen VAE DupUp3D shortcut, a host-side tensor rearrangement.

This is only repeat, reshape, select, and transpose. Learned arithmetic and
residual addition stay in NPU convolution and elementwise kernels.
"""

from __future__ import annotations

import argparse

import numpy as np


def first_frame(x: np.ndarray, output_channels: int, temporal_factor: int,
                spatial_factor: int = 2) -> np.ndarray:
    """Return the published DupUp3D(first_chunk=True) frame in NHWC layout."""
    if x.ndim != 3:
        raise ValueError("shortcut input must be [height,width,channels]")
    height, width, input_channels = x.shape
    if temporal_factor not in (1, 2) or spatial_factor != 2:
        raise ValueError("unsupported first-frame upsample factors")
    factor = temporal_factor * spatial_factor * spatial_factor
    total = output_channels * factor
    if total % input_channels:
        raise ValueError("output channels and factors do not divide input channels")
    repeats = total // input_channels
    channels_first = np.transpose(x, (2, 0, 1))
    repeated = np.repeat(channels_first, repeats, axis=0)
    arranged = repeated.reshape(output_channels, temporal_factor,
                                spatial_factor, spatial_factor, height, width)
    selected = arranged[:, temporal_factor - 1]
    return np.transpose(selected, (3, 1, 4, 2, 0)).reshape(
        height * spatial_factor, width * spatial_factor, output_channels)


def verify() -> None:
    import torch

    for ci, co, ft in ((1152, 1152, 2), (1152, 576, 2), (576, 288, 1)):
        h = 4
        x = np.arange(h * h * ci, dtype=np.float32).reshape(h, h, ci)
        got = first_frame(x, co, ft)
        a = torch.from_numpy(x).permute(2, 0, 1)[None, :, None]
        factor = ft * 4
        repeats = co * factor // ci
        ref = a.repeat_interleave(repeats, dim=1)
        ref = ref.view(1, co, ft, 2, 2, 1, h, h)
        ref = ref.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        ref = ref.view(1, co, ft, h * 2, h * 2)[:, :, ft - 1]
        expected = ref[0].permute(1, 2, 0).numpy()
        if not np.array_equal(got, expected):
            raise AssertionError(f"DupUp3D first-frame mismatch for {ci}->{co}, ft={ft}")
        print(f"DupUp3D first frame {ci}->{co}, ft={ft}: exact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()


if __name__ == "__main__":
    main()
