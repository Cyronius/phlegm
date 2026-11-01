"""Prepare and check native FLUX 3 autoencoder decoding on the NPU."""

from open_kernels.moss_vae_run import main as vae_main
from open_kernels.vae_decoder import BLOCK_OUT


if __name__ == "__main__":
    vae_main(profile=BLOCK_OUT, model="black-forest-labs/flux-3-image-klein")
