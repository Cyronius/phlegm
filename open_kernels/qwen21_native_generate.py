"""Generate a 512px Qwen Image 2.1 PNG with native Qwen3-VL, DiT and VAE NPU kernels."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from open_kernels.qwen21_native_dit_sample import sample
from open_kernels.qwen21_native_text import run as encode_text
from open_kernels.qwen21_prompt_kernels import prepare


def generate(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    text = encode_text(argparse.Namespace(model=args.text_model,
        kernels=args.text_kernels, binary=args.text_binary,
        prompt=args.prompt, out=out / "text"))
    token_count = json.loads((out / "text/fixture.json").read_text())["text_tokens"]
    block_kernels = prepare(args.block_kernels, args.kernel_probes, token_count,
                            out / "prompt-kernels")
    latents = sample(argparse.Namespace(model=args.model, weights=args.dit_weights,
        text_bf16=text, block_kernels=block_kernels, edge_kernels=args.edge_kernels,
        harness=args.harness, out=out / "dit", steps=args.steps, seed=args.seed,
        keep_intermediates=False))
    decode = out / "decode"
    subprocess.run((sys.executable, "-m", "open_kernels.qwen21_vae_run",
                    "--model", str(args.model),
                    "--conv-kernels", str(args.vae_conv_kernels),
                    "--ew-kernels", str(args.vae_ew_kernels),
                    "--attention-kernel", str(args.vae_attention_kernel),
                    "--weights", str(args.vae_weights),
                    "--latents", str(latents), "--out", str(decode)), check=True)
    image = decode / "native-qwen21.png"
    if not image.is_file():
        raise RuntimeError("native Qwen VAE did not produce a PNG")
    shutil.rmtree(decode / "scratch")
    (out / "generation.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "backend": "native-npu", "prompt": args.prompt,
        "text_tokens": token_count, "resolution": 512, "steps": args.steps,
        "seed": args.seed, "image": str(image)}, indent=2) + "\n")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    for name in ("model", "dit_weights", "text_model", "text_kernels", "text_binary",
                 "edge_kernels", "block_kernels", "kernel_probes", "vae_weights",
                 "vae_conv_kernels", "vae_ew_kernels", "vae_attention_kernel", "harness", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=41)
    print(generate(parser.parse_args()))


if __name__ == "__main__":
    main()
