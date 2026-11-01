"""Run FLUX 3's Qwen3-VL text tower on the native NPU encoder."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
from tokenizers import Tokenizer


TAPS = (4, 8, 12, 16, 20, 24, 28, 32)
HIDDEN = 2560
PAD_ID = 151643
PAD_MULTIPLE = 80
MAX_LENGTH = 4096


def token_ids(tokenizer_dir: Path, prompt: str) -> tuple[list[int], int]:
    if not prompt or not prompt.strip():
        raise ValueError("FLUX 3 needs a non-empty prompt")
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    rendered = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    real = tokenizer.encode(rendered, add_special_tokens=False).ids
    real = real[:MAX_LENGTH]
    length = min(math.ceil(len(real) / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
    return real + [PAD_ID] * (length - len(real)), len(real)


def encode_prompt(args) -> Path:
    model, kernels, binary = map(Path, (args.model, args.kernels, args.binary))
    for path in (model / "model.q4nx", model / "config.json", model / "tokenizer.json",
                 kernels / "manifest.json", binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    ids, real = token_ids(model, args.prompt)
    length = len(ids)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ids_file = out / "ids.txt"
    ids_file.write_text(",".join(map(str, ids)), encoding="ascii")
    tap_file = out / "native-taps.f32"
    command = [str(binary), "--model", str(model), "--kernels", str(kernels),
               "--ids-file", str(ids_file), "--max-ctx",
               str(math.ceil(length / 256) * 256), "--pmode", "none",
               "--encode-taps", str(tap_file), "--tap-layers", ",".join(map(str, TAPS)),
               "--tap-pad-start", str(real)]
    with (out / "encoder.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    values = np.fromfile(tap_file, dtype="<f4")
    if values.size != length * len(TAPS) * HIDDEN or not np.isfinite(values).all():
        raise ValueError("native Qwen3-VL encoder returned invalid layer taps")
    values.reshape(length, len(TAPS), HIDDEN).astype(bfloat16).tofile(out / "tokens.bf16")
    (out / "fixture.json").write_text(json.dumps({
        "model": "black-forest-labs/flux-3-image-klein",
        "scope": "native NPU Qwen3-VL text tower",
        "prompt": args.prompt,
        "real_tokens": real,
        "text_length": length,
        "tap_layers": TAPS,
        "channels": HIDDEN * len(TAPS),
    }, indent=2) + "\n")
    print(f"native FLUX 3 text encoder: {length} x {HIDDEN * len(TAPS)} -> {out / 'tokens.bf16'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    encode_prompt(parser.parse_args())


if __name__ == "__main__":
    main()
