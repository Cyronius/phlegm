"""Diagnostic Qwen Image 2.1 text embeddings from the native Qwen3-VL NPU tower."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
from tokenizers import Tokenizer


SYSTEM = "Comprehend and analyze the provided prompt."
HIDDEN = 4096


def prompt_ids(model: Path, prompt: str) -> tuple[list[int], int]:
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    prefix = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    chat = (prefix + f"<|im_start|>user\n{prompt or ' '}<|im_end|>\n"
            "<|im_start|>assistant\n")
    ids = tokenizer.encode(chat, add_special_tokens=False).ids
    drop = len(tokenizer.encode(prefix, add_special_tokens=False).ids)
    return ids, drop


def run(args) -> Path:
    model, kernels, binary = map(Path, (args.model, args.kernels, args.binary))
    for path in (model / "model.q4nx", model / "config.json", model / "tokenizer.json",
                 kernels / "manifest.json", binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    ids, drop = prompt_ids(model, args.prompt)
    if not 1 <= len(ids) - drop <= 64:
        raise ValueError("Qwen Image 2.1 currently supports prompts of 1..64 text tokens")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ids_file = out / "ids.txt"
    ids_file.write_text(",".join(map(str, ids)), encoding="ascii")
    tap_file = out / "native-taps.f32"
    command = [str(binary), "--model", str(model), "--kernels", str(kernels),
               "--ids-file", str(ids_file), "--max-ctx", "256", "--pmode", "none",
               "--encode-taps", str(tap_file), "--tap-layers", "36"]
    with (out / "encoder.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    values = np.fromfile(tap_file, dtype="<f4")
    if values.size != len(ids) * HIDDEN or not np.isfinite(values).all():
        raise ValueError("native Qwen3-VL text tower returned invalid last-layer residuals")
    values.reshape(len(ids), HIDDEN)[drop:].astype(bfloat16).tofile(out / "tokens.bf16")
    (out / "fixture.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "scope": "native NPU text tower diagnostic",
        "prompt": args.prompt, "input_tokens": len(ids), "drop_prefix_tokens": drop,
        "text_tokens": len(ids) - drop, "channels": HIDDEN,
        "hidden_state": "layer 36 residual before final RMSNorm",
    }, indent=2) + "\n")
    print(f"native Qwen Image 2.1 text: {len(ids) - drop} x {HIDDEN} -> {out / 'tokens.bf16'}")
    return out / "tokens.bf16"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
