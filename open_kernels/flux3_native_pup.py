"""Native FLUX 3 text-to-image prompt upsampler.

Uses the published T2I system prompt, non-thinking Qwen3.5 chat framing,
greedy decode, presence penalty, and stop tokens.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from tokenizers import Tokenizer


def render_prompt(model_dir: Path, prompt: str) -> list[int]:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be non-empty")
    system = json.loads((model_dir / "system_prompts.json").read_text())["t2i"]
    chat = (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return Tokenizer.from_file(str(model_dir / "tokenizer.json")).encode(
        chat, add_special_tokens=False).ids


def run(args) -> Path:
    model, kernels, binary = map(Path, (args.model, args.kernels, args.binary))
    for path in (model / "model.q4nx", model / "config.json", model / "tokenizer.json",
                 model / "system_prompts.json", kernels / "manifest.json", binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    ids = render_prompt(model, args.prompt)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ids_file = out / "ids.txt"
    ids_file.write_text(",".join(map(str, ids)), encoding="ascii")
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    stop_ids = {tokenizer.token_to_id(stop) for stop in ("<|im_end|>", "<|endoftext|>")}
    if None in stop_ids:
        raise ValueError("FLUX 3 upsampler tokenizer is missing a stop token")
    command = [str(binary), "--model", str(model), "--kernels", str(kernels),
               "--ids-file", str(ids_file), "--max-ctx", "8192", "--pmode", "none",
               "--gemm-block", "--max-tokens", str(args.max_tokens),
               "--presence-penalty", "1.5", "--stop-ids",
               ",".join(map(str, sorted(stop_ids)))]
    with (out / "encoder.log").open("w") as log:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=log, text=True, check=True)
    (out / "tokens.log").write_text(result.stdout)
    generated = [int(token) for token in re.findall(r"^token (\d+)$", result.stdout, re.M)]
    if not generated:
        raise ValueError("native FLUX 3 upsampler generated no tokens")
    stop_at = next((i for i, token in enumerate(generated) if token in stop_ids), None)
    caption = tokenizer.decode(generated[:stop_at] if stop_at is not None else generated,
                               skip_special_tokens=False)
    for stop in ("<|im_end|>", "<|endoftext|>"):
        caption = caption.split(stop, 1)[0]
    caption = caption.strip()
    (out / "caption.txt").write_text(caption + "\n")
    (out / "fixture.json").write_text(json.dumps({
        "model": "black-forest-labs/flux-3-image-klein-pup",
        "scope": "native NPU T2I prompt upsampler",
        "prompt": args.prompt, "input_tokens": len(ids),
        "output_tokens": len(generated), "complete": stop_at is not None,
    }, indent=2) + "\n")
    print(f"native FLUX 3 upsampler: {len(ids)} prompt tokens, "
          f"{len(generated)} generated tokens -> {out / 'caption.txt'}")
    return out / "caption.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
