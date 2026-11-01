"""Text-only MOSS Image prompt tokens for the native Qwen3.5 encoder.

The reference calls Qwen2Tokenizer.apply_chat_template with one user message,
add_generation_prompt=True and enable_thinking=True, then truncates to 4096.
For that exact input shape the bundled chat_template.jinja renders the framing
below. Keeping the text-only framing explicit avoids loading a model or its
remote Python merely to prepare token IDs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MAX_SEQUENCE_LENGTH = 4096


def chat_text(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return "<|im_start|>user\n" + prompt.strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"


def token_ids(tokenizer_dir: Path, prompt: str, max_length: int = MAX_SEQUENCE_LENGTH) -> list[int]:
    if max_length < 1 or max_length > MAX_SEQUENCE_LENGTH:
        raise ValueError("max_length must be between 1 and 4096")
    from tokenizers import Tokenizer

    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    tokenizer = Tokenizer.from_file(str(path))
    ids = tokenizer.encode(chat_text(prompt), add_special_tokens=True).ids[:max_length]
    if not ids:
        raise ValueError("tokenizer produced no tokens")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_dir", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--max-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--ids-file", type=Path, help="write comma-separated IDs for open_qwen36_cli")
    args = parser.parse_args()
    ids = token_ids(args.tokenizer_dir, args.prompt, args.max_length)
    csv = ",".join(map(str, ids))
    if args.ids_file:
        args.ids_file.write_text(csv + "\n", encoding="utf-8")
        print(f"{len(ids)} tokens -> {args.ids_file}")
    else:
        print(csv)


if __name__ == "__main__":
    main()
