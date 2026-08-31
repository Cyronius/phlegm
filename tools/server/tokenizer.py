"""Tokenizer interface + a PLACEHOLDER implementation.

The real tokenizer is being built separately under tools/kernel-interp/. This
module defines the thin interface the server depends on and ships a byte-level
identity codec so the HTTP layer is fully exercised without it. Swapping in the
real tokenizer is one line: implement `Tokenizer` and return it from
`get_tokenizer()`.

Interface (duck-typed, no ABC needed):
    encode(text: str) -> list[int]
    decode(ids: list[int]) -> str          # must be incremental-safe: decoding a
                                            # growing prefix must be monotonic
    apply_chat_template(messages) -> str    # flatten chat turns to a prompt string
    eos_id -> int | None
"""
from __future__ import annotations
from typing import Optional


class PlaceholderTokenizer:
    """UTF-8 byte identity codec.

    encode(text) -> the text's UTF-8 bytes as ints (0..255): genuinely reversible
    for input. decode() renders 0..255 as their byte, and any id >= 256 (e.g. a
    real vocab id coming back from the NPU) as a visible '⟨id⟩' marker so output
    is legible without a real vocab. This is a stub, not the model's BPE.
    """

    eos_id: Optional[int] = None  # unknown without the real vocab

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        pieces: list[str] = []
        for i in ids:
            if 0 <= i < 256:
                out.append(i)
            else:
                if out:
                    pieces.append(out.decode("utf-8", errors="replace"))
                    out = bytearray()
                pieces.append(f"⟨{i}⟩")
        if out:
            pieces.append(out.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def apply_chat_template(self, messages: list[dict]) -> str:
        # Minimal flatten. The real chat template lives with the real tokenizer;
        # this keeps prompt-assembly in one obvious place to replace.
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):  # OpenAI content-parts form
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)


def get_tokenizer():
    """Single swap point. Replace the body with the real tokenizer when ready:

        from real_tokenizer import RealTokenizer
        return RealTokenizer(vocab_path=...)
    """
    return PlaceholderTokenizer()
