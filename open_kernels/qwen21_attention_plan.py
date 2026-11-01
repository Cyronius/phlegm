"""Segment Qwen Image 2.1's block-causal mask for native dit_fa streams.

The published processor permits earlier keys and every key in the query's own
image block. A text causal pass followed by one bidirectional prefix pass per
image block has the same mask. The caller must stage each pass's Q rows and K/V
prefix, discard padded Q rows, and stitch outputs back in original row order.
This module is the schedule contract; it does not execute attention.
"""

from __future__ import annotations

from dataclasses import dataclass

Q_TILE = 512
KV_TILE = 64


def _round_up(value: int, tile: int) -> int:
    return (value + tile - 1) // tile * tile


@dataclass(frozen=True)
class AttentionPass:
    kind: str
    q_start: int
    q_end: int
    k_end: int
    q_rows: int
    kv_rows: int
    valid_len: int
    causal: bool

    def allows(self, query: int, key: int) -> bool:
        """Global-index mask represented by this pass, excluding padded rows."""
        if not self.q_start <= query < self.q_end:
            raise ValueError("query is outside this attention pass")
        return 0 <= key < self.valid_len and (not self.causal or key <= query)


def passes(text_tokens: int, image_block_lengths: tuple[int, ...]) -> tuple[AttentionPass, ...]:
    """Produce shapes supported by dit_fa for one batch element.

    Text keys are causal; each image block attends bidirectionally to itself
    and to the complete prefix. Positive block lengths preserve the published
    block identities, including multiple reference/image segments.
    """
    if text_tokens < 0 or any(n <= 0 for n in image_block_lengths):
        raise ValueError("text length must be nonnegative and image blocks positive")
    result = []
    if text_tokens:
        result.append(AttentionPass("text", 0, text_tokens, text_tokens,
                                    _round_up(text_tokens, Q_TILE),
                                    _round_up(text_tokens, KV_TILE), text_tokens, True))
    offset = text_tokens
    for index, length in enumerate(image_block_lengths):
        end = offset + length
        result.append(AttentionPass(f"image.{index}", offset, end, end,
                                    _round_up(length, Q_TILE),
                                    _round_up(end, KV_TILE), end, False))
        offset = end
    return tuple(result)


@dataclass(frozen=True)
class SegmentPass:
    """One dit_fa pass of an interleaved layout. Queries [q_start, q_end) run against
    keys [0, valid_len); only the rows in ``keep`` (global half-open ranges) are
    written back. A causal pass computes every row below q_end so that its causal
    mask uses global indices; rows it does not keep are recomputed by a later pass."""
    kind: str
    q_start: int
    q_end: int
    keep: tuple[tuple[int, int], ...]
    q_rows: int
    kv_rows: int
    valid_len: int
    causal: bool

    def allows(self, query: int, key: int) -> bool:
        if not any(a <= query < b for a, b in self.keep):
            raise ValueError("query is not written by this pass")
        return 0 <= key < self.valid_len and (not self.causal or key <= query)


def segment_passes(segments: tuple[tuple[str, int], ...]) -> tuple[SegmentPass, ...]:
    """Passes for an edit's published order, e.g. text, condition image, text, target.

    ``segments`` are ("text" | "image", length) in sequence order. Text is causal
    over everything before it, so ONE causal pass over [0, last text end) gives every
    text row; its image rows are discarded. Each image block is one bidirectional
    pass over its own rows against the prefix through the block's end."""
    if any(kind not in ("text", "image") or n <= 0 for kind, n in segments):
        raise ValueError("segments are ('text' | 'image', positive length)")
    result, offset, text_rows, last_text = [], 0, [], 0
    images = []
    for kind, n in segments:
        if kind == "text":
            text_rows.append((offset, offset + n))
            last_text = offset + n
        else:
            images.append((offset, offset + n))
        offset += n
    if text_rows:
        result.append(SegmentPass("text", 0, last_text, tuple(text_rows),
                                  _round_up(last_text, Q_TILE), _round_up(last_text, KV_TILE),
                                  last_text, True))
    for index, (start, end) in enumerate(images):
        result.append(SegmentPass(f"image.{index}", start, end, ((start, end),),
                                  _round_up(end - start, Q_TILE), _round_up(end, KV_TILE),
                                  end, False))
    return tuple(result)


def published_mask(query: int, key: int, text_tokens: int,
                   image_block_lengths: tuple[int, ...]) -> bool:
    """The `(query >= key) or same_image_block` definition for comparisons."""
    total = text_tokens + sum(image_block_lengths)
    if not (0 <= query < total and 0 <= key < total):
        raise ValueError("query or key is outside the sequence")
    if key <= query:
        return True
    start = text_tokens
    for length in image_block_lengths:
        end = start + length
        if start <= query < end and start <= key < end:
            return True
        start = end
    return False
