"""Small exhaustive checks for Qwen Image 2.1's segmented attention mask."""

from __future__ import annotations

import itertools
import unittest

from qwen21_attention_plan import passes, segment_passes


class AttentionPlanTest(unittest.TestCase):
    def test_matches_block_causal_mask(self) -> None:
        cases = 0
        for text in range(5):
            for nblocks in range(4):
                for blocks in itertools.product(range(1, 5), repeat=nblocks):
                    labels = [-1] * text
                    for index, length in enumerate(blocks):
                        labels.extend([index] * length)
                    if not labels:
                        continue
                    planned = passes(text, blocks)
                    self.assertEqual(sum(p.q_end - p.q_start for p in planned), len(labels))
                    for part in planned:
                        self.assertEqual(part.q_rows % 512, 0)
                        self.assertEqual(part.kv_rows % 64, 0)
                        self.assertLessEqual(part.valid_len, part.kv_rows)
                    for query in range(len(labels)):
                        part = next(p for p in planned if p.q_start <= query < p.q_end)
                        for key in range(len(labels)):
                            expected = key <= query or (labels[query] >= 0 and
                                                        labels[query] == labels[key])
                            self.assertEqual(part.allows(query, key), expected,
                                             (text, blocks, query, key))
                    cases += 1
        self.assertEqual(cases, 424)

    def test_interleaved_segments_match_block_causal_mask(self) -> None:
        cases = 0
        for kinds in itertools.product(("text", "image"), repeat=4):
            for lengths in itertools.product(range(1, 4), repeat=4):
                segments = tuple(zip(kinds, lengths))
                labels, block = [], 0
                for kind, n in segments:
                    labels.extend([-1] * n if kind == "text" else [block] * n)
                    block += kind == "image"
                planned = segment_passes(segments)
                for query in range(len(labels)):
                    owners = [p for p in planned if any(a <= query < b for a, b in p.keep)]
                    self.assertEqual(len(owners), 1, (segments, query))
                    for key in range(len(labels)):
                        expected = key <= query or (labels[query] >= 0 and labels[query] == labels[key])
                        self.assertEqual(owners[0].allows(query, key), expected, (segments, query, key))
                for part in planned:
                    self.assertEqual(part.q_rows % 512, 0)
                    self.assertEqual(part.kv_rows % 64, 0)
                cases += 1
        self.assertEqual(cases, 1296)

    def test_invalid_lengths(self) -> None:
        with self.assertRaises(ValueError):
            passes(-1, ())
        with self.assertRaises(ValueError):
            passes(1, (0,))


if __name__ == "__main__":
    unittest.main()
