"""Offline checks for FLUX 3 one-reference editing schedules (no NPU, no IRON).

Run from the repository root: python -m pytest open_kernels/test_flux3_edit_plan.py
"""

from __future__ import annotations

import unittest

from open_kernels.flux3_block import plan as block_plan
from open_kernels.flux3_native_generate import reference_canvas
from open_kernels.flux3_shared_block import plan as shared_plan


class Flux3EditPlanTest(unittest.TestCase):
    def test_text_to_image_shared_specs_are_unchanged(self) -> None:
        p = shared_plan(512, 160, 0)
        self.assertEqual(p.modes, ("text", "image"))
        self.assertEqual(p.valid_tokens, 160 + 1024)
        self.assertNotIn("ref_row", p.streams["ew"]["qk"])
        self.assertFalse(any("lean" in s for s in p.streams["ew"].values()))

    def test_one_reference_shared_layout(self) -> None:
        p = shared_plan(512, 80, 3, references=1)
        self.assertEqual(p.modes, ("text", "image", "image_cond"))
        self.assertEqual((p.valid_tokens, p.tokens), (80 + 2048, 2560))
        qk = p.streams["ew"]["qk"]
        self.assertEqual((qk["n_txt"], qk["grid_h"], qk["grid_w"], qk["ref_row"]), (80, 64, 32, 32))
        cond = p.streams["ew"]["norm_image_cond"]["a"]
        self.assertEqual((cond["off"], cond["T"]), ((80 + 1024) * 3072, 1024))
        self.assertTrue(all(s.get("lean") == 1 for k, s in p.streams["ew"].items() if k != "qk"))
        self.assertEqual(p.streams["fa"]["attention"]["valid_len"], 2128)
        with self.assertRaises(ValueError):
            shared_plan(512, 80, 0, references=2)

    def test_reference_block_uses_l_minus_one_for_every_row(self) -> None:
        image, cond = block_plan(512, 0, mode="image"), block_plan(512, 0, mode="image_cond")
        self.assertEqual(cond.streams["ew"]["qk"]["ref_row"], 0)
        self.assertNotIn("ref_row", image.streams["ew"]["qk"])
        for kset in ("gemm", "fa"):
            self.assertEqual(image.streams[kset], cond.streams[kset])
        self.assertEqual({k: v for k, v in image.streams["ew"].items() if k != "qk"},
                         {k: v for k, v in cond.streams["ew"].items() if k != "qk"})

    def test_published_reference_canvas(self) -> None:
        self.assertEqual(reference_canvas(512, 512), ((512, 512), (512, 512)))
        self.assertEqual(reference_canvas(500, 510), ((500, 510), (512, 512)))
        self.assertEqual(reference_canvas(1024, 768), ((1024, 768), (1024, 768)))
        fit, canvas = reference_canvas(4000, 3000)
        self.assertLessEqual(canvas[0] * canvas[1], 4 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
