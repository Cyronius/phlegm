"""Contract checks for the MOSS host-side schedule and FP32 update."""

import unittest

import numpy as np

from moss_schedule import cfg_euler_step, sigmas, timestep_features


class MossScheduleTest(unittest.TestCase):
    def test_default_snaps_to_training_grid(self):
        schedule = sigmas()
        self.assertEqual(schedule.dtype, np.float32)
        self.assertEqual(len(schedule), 51)
        self.assertEqual((schedule[0], schedule[-1]), (1, 0))
        self.assertTrue(np.all(np.diff(schedule) < 0))
        # The published training_shift=1 grid has sigma = k/1000.
        np.testing.assert_allclose(schedule[:-1] * 1000,
                                   np.rint(schedule[:-1] * 1000), atol=1e-3)

    def test_large_request_removes_duplicate_training_steps(self):
        schedule = sigmas(1200)
        self.assertLessEqual(len(schedule), 1001)
        self.assertTrue(np.all(np.diff(schedule) < 0))

    def test_timestep_and_cfg_euler_are_fp32(self):
        features = timestep_features(1)
        self.assertEqual(features.shape, (256,))
        self.assertEqual(features.dtype, np.float32)
        self.assertAlmostEqual(float(features[0]), float(np.cos(1000)), places=6)
        self.assertAlmostEqual(float(features[128]), float(np.sin(1000)), places=6)
        updated = cfg_euler_step(np.array([1], np.float32), np.array([2], np.float32),
                                 np.array([1], np.float32), 3, 1, 0.5)
        np.testing.assert_array_equal(updated, np.array([-1], np.float32))


if __name__ == "__main__":
    unittest.main()
