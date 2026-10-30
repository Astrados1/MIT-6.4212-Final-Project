import unittest

import numpy as np

from grasp_eval.scoring import LiftCriteria, count_close_commands, score_first_grasp

DT = 1 / 15


def make_trace(obj_z, cmd, gripper_offset=0.0, n=None):
    """Object at the gripper (plus a z offset) unless gripper_offset moves the gripper away."""
    n = len(obj_z)
    t = np.arange(1, n + 1) * DT
    obj = np.zeros((n, 3))
    obj[:, 2] = obj_z
    grip = obj.copy()
    grip[:, 0] += gripper_offset
    return t, obj, grip, np.asarray(cmd, dtype=float)


class CountClosesTest(unittest.TestCase):
    def test_rising_edges(self):
        self.assertEqual(count_close_commands([0, 0, 1, 1, 0, 1, 0]), 2)
        self.assertEqual(count_close_commands([1, 1, 0]), 1)
        self.assertEqual(count_close_commands([]), 0)


class ScoreFirstGraspTest(unittest.TestCase):
    def test_stable_lift(self):
        # 10 open steps, close, then held 8 cm up for 1 s, then release.
        z = [0.02] * 15 + [0.10] * 15 + [0.10] * 5
        cmd = [0] * 10 + [1] * 20 + [0] * 5
        s = score_first_grasp(*make_trace(z, cmd), baseline_z=0.02)
        self.assertTrue(s.lifted)
        self.assertIsNone(s.failure_hint)
        self.assertEqual(s.num_close_commands, 1)
        self.assertAlmostEqual(s.first_close_t, 11 * DT)
        self.assertAlmostEqual(s.release_t, 31 * DT)

    def test_missed_grasp(self):
        z = [0.02] * 40
        cmd = [0] * 10 + [1] * 30
        s = score_first_grasp(*make_trace(z, cmd), baseline_z=0.02)
        self.assertFalse(s.lifted)
        self.assertEqual(s.failure_hint, "missed_grasp")

    def test_short_hold_is_drop(self):
        # Rises for 3 samples (~0.13 s) then falls while still closed.
        z = [0.02] * 12 + [0.10] * 3 + [0.02] * 20
        cmd = [0] * 10 + [1] * 25
        s = score_first_grasp(*make_trace(z, cmd), baseline_z=0.02)
        self.assertFalse(s.lifted)
        self.assertEqual(s.failure_hint, "drop_after_closure_or_lift")

    def test_tossed_object_far_from_gripper_does_not_count(self):
        z = [0.02] * 10 + [0.20] * 30
        cmd = [0] * 10 + [1] * 30
        s = score_first_grasp(*make_trace(z, cmd, gripper_offset=0.3), baseline_z=0.02)
        self.assertFalse(s.lifted)
        self.assertEqual(s.failure_hint, "missed_grasp")
        self.assertIsNone(s.max_rise_near_gripper_m)

    def test_lift_after_release_is_not_scored(self):
        # First close misses and reopens; a later second grasp lifts. Only the first is scored.
        z = [0.02] * 30 + [0.10] * 30
        cmd = [0] * 5 + [1] * 10 + [0] * 10 + [1] * 35
        s = score_first_grasp(*make_trace(z, cmd), baseline_z=0.02)
        self.assertFalse(s.lifted)
        self.assertEqual(s.num_close_commands, 2)

    def test_no_close(self):
        s = score_first_grasp(*make_trace([0.02] * 10, [0] * 10), baseline_z=0.02)
        self.assertFalse(s.lifted)
        self.assertIsNone(s.first_close_t)
        self.assertIsNone(s.failure_hint)

    def test_thresholds_are_respected(self):
        z = [0.02] * 10 + [0.06] * 30  # 4 cm rise
        cmd = [0] * 10 + [1] * 30
        trace = make_trace(z, cmd)
        self.assertFalse(score_first_grasp(*trace, baseline_z=0.02).lifted)
        self.assertTrue(score_first_grasp(*trace, baseline_z=0.02, criteria=LiftCriteria(min_rise_m=0.03)).lifted)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            score_first_grasp(np.arange(3), np.zeros((3, 3)), np.zeros((2, 3)), np.zeros(3), 0.0)


if __name__ == "__main__":
    unittest.main()
