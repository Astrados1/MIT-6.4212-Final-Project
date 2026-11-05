import unittest

import numpy as np

from grasp_eval.plan_segments import (
    ClientCursor, UnsupportedPlanError, clear_client_plan, find_grasp_segment, parse_label, run_segment, tag_action,
)
from tests.recovery_fakes import FakeTiptopClient, scene_plan


def steps_of(plan):
    return [s for s in plan["steps"] if s["type"] != "metadata"]


class ParseLabelTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(parse_label("Pick(rubiks_cube, grasp1, q1)"), ("Pick", ["rubiks_cube", "grasp1", "q1"]))
        self.assertEqual(parse_label("GoToInitial(q0)"), ("GoToInitial", ["q0"]))
        self.assertEqual(parse_label(None), (None, []))
        self.assertEqual(parse_label("garbage"), (None, []))


class FindSegmentTest(unittest.TestCase):
    def test_scene1_structure_uses_first_place_trajectory_as_retraction(self):
        # Same label/type sequence as droid-sim-evals tiptop_assets/tiptop_scene1_plan.json.
        seg = find_grasp_segment(steps_of(scene_plan([0.5, 0, 0.03], np.zeros(7))))
        self.assertEqual(seg.target_label, "rubiks_cube")
        self.assertEqual(seg.pick_traj_indices, (0, 1))
        self.assertEqual(seg.close_index, 2)
        self.assertEqual(seg.retraction_index, 3)
        self.assertTrue(seg.retraction_label.startswith("Place(rubiks_cube"))
        self.assertEqual(seg.retraction_source, "place_retract")

    def test_pick_only_plan_uses_go_to_initial_retraction(self):
        seg = find_grasp_segment(steps_of(scene_plan([0.5, 0, 0.03], np.zeros(7), pick_only=True)))
        self.assertEqual(seg.retraction_index, 3)
        self.assertEqual(seg.retraction_source, "go_to_initial_retract")

    def test_three_pick_trajectories_supported(self):
        s = steps_of(scene_plan([0.5, 0, 0.03], np.zeros(7)))
        s = [dict(s[0])] + s
        self.assertEqual(find_grasp_segment(s).retraction_index, 4)

    def test_target_label_must_match(self):
        s = steps_of(scene_plan([0.5, 0, 0.03], np.zeros(7)))
        self.assertEqual(find_grasp_segment(s, "Rubik's Cube").close_index, 2)  # normalized match
        with self.assertRaises(UnsupportedPlanError):
            find_grasp_segment(s, "red_bowl")

    def test_refuses_unsupported_structures(self):
        base = steps_of(scene_plan([0.5, 0, 0.03], np.zeros(7)))
        aux = steps_of(scene_plan([0.5, 0, 0.03], np.zeros(7), target="mug"))
        cases = {
            "no close": [s for s in base if s.get("action") != "close"],
            "aux pick first": aux[:7] + base,  # pick+place of another object before the target
            "retraction missing": base[:3],
            "gripper after close": base[:3] + [base[6]],
            "place of other object": base[:3] + [dict(base[3], label="Place(mug, grasp1, pose1, red_bowl, q2)")],
            "empty": [],
        }
        for name, s in cases.items():
            with self.subTest(name), self.assertRaises(UnsupportedPlanError):
                find_grasp_segment(s)


class ExecutionBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.plan = scene_plan([0.5, 0, 0.03], np.zeros(7))
        self.client = FakeTiptopClient(lambda req: scene_plan([0.5, 0, 0.03], np.zeros(7)))
        self.obs = {"q": np.zeros(7), "g": 0.0, "frame": 0}

    def _run(self, max_actions=500):
        c = self.client
        before = ClientCursor.of(c)
        ret = c.infer(self.obs, "x")
        first = (ret, tag_action(before, ClientCursor.of(c)))
        seg = find_grasp_segment(c._plan)
        applied = []

        def step_env(action, tag):
            applied.append((action.copy(), tag))
            self.obs = {"q": action[:7].copy(), "g": float(action[7]), "frame": len(applied)}
            return self.obs, False, False

        res = run_segment(c, self.obs, "x", seg, step_env, max_actions, first=first)
        return seg, res, applied

    def test_final_retraction_waypoint_is_applied_and_nothing_after(self):
        seg, res, applied = self._run()
        self.assertTrue(res.ok, res.reason)
        self.assertTrue(res.retraction_applied)
        retraction = np.asarray(steps_of(self.plan)[seg.retraction_index]["positions"])
        np.testing.assert_allclose(applied[-1][0][:7], retraction[-1], atol=1e-6)
        self.assertEqual(applied[-1][1].plan_step, seg.retraction_index)
        self.assertTrue(applied[-1][1].last_of_step)
        self.assertTrue(all(t.plan_step is not None and t.plan_step <= seg.retraction_index for _, t in applied))
        self.assertEqual(res.final_gripper_cmd, 1.0)
        # Client bookkeeping already points past the retraction although transport was never applied.
        self.assertEqual(self.client._current_plan_step, seg.retraction_index + 1)

    def test_stride_and_gripper_dwell_match_upstream(self):
        seg, res, applied = self._run()
        kinds = [t.kind for _, t in applied]
        self.assertEqual(kinds.count("gripper"), 21)  # 1 initial + 20 dwell ticks
        # 9 waypoints at stride 3 -> indices 0, 3, 6, 8
        self.assertEqual(sum(1 for _, t in applied if t.plan_step == seg.retraction_index), 4)

    def test_plan_done_is_true_before_the_last_waypoint_is_applied(self):
        # Why the week-one loop can't be reused as-is: on a plan truncated after the retraction,
        # plan_done flips when the final waypoint is *returned*, before env.step applies it.
        c = FakeTiptopClient(lambda req: {"steps": steps_of(self.plan)[:4]})
        obs = {"q": np.zeros(7), "g": 0.0, "frame": 0}
        n = 0
        while True:
            ret = c.infer(obs, "x")
            n += 1
            if c.plan_done:
                break
            obs = {"q": ret["action"][:7], "g": float(ret["action"][7]), "frame": n}
        np.testing.assert_allclose(ret["action"][:7], steps_of(self.plan)[3]["positions"][-1], atol=1e-6)
        self.assertFalse(np.allclose(obs["q"], ret["action"][:7]))  # the upstream loop would drop it

    def test_budget_exhaustion_stops_before_retraction(self):
        _, res, applied = self._run(max_actions=10)
        self.assertFalse(res.ok)
        self.assertFalse(res.retraction_applied)
        self.assertEqual(len(applied), 10)

    def test_clear_client_plan(self):
        self._run()
        clear_client_plan(self.client)
        self.assertIsNone(self.client._plan)
        self.assertIsNone(self.client._current_trajectory)
        self.assertEqual(self.client._current_plan_step, 0)
        self.assertIsNone(self.client._gripper_action_pending)
        self.assertFalse(self.client.plan_done)


if __name__ == "__main__":
    unittest.main()
