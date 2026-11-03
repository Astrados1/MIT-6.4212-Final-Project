import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_eval.attribution import attribute, goal_translation
from grasp_eval.paper_taxonomy import PAPER_CATEGORIES, classify, cutamp_failure_kind
from grasp_eval.probe_math import yaw_quat
from grasp_eval.scoring import in_hand_motion
from grasp_eval.summarize import build_rows, main as summarize_main, render_paper_taxonomy
from grasp_eval.task_goal import aabb_iou, check_goal, shape_comparison, world_aabb

CUBE = [[-0.03, -0.03, -0.03], [0.03, 0.03, 0.03]]       # 6 cm cube, origin at its center
BOWL = [[-0.08, -0.08, 0.0], [0.08, 0.08, 0.06]]          # 16 cm bowl, origin at its bottom
ID = [1.0, 0.0, 0.0, 0.0]


def _record(final_cube, **kw):
    rec = {"run_id": "r", "target_object": "cube", "server_response_received": True, "server_success": True,
           "executed_steps": 100, "stable_lift": True, "score": {"num_close_commands": 1},
           "goal": [{"objects": ["$target"], "relation": "in", "container": "bowl"}],
           "object_local_bounds": {"cube": CUBE, "bowl": BOWL},
           "initial_object_positions": {"cube": [0.5, 0.0, 0.03], "bowl": [0.6, 0.2, 0.0]},
           "initial_object_quats": {"cube": ID, "bowl": ID},
           "final_object_positions": {"cube": final_cube, "bowl": [0.6, 0.2, 0.0]},
           "final_object_quats": {"cube": ID, "bowl": ID}}
    rec.update(kw)
    return rec


class TaskGoalTest(unittest.TestCase):
    def test_world_aabb_rotates_box(self):
        lo, hi = world_aabb([-0.1, -0.02, 0], [0.1, 0.02, 0.01], [1, 2, 0], yaw_quat(np.pi / 2))
        np.testing.assert_allclose(lo, [0.98, 1.9, 0], atol=1e-9)
        np.testing.assert_allclose(hi, [1.02, 2.1, 0.01], atol=1e-9)

    def test_iou_and_shape(self):
        self.assertAlmostEqual(aabb_iou([0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]), 1.0)
        self.assertAlmostEqual(aabb_iou([0, 0, 0], [1, 1, 1], [0, 0, 0], [0.5, 1, 1]), 0.5)
        s = shape_comparison([0, 0, 0], [0.06, 0.03, 0.06], [0, 0, 0], [0.06, 0.06, 0.06])
        self.assertEqual(s["extent_ratio"], [1.0, 0.5, 1.0])
        self.assertAlmostEqual(s["aabb_iou"], 0.5)

    def test_in_bowl(self):
        self.assertTrue(check_goal(_record([0.6, 0.2, 0.035])["goal"], _record([0.6, 0.2, 0.035]), True)["task_success"])
        r = _record([0.4, 0.0, 0.03])  # still on the table, outside the bowl
        self.assertFalse(check_goal(r["goal"], r, True)["task_success"])
        r = _record([0.6, 0.2, 0.30])  # above the bowl: not resting in it
        self.assertFalse(check_goal(r["goal"], r, True)["task_success"])

    def test_on_and_closed_gripper_and_unchecked(self):
        r = _record([0.6, 0.2, 0.09], goal=[{"objects": ["cube"], "relation": "on", "container": "bowl"}])
        self.assertTrue(check_goal(r["goal"], r, True)["task_success"])
        self.assertFalse(check_goal(r["goal"], r, False)["task_success"])
        r = _record([0.6, 0.2, 0.035], goal=[{"objects": ["$target"], "relation": "in", "container": None}])
        self.assertIsNone(check_goal(r["goal"], r, True)["task_success"])
        self.assertIsNone(check_goal(None, r, True)["task_success"])

    def test_count_clause(self):
        r = _record([0.6, 0.2, 0.035])
        r["object_local_bounds"]["block"] = CUBE
        r["final_object_positions"]["block"] = [0.3, 0.0, 0.03]
        r["final_object_quats"]["block"] = ID
        goal = [{"objects": ["cube", "block"], "count": 1, "relation": "in", "container": "bowl"}]
        self.assertTrue(check_goal(goal, r, True)["task_success"])
        goal[0]["count"] = 2
        self.assertFalse(check_goal(goal, r, True)["task_success"])


def _carry(n=60, close=10, lift=20, release=50, slip_at=None, drop_at=None, rotate_hand=False):
    t = np.arange(1, n + 1) / 15
    grip = np.zeros((n, 3))
    grip[:, :] = [0.5, 0.0, 0.03]
    grip[lift:, 2] = 0.15
    gq = np.tile(ID, (n, 1))
    if rotate_hand:  # hand yaws while carrying; a rigid grasp rotates the object with it
        for i in range(lift, n):
            gq[i] = yaw_quat(np.deg2rad(2 * (i - lift)))
    obj = grip.copy()
    oq = gq.copy()
    if rotate_hand:  # object 2 cm off the finger midpoint along the hand's x axis
        for i in range(n):
            c, s = np.cos(np.deg2rad(2 * max(0, i - lift))), np.sin(np.deg2rad(2 * max(0, i - lift)))
            obj[i] = grip[i] + np.array([0.02 * c, 0.02 * s, 0.0])
    if slip_at is not None:
        obj[slip_at:release, 0] += 0.03
    if drop_at is not None:
        obj[drop_at:, 2] = 0.03
    cmd = np.zeros(n)
    cmd[close:release] = 1
    return t, obj, oq, grip, gq, cmd


class InHandMotionTest(unittest.TestCase):
    def test_rigid_carry(self):
        m = in_hand_motion(*_carry(), baseline_z=0.03)
        self.assertTrue(m["carried"])
        self.assertEqual(m["max_in_hand_shift_m"], 0.0)
        self.assertFalse(m["dropped_before_release"])
        self.assertTrue(m["released_by_plan"])

    def test_hand_rotation_is_not_slip(self):
        m = in_hand_motion(*_carry(rotate_hand=True), baseline_z=0.03)
        self.assertLess(m["max_in_hand_shift_m"], 1e-6)
        self.assertLess(m["max_in_hand_rot_deg"], 1e-3)

    def test_slip_and_drop(self):
        m = in_hand_motion(*_carry(slip_at=35), baseline_z=0.03)
        self.assertAlmostEqual(m["max_in_hand_shift_m"], 0.03, places=4)
        self.assertFalse(m["dropped_before_release"])
        m = in_hand_motion(*_carry(drop_at=40), baseline_z=0.03)
        self.assertTrue(m["dropped_before_release"])
        self.assertAlmostEqual(m["drop_t"], 41 / 15)

    def test_never_lifted(self):
        t, obj, oq, grip, gq, cmd = _carry()
        obj[:, 2] = 0.03
        self.assertFalse(in_hand_motion(t, obj, oq, grip, gq, cmd, baseline_z=0.03)["carried"])


class GoalTranslationTest(unittest.TestCase):
    goal = [{"objects": ["$target"], "relation": "in", "container": "bowl"}]
    labels = {"rubiks_cube": "cube", "blue_bowl": "bowl", "mug": "mug"}

    def tr(self, atoms, goal=None):
        return goal_translation(atoms, self.labels, goal or self.goal, "cube")

    def test_correct_and_wrong(self):
        self.assertTrue(self.tr([{"predicate": "on", "args": ["rubiks_cube", "blue_bowl"]}])["goal_translation_correct"])
        self.assertFalse(self.tr([{"predicate": "on", "args": ["rubiks_cube", "mug"]}])["goal_translation_correct"])
        self.assertFalse(self.tr([{"predicate": "holding", "args": ["rubiks_cube"]}])["goal_translation_correct"])
        self.assertFalse(self.tr([{"predicate": "on", "args": ["rubiks_cube", "blue_bowl"]},
                                  {"predicate": "on", "args": ["mug", "blue_bowl"]}])["goal_translation_correct"])

    def test_unmapped_label_is_unverifiable(self):
        self.assertIsNone(self.tr([{"predicate": "on", "args": ["ghost", "blue_bowl"]}])["goal_translation_correct"])

    def test_unknown_names_skip(self):
        self.assertEqual(self.tr([], goal=[{"objects": ["$target"], "relation": "in", "container": None}]), {})


class AttributionAdditionsTest(unittest.TestCase):
    def test_grasp_source_shape_and_detection(self):
        rec = _record([0.4, 0.0, 0.03], stable_lift=False, score={"num_close_commands": 1, "failure_hint": "missed_grasp"})
        perceived = {
            "objects": [
                # cube seen as a 3 cm sliver (partial view); z includes the client's -1.5 cm offset
                {"name": "rubiks_cube", "role": "movable", "aabb_min": [0.47, -0.03, -0.015],
                 "aabb_max": [0.50, 0.03, 0.045], "aabb_center": [0.485, 0.0, 0.015]},
                {"name": "table", "role": "static", "aabb_min": [-1, -1, -0.05], "aabb_max": [1, 1, -0.015],
                 "aabb_center": [0, 0, -0.03]},
            ],
            "m2t2_grasp_counts": {"rubiks_cube": 0},
            "grounded_atoms": [{"predicate": "on", "args": ["rubiks_cube", "bowl"]}],
        }
        plan = {"steps": [{"type": "gripper", "label": "Pick(rubiks_cube, g, q)", "action": "close"}]}
        a = attribute(rec, None, perceived, plan)
        self.assertEqual(a["grasp_source"], "heuristic")
        self.assertEqual(a["shape"]["cube"]["extent_ratio"], [0.5, 1.0, 1.0])
        self.assertEqual(a["goal_objects_without_detection"], ["bowl"])
        self.assertIsNone(a["goal_translation_correct"])  # "bowl" label maps to nothing perceived
        self.assertEqual(a["paper_category"], "gemini_vlm_failure/object_detection_failure")

    def test_cutamp_reason_kinds(self):
        self.assertEqual(cutamp_failure_kind("Motion planning failed for 3/5 satisfying particle(s)"), "motion_planning")
        self.assertEqual(cutamp_failure_kind("No satisfying particles found after optimizing 2/4 plan(s) "
                                             "(time budget 60.0s exceeded)"), "time_budget")
        self.assertEqual(cutamp_failure_kind("No valid plan skeletons found for the given goal"), "no_plan_skeleton")
        self.assertIsNone(cutamp_failure_kind(None))


class ClassifyTest(unittest.TestCase):
    ok = {"run_id": "r", "target_object": "cube", "server_response_received": True, "server_success": True,
          "executed_steps": 100, "stable_lift": True, "score": {"num_close_commands": 1}}

    def label(self, rec=None, attr=None, **kw):
        return classify(dict(self.ok, **(rec or {})), attr or {}, **kw)["paper_category"]

    def test_success_and_unchecked(self):
        self.assertEqual(self.label(attr={"task_check": {"task_success": True}}), "success")
        self.assertEqual(self.label(), "needs_review")
        self.assertEqual(self.label(task_success=True), "success")

    def test_infrastructure(self):
        self.assertEqual(classify(None, None)["paper_category"], "infrastructure")
        self.assertEqual(self.label({"harness_exception": "boom"}), "infrastructure")
        self.assertEqual(self.label({"server_success": False, "server_error": "429 RESOURCE_EXHAUSTED quota"}),
                         "infrastructure")

    def test_gemini(self):
        self.assertEqual(self.label({"server_success": False, "server_error": "Gemini returned a non-JSON response"}),
                         "gemini_vlm_failure/invalid_response")
        self.assertEqual(self.label(attr={"goal_translation_correct": False}),
                         "gemini_vlm_failure/goal_translation_failure")
        self.assertEqual(self.label(attr={"goal_objects_without_detection": ["bowl"]}),
                         "gemini_vlm_failure/object_detection_failure")
        self.assertEqual(self.label(attr={"first_pick_matches_target": False}),
                         "gemini_vlm_failure/object_detection_failure")

    def test_scene_completion(self):
        self.assertEqual(self.label(attr={"shape": {"cube": {"extent_ratio": [0.5, 1, 1], "aabb_iou": 0.5}}}),
                         "scene_completion_failure/partial_observability")
        self.assertEqual(self.label(attr={"shape": {"bowl": {"extent_ratio": [1.5, 1, 1], "aabb_iou": 0.6}}}),
                         "scene_completion_failure/shape_over_approximation")

    def test_cutamp_no_plan(self):
        no_plan = {"server_success": False, "executed_steps": 0,
                   "server_error": "cuTAMP failed to find a plan: Motion planning failed for 4/4 satisfying particle(s)"}
        self.assertEqual(self.label(no_plan), "cutamp_failure/curobo_motion_planning_failure")
        no_plan["server_error"] = "cuTAMP failed to find a plan: No valid plan skeletons found for the given goal"
        self.assertEqual(self.label(no_plan), "cutamp_failure/no_feasible_plan")

    def test_grasp_failures(self):
        failed = {"stable_lift": False, "score": {"num_close_commands": 1, "failure_hint": "missed_grasp"}}
        self.assertEqual(self.label(failed, {"grasp_source": "m2t2"}), "grasp_failure/m2t2_unstable_grasp")
        self.assertEqual(self.label(failed, {"grasp_source": "heuristic"}), "grasp_failure/heuristic_unstable_grasp")
        self.assertEqual(self.label(failed), "grasp_failure/unknown_grasp_source")
        self.assertEqual(self.label(failed, {"joint_tracking_error_at_close_rad": 0.2}),
                         "execution_failure/tracking_or_collision")
        # Lifted, then dropped while carried: an unstable grasp, not a success.
        self.assertEqual(self.label(attr={"grasp_source": "m2t2", "in_hand": {"dropped_before_release": True},
                                          "task_check": {"task_success": False}}),
                         "grasp_failure/m2t2_unstable_grasp")

    def test_rigid_grasp_assumption(self):
        attr = {"in_hand": {"max_in_hand_shift_m": 0.03, "max_in_hand_rot_deg": 2.0}, "task_check": {"task_success": False}}
        self.assertEqual(self.label(attr=attr), "cutamp_failure/rigid_grasp_assumption")
        attr["in_hand"]["max_in_hand_shift_m"] = 0.002
        self.assertEqual(self.label(attr=attr), "needs_review")

    def test_every_label_is_registered(self):
        self.assertTrue(all("/" in k or k in ("success", "infrastructure", "needs_review") for k in PAPER_CATEGORIES))


class PaperSummaryTest(unittest.TestCase):
    def _write(self, d: Path, records, attrs, labels=None):
        (d / "schedule.jsonl").write_text("".join(json.dumps({"run_id": r["run_id"], "variant": 0}) + "\n"
                                                  for r in records) + json.dumps({"run_id": "crash", "variant": 0}) + "\n")
        (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (d / "attribution.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attrs))
        if labels:
            (d / "manual_labels.csv").write_text(labels)

    def test_table_sankey_and_overrides(self):
        base = dict(ClassifyTest.ok)
        records = [dict(base, run_id=f"r{i}") for i in range(4)]
        attrs = [
            {"run_id": "r0", "paper_category": "success", "evidence": [], "task_check": {"task_success": True}},
            {"run_id": "r1", "paper_category": "grasp_failure/m2t2_unstable_grasp", "evidence": ["x"]},
            {"run_id": "r2", "paper_category": "needs_review", "evidence": ["unchecked"]},
            {"run_id": "r3", "paper_category": "needs_review", "evidence": ["unchecked"]},
        ]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._write(d, records, attrs,
                        "run_id,stable_lift,failure_category,task_success,note,paper_category\n"
                        "r2,,,,,cutamp_failure/rigid_grasp_assumption\n"
                        "r3,,,true,,\n")
            rows, _ = build_rows(d)
            by = {r["run_id"]: r for r in rows}
            self.assertEqual(by["r2"]["paper_category"], "cutamp_failure/rigid_grasp_assumption")
            self.assertEqual(by["r3"]["paper_category"], "success")  # manual task label reclassifies
            self.assertEqual(by["crash"]["paper_category"], "infrastructure")
            text = render_paper_taxonomy(rows)
            self.assertIn("Denominator: 4 paper-comparable trials = 5 scheduled minus 1 excluded", text)
            self.assertIn("| Success | 2 | 50% |  | 118 | yes |", text)
            self.assertIn("| &nbsp;&nbsp;&nbsp;&nbsp;M2T2 Unstable Grasp | 1 | 25% | 50% | 20 | yes |", text)
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                summarize_main([str(d)])
            sankey = json.loads((d / "paper_taxonomy.json").read_text())
            self.assertEqual(sankey["comparable_trials"], 4)
            self.assertIn({"source": "All Trials", "target": "Success", "value": 2}, sankey["links"])

    def test_bad_paper_category_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._write(d, [dict(ClassifyTest.ok, run_id="r0")], [],
                        "run_id,stable_lift,failure_category,task_success,note,paper_category\nr0,,,,,oops\n")
            with self.assertRaises(ValueError):
                build_rows(d)


if __name__ == "__main__":
    unittest.main()
