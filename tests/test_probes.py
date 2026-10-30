import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_eval import run_suite
from grasp_eval.attribution import attribute, pick_objects, plan_timing
from grasp_eval.observed_eval import resolve_target
from grasp_eval.probe_math import corrupt_depth, displaced_pose, perturb_camera_pose, quat_multiply, yaw_quat
from grasp_eval.summarize import build_rows, load_attributions, render, render_attribution


class ProbeMathTest(unittest.TestCase):
    def test_displacement_shifts_xy_and_keeps_z(self):
        pos, quat = displaced_pose([0.5, 0.1, 0.03], [1, 0, 0, 0], 0.02, -0.01, 0)
        np.testing.assert_allclose(pos, [0.52, 0.09, 0.03])
        np.testing.assert_allclose(quat, [1, 0, 0, 0])

    def test_yaw_composes_in_world_frame(self):
        q0 = yaw_quat(np.deg2rad(30))
        _, q = displaced_pose([0, 0, 0], q0, 0, 0, 15)
        np.testing.assert_allclose(q, yaw_quat(np.deg2rad(45)), atol=1e-12)
        np.testing.assert_allclose(quat_multiply([1, 0, 0, 0], q0), q0)

    def test_camera_perturbation_magnitude_and_reproducibility(self):
        T = np.eye(4)
        T[:3, 3] = [0.3, 0.0, 0.5]
        a, da = perturb_camera_pose(T, 0.005, 2.0, np.random.default_rng(7))
        b, _ = perturb_camera_pose(T, 0.005, 2.0, np.random.default_rng(7))
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(np.linalg.norm(a[:3, 3] - T[:3, 3]), 0.005, places=6)
        angle = np.degrees(np.arccos((np.trace(a[:3, :3]) - 1) / 2))
        self.assertAlmostEqual(angle, 2.0, places=4)
        self.assertEqual(da["rot_deg"], 2.0)

    def test_depth_corruption(self):
        d = np.full((100, 100), 0.5, dtype=np.float32)
        d[0, 0] = 0.0  # invalid pixel stays invalid
        out = corrupt_depth(d, 0.002, 0.3, np.random.default_rng(0))
        self.assertEqual(out[0, 0], 0.0)
        dropped = np.mean(out == 0)
        self.assertTrue(0.25 < dropped < 0.35)
        kept = out[out > 0]
        self.assertAlmostEqual(float(np.std(kept)), 0.002, places=3)
        self.assertTrue(np.all(d[1:, 1:] == 0.5))  # input not modified


def _plan(dt=0.02):
    return {"steps": [
        {"type": "trajectory", "label": "Pick(Rubik's_cube, grasp1, q1)", "dt": dt, "positions": []},
        {"type": "gripper", "label": "Pick(Rubik's_cube, grasp1, q1)", "action": "close"},
        {"type": "trajectory", "label": "Place(Rubik's_cube, bowl, p1, q2)", "dt": dt, "positions": []},
        {"type": "trajectory", "label": "Pick(soup_can, grasp2, q3)", "dt": dt, "positions": []},
    ]}


def _trace(n=40, close_at=20, offset=(0.03, 0.0, 0.0)):
    names = np.array(["bowl", "rubiks_cube"])
    objs = np.zeros((n, 2, 3))
    objs[:, 0] = [0.6, 0.2, 0.02]
    objs[:, 1] = [0.5, 0.0, 0.03]
    grip = objs[:, 1] + np.asarray(offset)
    cmd = np.zeros(n)
    cmd[close_at:] = 1
    q = np.zeros((n, 7))
    q_cmd = q.copy()
    q_cmd[close_at - 1, 3] = 0.01
    return {"t": np.arange(1, n + 1) / 15, "object_names": names, "objects_pos": objs, "gripper_pos": grip,
            "gripper_cmd": cmd, "joint_pos": q, "joint_cmd": q_cmd}


class AttributionTest(unittest.TestCase):
    def test_pick_objects_and_timing(self):
        self.assertEqual(pick_objects(_plan()), ["Rubik's_cube", "soup_can"])
        t = plan_timing(_plan(0.02))
        self.assertEqual(t["client_waypoint_stride"], 3)
        self.assertAlmostEqual(t["execution_time_scale"][0], 1.111, places=3)
        self.assertEqual(plan_timing(_plan(1 / 45))["execution_time_scale"], [1.0])

    def test_full_attribution_for_failed_run(self):
        record = {"run_id": "r", "target_object": "rubiks_cube", "stable_lift": False, "server_success": True,
                  "executed_steps": 40,
                  "initial_object_positions": {"bowl": [0.6, 0.2, 0.02], "rubiks_cube": [0.5, 0.0, 0.03]}}
        perceived = {"objects": [
            # perceived 1 cm off in x; z includes the client's -1.5 cm camera offset
            {"name": "Rubik's_cube", "role": "movable", "aabb_center": [0.51, 0.0, 0.015]},
            {"name": "table", "role": "static", "aabb_center": [0.5, 0.0, -0.01]},
        ]}
        a = attribute(record, _trace(), perceived, _plan())
        self.assertEqual(a["first_pick_maps_to_sim_object"], "rubiks_cube")
        self.assertTrue(a["first_pick_matches_target"])
        self.assertAlmostEqual(a["perception_xy_error_m"], 0.01, places=4)
        self.assertAlmostEqual(a["perception_z_error_m"], 0.0, places=4)
        self.assertEqual(a["sim_objects_without_nearby_detection"], ["bowl"])
        self.assertAlmostEqual(a["gripper_xy_offset_at_close_m"], 0.03, places=4)
        self.assertAlmostEqual(a["joint_tracking_error_at_close_rad"], 0.01, places=4)
        self.assertEqual(a["target_pushed_before_close_m"], 0.0)
        self.assertEqual(a["stage_hint"], "grasp_pose_or_frame")

    def test_grounding_mismatch_is_first_hint(self):
        record = {"run_id": "r", "target_object": "rubiks_cube", "stable_lift": False, "server_success": True,
                  "executed_steps": 40,
                  "initial_object_positions": {"bowl": [0.6, 0.2, 0.02], "rubiks_cube": [0.5, 0.0, 0.03]}}
        perceived = {"objects": [{"name": "Rubik's_cube", "role": "movable", "aabb_center": [0.6, 0.2, 0.0]}]}
        a = attribute(record, _trace(), perceived, _plan())
        self.assertEqual(a["first_pick_maps_to_sim_object"], "bowl")
        self.assertEqual(a["stage_hint"], "grounding")

    def test_success_and_missing_inputs(self):
        record = {"run_id": "r", "target_object": "rubiks_cube", "stable_lift": True, "server_success": True,
                  "executed_steps": 40, "initial_object_positions": {}}
        a = attribute(record, None, None, None)
        self.assertFalse(a["perceived_scene_available"])
        self.assertIsNone(a["stage_hint"])

    def test_unmatched_pick_and_missing_trace_are_flagged(self):
        record = {"run_id": "r", "target_object": "rubiks_cube", "stable_lift": False, "server_success": True,
                  "executed_steps": 40, "initial_object_positions": {"rubiks_cube": [0.5, 0.0, 0.03]}}
        perceived = {"objects": [{"name": "red_block", "role": "movable", "aabb_center": [0.5, 0.0, 0.015]}]}
        a = attribute(record, None, perceived, _plan())
        self.assertFalse(a["first_pick_found_in_perceived_scene"])
        self.assertNotIn("perception_xy_error_m", a)
        self.assertEqual(a["stage_hint"], "insufficient_data")


class ResolveTargetTest(unittest.TestCase):
    names = ["bowl", "rubiks_cube", "soup_can"]

    def test_explicit(self):
        self.assertEqual(resolve_target("bowl", None, self.names, {}, None, None, None), ("bowl", "explicit"))

    def test_from_plan_label(self):
        plan = {"steps": [{"type": "trajectory", "label": "Pick(soup can, g, q)"}]}
        self.assertEqual(resolve_target("from-plan", plan, self.names, {}, None, None, None),
                         ("soup_can", "plan_label"))

    def test_alias_then_nearest_fallback(self):
        plan = {"steps": [{"type": "trajectory", "label": "Pick(Rubik's_cube, g, q)"}]}
        self.assertEqual(resolve_target("from-plan", plan, self.names, {"Rubik's_cube": "rubiks_cube"},
                                        None, None, None), ("rubiks_cube", "plan_label"))
        plan = {"steps": [{"type": "trajectory", "label": "Pick(red_block, g, q)"}]}
        objs = np.zeros((5, 3, 3))
        objs[:, 2] = [1, 1, 0]
        grip = np.tile([1.0, 1.0, 0.05], (5, 1))
        self.assertEqual(resolve_target("from-plan", plan, self.names, {}, 2, objs, grip),
                         ("soup_can", "nearest_at_close"))
        self.assertEqual(resolve_target("from-plan", None, self.names, {}, None, None, None), (None, "unresolved"))


class RunSuiteTest(unittest.TestCase):
    def _args(self, *argv):
        return run_suite.parse_args(list(argv))

    def test_displacement_schedule(self):
        a = self._args("displacement", "--target-object", "cube", "--dry-run", "--distances-cm", "2",
                       "--directions", "x+", "y-", "--yaws-deg", "30", "--repeats", "1")
        runs = run_suite.build_displacement(a)
        run_suite.assign_ids(runs)
        self.assertEqual([r.condition for r in runs], ["fresh_nominal", "replay_0cm", "disp_2cm_x+",
                                                       "disp_2cm_y-", "yaw_30deg"])
        self.assertTrue(all(r.plan_from is runs[0] for r in runs[1:]))
        self.assertEqual(runs[3].flags, {"displace-dx-m": 0.0, "displace-dy-m": -0.02})
        self.assertEqual(runs[2].group, "2cm")

    def test_replay_source_requires_a_saved_plan(self):
        a = self._args("displacement", "--target-object", "cube", "--dry-run", "--repeats", "1")
        runs = run_suite.build_displacement(a)
        run_suite.assign_ids(runs)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.assertIsNone(run_suite.replay_source(out, runs[1]))
            src_dir = out / "runs" / runs[0].run_id
            src_dir.mkdir(parents=True)
            plan = out / "plan.json"
            plan.write_text("{}")
            (src_dir / "record.json").write_text(json.dumps(
                {"server_success": True, "artifacts": {"server_plan": str(plan)}}))
            self.assertEqual(run_suite.replay_source(out, runs[1]), str(plan))

    def test_argument_validation(self):
        with self.assertRaises(SystemExit):
            self._args("pilot", "--dry-run")  # no target
        with self.assertRaises(SystemExit):
            self._args("displacement", "--target-object", "from-plan", "--dry-run")
        with self.assertRaises(SystemExit):
            self._args("sensor", "--target-object", "cube")  # no --tiptop-dir without dry run


class GroupedSummaryTest(unittest.TestCase):
    def test_condition_group_and_attribution_tables(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sched, runs, attrs = [], [], []
            for i, (cond, group, lift) in enumerate([("disp_1cm_x+", "1cm", True), ("disp_1cm_x-", "1cm", True),
                                                     ("disp_4cm_x+", "4cm", False), ("disp_4cm_x-", "4cm", False)]):
                rid = f"r{i}"
                sched.append({"run_id": rid, "variant": 0, "condition": cond, "group": group})
                runs.append({"run_id": rid, "server_success": True, "executed_steps": 50, "stable_lift": lift,
                             "harness_exception": None, "failure_category": None if lift else "missed_grasp",
                             "target_source": "explicit"})
                attrs.append({"run_id": rid, "gripper_xy_offset_at_close_m": 0.01 if lift else 0.04,
                              "stage_hint": None if lift else "grasp_pose_or_frame",
                              "plan_timing": {"trajectory_dts_s": [0.02], "execution_time_scale": [1.111]}})
            for name, rows in (("schedule", sched), ("runs", runs), ("attribution", attrs)):
                (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            rows, warnings = build_rows(d)
            text = render(rows, warnings, "t") + render_attribution(rows, load_attributions(d))
            self.assertIn("| 1cm | 2 | 2 | 2/2 (100%) | 2/2 (100%) |", text)
            self.assertIn("| 4cm | 2 | 2 | 0/2 (0%) | 0/2 (0%) |", text)
            self.assertIn("## By condition", text)
            self.assertIn("| gripper vs object xy at close (m) | 0.010 (n=2) | 0.040 (n=2) |", text)
            self.assertIn("- grasp_pose_or_frame: 2", text)
            self.assertIn("executed/planned duration [1.111]", text)


if __name__ == "__main__":
    unittest.main()


class SuiteOrchestrationTest(unittest.TestCase):
    """Runs run_suite.main end to end with uv/pixi replaced by in-process stand-ins."""

    def _fake_run(self, cmd, cwd=None, stdout=None, stderr=None):
        import argparse
        from types import SimpleNamespace

        from grasp_eval import attribute_run

        if cmd[0] == "pixi":  # perceived-scene extraction
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"objects": [{"name": "Rubik's_cube", "role": "movable",
                                                    "aabb_center": [0.51, 0.0, 0.015]}]}))
            return SimpleNamespace(returncode=0)
        if cmd[3].endswith("attribute_run.py"):
            attribute_run.main(cmd[4:])
            return SimpleNamespace(returncode=0)
        p = argparse.ArgumentParser()
        for k in ("--run-id", "--out-dir", "--variant", "--plan-json", "--target-object",
                  "--displace-dx-m", "--displace-dy-m", "--displace-yaw-deg"):
            p.add_argument(k)
        a, _ = p.parse_known_args(cmd[3:])
        self.calls.append(a)
        out = Path(a.out_dir)
        has_plan = a.variant != "1"  # variant 1: the planner fails
        plan = Path(a.plan_json) if a.plan_json else out / "server" / a.run_id / "tiptop_plan.json"
        if has_plan and not a.plan_json:
            plan.parent.mkdir(parents=True)
            plan.write_text(json.dumps(_plan()))
        dist = np.hypot(float(a.displace_dx_m or 0), float(a.displace_dy_m or 0))
        lift = bool(has_plan and dist < 0.025 and float(a.displace_yaw_deg or 0) < 40)
        rec = {"run_id": a.run_id, "target_object": a.target_object, "server_success": has_plan,
               "server_response_received": True, "executed_steps": 50 if has_plan else 0,
               "stable_lift": lift if has_plan else None, "harness_exception": None,
               "failure_category": None if lift else ("missed_grasp" if has_plan else "planning"),
               "target_source": "explicit", "server_save_dir": str(plan.parent) if has_plan else None,
               "artifacts": {"server_plan": str(plan)} if has_plan else {},
               "initial_object_positions": {"cube": [0.5, 0.0, 0.03]}}
        run_dir = out / "runs" / a.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "record.json").write_text(json.dumps(rec))
        with open(out / "runs.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return SimpleNamespace(returncode=0)

    def test_displacement_suite_end_to_end(self):
        import contextlib
        import io
        from unittest import mock

        self.calls = []
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(run_suite, "PROJECT_ROOT", Path(d)), \
                mock.patch.object(run_suite.subprocess, "run", self._fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = run_suite.main(["displacement", "--tiptop-dir", d, "--target-object", "cube", "--variants", "0", "1",
                                 "--distances-cm", "2", "4", "--directions", "x+", "y-", "--yaws-deg", "45",
                                 "--repeats", "1", "--out-name", "t"])
            out = Path(d) / "results" / "t"
            schedule = [json.loads(l) for l in (out / "schedule.jsonl").read_text().splitlines()]
            skipped = (out / "skipped.jsonl").read_text().splitlines()
            summary = (out / "summary.md").read_text()
            attrs = (out / "attribution.jsonl").read_text().splitlines()
            perceived = list((out / "perceived").glob("*.json"))
        self.assertEqual(rc, 0)
        # variant 0: fresh + 0cm + 4 displacements + 1 yaw; variant 1: only its failed fresh run
        self.assertEqual(len(schedule), 8)
        self.assertEqual(len(skipped), 6)
        replays = [c for c in self.calls if c.plan_json]
        self.assertEqual(len(replays), 6)
        self.assertEqual({Path(c.plan_json).parent.name for c in replays}, {"000_s1v0_fresh_nominal_t0"})
        self.assertEqual(len(perceived), 1)  # replays reuse the source run's perceived scene
        self.assertEqual(len(attrs), 8)  # every run that wrote a record, including the planning failure
        self.assertIn("| 2cm | 2 | 2 | 2/2 (100%) | 2/2 (100%) |", summary)
        self.assertIn("| 4cm | 2 | 2 | 0/2 (0%) | 0/2 (0%) |", summary)
        self.assertIn("6 replay run(s) not scheduled", summary)
        self.assertIn("perceived vs true object xy (m) | 0.010", summary)
        self.assertIn("- insufficient_data: 3", summary)  # stand-in runs write no trace
