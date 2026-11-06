"""End-to-end CPU tests: TaskSupervisor + TaskSimBackend on the multi-object toy world, both lift modes."""

import unittest

import numpy as np

from grasp_eval.recovery_eval import ReturnConfig
from grasp_eval.recovery_records import TaskTrialRecord
from grasp_eval.task_checkpoint import TaskCheckCriteria, TaskLimits, TaskStop, TaskSupervisor, Verdict
from grasp_eval.task_recovery_eval import PHASES, TaskSimBackend, damped_vertical_step, fill_task_record
from tests.task_fakes import REST_Z, FakeTiptopClient, PlanningError, TaskServer, TaskToyWorld

BOTH = ("reverse_approach", "vertical_ik")


def run(mode, cubes=None, behaviours=("ok",), limits=TaskLimits(), script=None, replay=None, **world_kw):
    world = TaskToyWorld(cubes=cubes, behaviours=behaviours, **world_kw)
    names = [n for n in world.object_names if n != "bowl"]
    goal = [{"objects": names, "relation": "in", "container": "bowl"}]
    server = TaskServer(world, names, script)
    client = FakeTiptopClient(server)
    b = TaskSimBackend(world, client, world.obs(), instruction="Put the cubes in the bowl.", lift_mode=mode,
                       limits=limits, criteria=TaskCheckCriteria(), return_cfg=ReturnConfig(), goal=goal,
                       replay_plan=replay)
    result = TaskSupervisor(b, limits).run()
    rec = TaskTrialRecord("T", 1, 0, 0, "Put the cubes in the bowl.", goal=goal)
    fill_task_record(rec, result, b)
    return rec, result, world, server, b


def phase_samples(b, *phases):
    ph = np.asarray(b.trace["phase"])
    return np.flatnonzero(np.isin(ph, [PHASES.index(p) for p in phases]))


class TaskBackendTest(unittest.TestCase):
    def test_held_grasp_continues_the_plan_and_succeeds(self):
        for mode in BOTH:
            with self.subTest(mode=mode):
                rec, r, world, server, b = run(mode)
                self.assertEqual(r.stop, TaskStop.PLAN_EXECUTED, r.detail)
                self.assertEqual([c["verdict"] for c in rec.checkpoints], ["held"])
                self.assertTrue(rec.task_success, rec.task_check)
                self.assertEqual((rec.num_retries, server.calls), (0, 1))
                self.assertTrue(world.in_bowl("cube"))
                self.assertGreater(rec.checkpoints[0]["lift_steps"], 0)

    def test_no_lateral_motion_and_low_object_while_unconfirmed(self):
        for mode in BOTH:
            with self.subTest(mode=mode):
                rec, r, world, server, b = run(mode)
                idx = phase_samples(b, "lift", "hold")
                g = np.asarray(b.trace["gripper_pos"])[idx]
                self.assertLess(float(np.max(np.linalg.norm(g[:, :2] - g[0, :2], axis=1))), 1e-6)
                k = b.names.index("cube")
                obj_z = np.asarray(b.trace["objects_center"])[idx, k, 2]
                cap = TaskCheckCriteria().hand_rise_cap_m
                self.assertLessEqual(float(obj_z.max() - REST_Z), cap + 1e-6)

    def test_clean_miss_retries_with_a_fresh_plan_and_succeeds(self):
        for mode in BOTH:
            with self.subTest(mode=mode):
                rec, r, world, server, b = run(mode, behaviours=("miss", "ok"), replay="/baseline/tiptop_plan.json")
                self.assertEqual([c["verdict"] for c in rec.checkpoints], ["empty", "held"])
                self.assertEqual([c["action"] for c in rec.checkpoints], ["retry", "continue"])
                self.assertTrue(rec.task_success)
                self.assertTrue(rec.recovered)
                self.assertEqual((rec.num_retries, server.calls, rec.num_planning_calls), (1, 2, 2))
                self.assertEqual([p["source"] for p in rec.plans], ["replay", "server"])
                # the retry plans from the observation pose (the trial's start joints)
                np.testing.assert_allclose(rec.plans[1]["start_joint_pos"], np.zeros(7), atol=1e-6)
                self.assertTrue(any(e["what"] == "recovered_to_observation_pose" for e in rec.events))

    def test_slip_sets_the_object_down_before_opening(self):
        for mode in BOTH:
            with self.subTest(mode=mode):
                rec, r, world, server, b = run(mode, behaviours=("slip", "ok"), slip_after=8)
                self.assertEqual(rec.checkpoints[0]["verdict"], "slipped", rec.checkpoints[0]["reason"])
                self.assertTrue(rec.task_success)
                # nothing was released from height: an open with the object in hand happened only low
                crit = TaskCheckCriteria()
                for ev in world.open_events[:-1]:  # the last open is the deliberate place in the bowl
                    self.assertLessEqual(ev["rise"], crit.set_down_rise_m + 0.005, ev)

    def test_multi_object_miss_replans_only_the_remaining_object(self):
        cubes = {"a": [0.45, -0.1], "b": [0.55, 0.1]}
        for mode in BOTH:
            with self.subTest(mode=mode):
                rec, r, world, server, b = run(mode, cubes=cubes, behaviours=("ok", "miss", "ok"))
                self.assertEqual(server.planned, [["a", "b"], ["b"]])
                self.assertEqual([(c["attempt"], c["unit"], c["verdict"]) for c in rec.checkpoints],
                                 [(0, 0, "held"), (0, 1, "empty"), (1, 0, "held")])
                self.assertTrue(rec.task_success, rec.task_check)

    def test_checkpoint_only_control_never_retries(self):
        rec, r, world, server, b = run("reverse_approach", behaviours=("miss", "ok"), limits=TaskLimits(max_retries=0))
        self.assertEqual((rec.num_retries, server.calls), (0, 1))
        self.assertEqual(rec.checkpoints[0]["action"], "continue")
        self.assertFalse(rec.task_success)  # like the baseline, the plan carried nothing
        self.assertEqual(rec.mode, "task_checkpoint_only")

    def test_drop_during_carry_is_recorded_not_acted_on(self):
        rec, r, world, server, b = run("reverse_approach", behaviours=("drop",))
        self.assertEqual(rec.checkpoints[0]["verdict"], "held")
        self.assertTrue(any(e["what"] == "dropped_during_carry" for e in rec.events))
        self.assertEqual(r.stop, TaskStop.PLAN_EXECUTED)
        self.assertFalse(rec.task_success)

    def test_lost_object_stops_the_trial(self):
        rec, r, world, server, b = run("vertical_ik", behaviours=("lost",))
        self.assertEqual(r.stop, TaskStop.OBJECT_LOST)
        self.assertEqual(rec.num_retries, 0)

    def test_failed_replan_keeps_the_scored_state(self):
        rec, r, world, server, b = run("reverse_approach", behaviours=("miss",),
                                       script=[None, PlanningError("cuTAMP failed")])
        self.assertEqual(r.stop, TaskStop.RETRY_NO_PLAN)
        self.assertFalse(rec.task_success)

    def test_task_state_is_also_scored_at_the_baseline_episode_end(self):
        rec, r, world, server, b = run("reverse_approach", behaviours=("miss", "ok"),
                                       limits=TaskLimits(baseline_episode_s=3.0))
        self.assertTrue(rec.task_success)
        self.assertFalse(rec.task_success_at_baseline_limit)

    def test_unsupported_plan_runs_unchanged_without_checkpoints(self):
        world = TaskToyWorld()
        server = TaskServer(world, ["cube"])
        client = FakeTiptopClient(server)
        b = TaskSimBackend(world, client, world.obs(), instruction="x", lift_mode="reverse_approach",
                           limits=TaskLimits(), criteria=TaskCheckCriteria(), return_cfg=ReturnConfig(),
                           goal=[{"objects": ["cube"], "relation": "in", "container": "bowl"}])
        orig = server.__call__

        def no_close(request):
            p = orig(request)
            p["steps"] = [s for s in p["steps"] if not (s.get("type") == "gripper" and s.get("action") == "close")]
            return p

        client._server = no_close
        r = TaskSupervisor(b, TaskLimits()).run()
        self.assertEqual((r.stop, r.checkpoints), (TaskStop.PLAN_EXECUTED, []))
        self.assertIn("units_error", b.plans[0])


class IKStepTest(unittest.TestCase):
    def test_damped_step_moves_along_z_only(self):
        rng = np.random.default_rng(0)
        J = rng.normal(size=(6, 7))
        q = damped_vertical_step(J, np.zeros(7), 0.002, damping=1e-4)
        np.testing.assert_allclose(J @ q, [0, 0, 0.002, 0, 0, 0], atol=1e-6)

    def test_step_is_clipped_and_limited(self):
        J = np.zeros((6, 7))
        J[2, 0] = 1e-3  # nearly singular: a naive solve would explode
        q = damped_vertical_step(J, np.zeros(7), 0.01, lower=-0.01 * np.ones(7), upper=0.01 * np.ones(7))
        self.assertLessEqual(float(np.max(np.abs(q))), 0.01 + 1e-12)


if __name__ == "__main__":
    unittest.main()
