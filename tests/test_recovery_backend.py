"""SimRecoveryBackend + RecoverySupervisor end to end on CPU (toy world, upstream-faithful client)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_eval.checkpoint import CheckpointLimits, OracleStateDetector, RecoverySupervisor, Stop
from grasp_eval.recovery_eval import (
    PHASES, ReturnConfig, SimRecoveryBackend, fill_trial_record, joint_interpolation, parse_args, request_fingerprint,
)
from grasp_eval.recovery_records import RecoveryTrialRecord
from grasp_eval.recovery_summary import load, render
from grasp_eval.records import append_jsonl
from grasp_eval.scoring import LiftCriteria
from tests.recovery_fakes import FakeTiptopClient, PlanningError, ToyWorld, scene_plan


class Server:
    """Plans from the current (true) cube position and the request's q_init, like a perfect perceiver."""

    def __init__(self, world, script=None, save_dirs=None):
        self.world = world
        self.script = list(script or [])  # per call: None (plan), an exception, or a callable(request)
        self.calls = 0
        self.save_dirs = save_dirs

    def __call__(self, request):
        i = self.calls
        self.calls += 1
        item = self.script[i] if i < len(self.script) else None
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(request)
        sd = self.save_dirs[i] if self.save_dirs else f"/srv/out/{i}"
        return scene_plan(self.world.pos["cube"], request["q_init"], save_dir=sd)


def make(outcomes=(True,), limits=CheckpointLimits(), script=None, save_dirs=None, displacement=None, **world_kw):
    world = ToyWorld(outcomes, **{k: v for k, v in world_kw.items() if k == "episode_steps"})
    for k, v in world_kw.items():
        if k != "episode_steps":
            setattr(world, k, v)
    server = Server(world, script, save_dirs)
    client = FakeTiptopClient(server)
    backend = SimRecoveryBackend(
        world, client, world.obs(), trial_id="T", instruction="Put the Rubik's cube in the bowl.", target="cube",
        target_label="rubiks_cube", limits=limits, detector=OracleStateDetector(), lift_criteria=LiftCriteria(),
        return_cfg=ReturnConfig(), displacement=displacement)
    return world, server, client, backend


def run(**kw):
    world, server, client, backend = make(**kw)
    result = RecoverySupervisor(backend, backend.limits).run()
    return result, world, server, client, backend


class RecoveryBackendTest(unittest.TestCase):
    def test_success_first_attempt_stops_before_transport(self):
        r, world, server, client, b = run()
        self.assertEqual(r.stop, Stop.SUCCESS, r.detail)
        self.assertEqual((r.planning_calls, r.closes, server.calls), (1, 1, 1))
        a = b.attempts[0]
        self.assertTrue(a.retraction_applied)
        self.assertEqual(a.outcome, "holding_target")
        self.assertTrue(a.outcome_privileged)
        self.assertEqual(a.segment["retraction_index"], 3)
        # Never executed transport: the cube is held 5 cm above where it started, not over the bowl.
        np.testing.assert_allclose(world.pos["cube"], [0.5, 0.0, 0.08], atol=1e-6)
        phases = {PHASES[p] for p in b.trace["phase"]}
        self.assertEqual(phases, {"approach", "close", "retract", "hold"})
        self.assertEqual(a.hold_steps, 15)
        self.assertIsNone(r.harness_exception)

    def test_failure_then_success_replans_from_current_scene(self):
        r, world, server, client, b = run(outcomes=(False, True))
        self.assertEqual(r.stop, Stop.SUCCESS, r.detail)
        self.assertFalse(r.first_attempt_success)
        self.assertEqual((r.planning_calls, r.closes, server.calls, len(b.attempts)), (2, 2, 2, 2))
        a0, a1 = b.attempts
        self.assertEqual(a0.outcome, "failed_empty")
        self.assertEqual(a1.outcome, "holding_target")
        # Fresh request: client reconnected, new observation frame, q_init = actual current joints.
        self.assertEqual(client.connections, 2)
        self.assertNotEqual(client.requests[0]["frame"], client.requests[1]["frame"])
        np.testing.assert_allclose(client.requests[1]["q_init"], a1.start_joint_pos, atol=1e-6)
        self.assertEqual(a1.request["frame"], client.requests[1]["frame"])
        self.assertNotEqual(a0.server_save_dir, a1.server_save_dir)
        # Same instruction both times.
        self.assertEqual({q["task"] for q in client.requests}, {"Put the Rubik's cube in the bowl."})
        # Retry phases happened in between, the gripper was opened, and the old plan was dropped.
        phases = [PHASES[p] for p in b.trace["phase"]]
        self.assertIn("retry_open", phases)
        self.assertLess(phases.index("retry_settle"), len(phases) - 1)
        self.assertTrue(a0.retry["final_joint_error_max_rad"] <= 0.05)
        # One continuous clock: attempt 1 starts where the retry ended.
        self.assertGreater(a1.started_sim_s, a0.started_sim_s)
        self.assertAlmostEqual(a1.started_sim_s, a0.retry["end_sim_s"], places=4)

    def test_retry_never_resets_the_scene(self):
        disp = {"dx_m": 0.02, "dy_m": 0.0, "yaw_deg": 0.0}
        r, world, server, client, b = run(outcomes=(False, True), displacement=disp)
        a0, a1 = b.attempts
        self.assertTrue(a0.displacement["applied"])
        self.assertIsNone(a1.displacement)  # applied once, before attempt 1 only
        # The plan for attempt 0 targeted the old pose -> miss. Attempt 1 sees the displaced cube.
        np.testing.assert_allclose(a1.object_positions["cube"], [0.52, 0.0, 0.03], atol=1e-6)
        self.assertEqual(r.stop, Stop.SUCCESS, r.detail)
        t = np.asarray(b.trace["t"])
        self.assertTrue(np.all(np.diff(t) > 0))  # time never goes back

    def test_two_misses_terminate(self):
        r, world, server, client, b = run(outcomes=(False, False))
        self.assertEqual(r.stop, Stop.ATTEMPTS_EXHAUSTED)
        self.assertEqual((r.closes, server.calls), (2, 2))
        self.assertEqual([a.outcome for a in b.attempts], ["failed_empty", "failed_empty"])

    def test_one_attempt_control(self):
        r, world, server, client, b = run(outcomes=(False, True), limits=CheckpointLimits(1, 1))
        self.assertEqual(r.stop, Stop.ATTEMPTS_EXHAUSTED)
        self.assertEqual((server.calls, len(b.attempts)), (1, 1))
        self.assertNotIn("retry_open", {PHASES[p] for p in b.trace["phase"]})

    def test_planning_failure_and_api_failure(self):
        r, *_ = run(script=[PlanningError("cuTAMP failed to find a plan: x")])
        self.assertEqual(r.stop, Stop.NO_PLAN)
        self.assertEqual(r.closes, 0)
        r, *_ = run(script=[ConnectionError("refused")])
        self.assertEqual(r.stop, Stop.INFRASTRUCTURE)
        r, _, server, _, b = run(outcomes=(False, True), script=[None, PlanningError("no plan")])
        self.assertEqual((r.stop, r.planning_calls), (Stop.NO_PLAN, 2))
        self.assertEqual(b.attempts[1].server_error, "no plan")

    def test_unsupported_plan_is_refused_before_any_motion(self):
        def multi(req):
            p = scene_plan([0.5, 0, 0.03], req["q_init"])
            aux = scene_plan([0.6, 0, 0.03], req["q_init"], target="mug")
            p["steps"] = aux["steps"][:8] + p["steps"][1:]
            return p

        r, world, *_ , b = run(script=[multi])
        self.assertEqual(r.stop, Stop.UNSUPPORTED_PLAN)
        self.assertEqual(len(b.trace["t"]), 0)
        self.assertIn("only single-grasp plans", b.attempts[0].segment_error)

    def test_reused_server_output_dir_is_infrastructure(self):
        r, *_ = run(outcomes=(False, True), save_dirs=["/srv/out/same", "/srv/out/same"])
        self.assertEqual(r.stop, Stop.INFRASTRUCTURE)
        self.assertIn("reused", r.detail)

    def test_return_disturbance_stops_trial(self):
        r, *_ = run(outcomes=(False, True), bump_bowl_during_return=True)
        self.assertEqual(r.stop, Stop.RETURN_FAILED)

    def test_slip_after_lift_is_not_success(self):
        # Held through the retraction, then slips out during the confirmation hold.
        r, world, *_ , b = run(outcomes=(True,), drop_after_steps_held=30)
        self.assertEqual(r.stop, Stop.UNCERTAIN)
        self.assertEqual(b.attempts[0].outcome, "uncertain")
        self.assertEqual(b.attempts[0].checkpoint["reason"], "target_still_moving_at_checkpoint")

    def test_time_budget(self):
        r, *_ = run(outcomes=(False, True), limits=CheckpointLimits(max_sim_s=6.0))
        self.assertIn(r.stop, (Stop.SIM_TIME_LIMIT,))
        r, *_ = run(limits=CheckpointLimits(max_sim_s=1.0))
        self.assertEqual(r.stop, Stop.SIM_TIME_LIMIT)

    def test_records_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for i, (outcomes, lim) in enumerate([((False, True), CheckpointLimits()),
                                                 ((False, True), CheckpointLimits(1, 1)),
                                                 ((True,), CheckpointLimits())]):
                r, world, server, client, b = run(outcomes=outcomes, limits=lim)
                rec = RecoveryTrialRecord(trial_id=f"T{i}", scene=1, variant=0, trial=i, instruction="x",
                                          target_object="cube", mode=lim.mode)
                for a in b.attempts:
                    a.trial_id = rec.trial_id
                fill_trial_record(rec, r, b)
                for a in b.attempts:
                    append_jsonl(out / "attempts.jsonl", a.to_dict())
                append_jsonl(out / "recovery_trials.jsonl", rec.to_dict())
            json.dumps(rec.to_dict())  # serializable
            append_jsonl(out / "schedule.jsonl", {"trial_id": "T9", "mode": "checkpoint_recovery_2",
                                                  "detector": "oracle-state", "instruction": "x"})
            trials, attempts, warnings = load(out)
            text = render(trials, attempts, warnings, "t")
            self.assertIn("oracle-state", text)
            self.assertIn("privileged simulator state", text)
            self.assertIn("recovery success, among retry-eligible | 1/1", text)
            self.assertIn("no_record", text)  # crashed trial counted, not dropped
            recs = {t["trial_id"]: t for t in trials}
            self.assertTrue(recs["T0"]["final_success"] and not recs["T0"]["first_attempt_success"])
            self.assertEqual(recs["T0"]["num_attempts"], 2)
            self.assertFalse(recs["T1"]["final_success"])
            self.assertTrue(recs["T2"]["first_attempt_success"])
            self.assertEqual(len(attempts["T0"]), 2)


class HelpersTest(unittest.TestCase):
    def test_joint_interpolation_speed_and_endpoint(self):
        q0, q1 = np.zeros(7), np.r_[1.0, np.zeros(6)]
        path = joint_interpolation(q0, q1, 0.5, 1 / 15)
        np.testing.assert_allclose(path[-1], q1)
        speeds = np.abs(np.diff(np.vstack([q0] + path), axis=0)).max(axis=1) * 15
        self.assertLessEqual(speeds.max(), 0.5 + 1e-9)

    def test_request_fingerprint(self):
        fp = request_fingerprint({"q_init": np.arange(7), "rgb": np.zeros((2, 2, 3), np.uint8),
                                  "depth": np.ones((2, 2), np.float32), "world_from_cam": np.eye(4), "task": "x"})
        self.assertEqual(len(fp["rgb_sha256_16"]), 16)
        self.assertEqual(fp["q_init"][-1], 6.0)

    def test_cli_defaults(self):
        a = parse_args(["--scene", "1", "--variant", "0", "--instruction", "x", "--target-object", "cube",
                        "--out-dir", "o", "--max-attempts", "1"])
        self.assertEqual((a.max_attempts, a.max_planning_calls, a.detector), (1, 1, "oracle-state"))


if __name__ == "__main__":
    unittest.main()
