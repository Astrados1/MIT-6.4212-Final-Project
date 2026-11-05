import unittest

import numpy as np

from grasp_eval.checkpoint import (
    WRONG_OBJECT_REASON, CheckpointLimits, CheckResult, OracleCheckCriteria, OracleStateDetector, OracleStatePacket,
    Outcome, RecoverySupervisor, StepOutcome, Stop,
)


# --------------------------------------------------------------------------- oracle detector

DT = 1 / 15


def window(target_follows: float, *, drop_at=None, other=None, hold=15, close=21, retract=4, z0=0.03):
    """Close dwell, a 5 cm retraction, then a hold. target_follows = fraction of the hand rise."""
    n = close + retract + hold
    t = np.arange(1, n + 1) * DT
    grip = np.tile([0.5, 0.0, z0], (n, 1)).astype(float)
    lift = np.r_[np.zeros(close), np.linspace(0.05 / retract, 0.05, retract), np.full(hold, 0.05)]
    grip[:, 2] += lift
    obj = np.tile([0.5, 0.0, z0], (n, 1)).astype(float)
    obj[:, 2] += target_follows * lift
    if drop_at is not None:
        obj[drop_at:, 2] = z0
    return OracleStatePacket(t=t, target_pos=obj, gripper_pos=grip, gripper_cmd=np.ones(n),
                             retract_start=close, hold_start=close + retract, other_pos=other or {})


class OracleDetectorTest(unittest.TestCase):
    det = OracleStateDetector()

    def test_holding(self):
        r = self.det.check(window(1.0))
        self.assertEqual(r.outcome, Outcome.HOLDING_TARGET, r.reason)
        self.assertTrue(r.privileged)
        self.assertEqual(r.source, "oracle-state")
        self.assertAlmostEqual(r.evidence["gripper_rise_m"], 0.05, places=3)

    def test_held_cube_at_finger_origin_offset(self):
        # Nov-5 smoke: a clearly held cube sits ~15.5 cm from the inner-finger origin midpoint.
        # The frozen 0.20 m limit accepts it; the old 0.10 m limit called every real grasp "not held".
        p = window(1.0)
        p.target_pos = p.target_pos + np.array([0.0, 0.0, -0.155])
        self.assertEqual(self.det.check(p).outcome, Outcome.HOLDING_TARGET)
        old = OracleStateDetector(OracleCheckCriteria(max_obj_to_gripper_m=0.10)).check(p)
        self.assertNotEqual(old.outcome, Outcome.HOLDING_TARGET)

    def test_empty(self):
        r = self.det.check(window(0.0))
        self.assertEqual(r.outcome, Outcome.FAILED_EMPTY, r.reason)

    def test_lift_then_drop_is_not_held_now(self):
        # Lifted with the hand, then fell back before the hold ended: never "currently held".
        p = window(1.0, drop_at=27)
        r = self.det.check(p)
        self.assertNotEqual(r.outcome, Outcome.HOLDING_TARGET)

    def test_dropped_and_settled_early_counts_as_empty(self):
        r = self.det.check(window(1.0, drop_at=22))  # fell during the retraction, settled through the hold
        self.assertEqual(r.outcome, Outcome.FAILED_EMPTY, r.reason)

    def test_partial_follow_is_uncertain(self):
        r = self.det.check(window(0.3))
        self.assertEqual(r.outcome, Outcome.UNCERTAIN)

    def test_wrong_object(self):
        p = window(0.0)
        p.other_pos = {"mug": p.gripper_pos + np.array([0.0, 0.0, -0.02])}
        r = self.det.check(p)
        self.assertEqual((r.outcome, r.reason), (Outcome.UNCERTAIN, WRONG_OBJECT_REASON))

    def test_disturbed_neighbour_makes_empty_ambiguous(self):
        p = window(0.0)
        mug = np.tile([0.53, 0.0, 0.03], (len(p.t), 1))
        mug[-10:, 0] += 0.02  # pushed 2 cm, close to the hand
        p.other_pos = {"mug": mug}
        r = self.det.check(p)
        self.assertEqual((r.outcome, r.reason), (Outcome.UNCERTAIN, "disturbed_object_near_hand"))

    def test_gripper_must_stay_closed_and_hand_must_rise(self):
        p = window(1.0)
        p.gripper_cmd[-1] = 0.0
        self.assertEqual(self.det.check(p).outcome, Outcome.UNCERTAIN)
        p = window(1.0)
        p.gripper_pos[:, 2] = 0.03
        self.assertEqual(self.det.check(p).reason, "retraction_did_not_raise_gripper")

    def test_finger_evidence_when_enabled(self):
        det = OracleStateDetector(OracleCheckCriteria(empty_min_finger_closure=0.9))
        p = window(0.0)
        self.assertEqual(det.check(p).reason, "finger_evidence_required_but_missing")
        p.finger_closure = np.full(len(p.t), 0.6)
        self.assertEqual(det.check(p).reason, "fingers_blocked_but_target_not_held")
        p.finger_closure = np.full(len(p.t), 1.0)
        self.assertEqual(det.check(p).outcome, Outcome.FAILED_EMPTY)


# --------------------------------------------------------------------------- supervisor


class ScriptedBackend:
    """Each attempt: (plan outcome, execute outcome, check outcome). Retry outcomes in order."""

    def __init__(self, attempts, retry=None, sim_per_phase=1.0):
        self.script = attempts
        self.retry = list(retry or [])
        self.t = 0.0
        self.n_closes = 0
        self.sim_per_phase = sim_per_phase
        self.calls = []

    def _adv(self, name, i):
        self.calls.append((name, i))
        self.t += self.sim_per_phase

    def observe(self, i):
        self._adv("observe", i)
        return StepOutcome.success()

    def plan(self, i):
        self._adv("plan", i)
        r = self.script[i][0]
        if isinstance(r, BaseException):
            raise r
        return r

    def execute_grasp(self, i):
        self._adv("execute", i)
        r = self.script[i][1]
        if r.ok:
            self.n_closes += 1
        return r

    def check(self, i):
        self._adv("check", i)
        return self.script[i][2]

    def prepare_retry(self, i):
        self._adv("retry", i)
        return self.retry.pop(0) if self.retry else StepOutcome.success()

    def sim_time(self):
        return self.t

    def closes(self):
        return self.n_closes


OK = StepOutcome.success()
HELD = CheckResult(Outcome.HOLDING_TARGET, "held", source="oracle-state")
EMPTY = CheckResult(Outcome.FAILED_EMPTY, "empty", source="oracle-state")
UNSURE = CheckResult(Outcome.UNCERTAIN, "ambiguous", source="oracle-state")
WRONG = CheckResult(Outcome.UNCERTAIN, WRONG_OBJECT_REASON, source="oracle-state")


def run(script, limits=CheckpointLimits(), retry=None, clock=None, **kw):
    b = ScriptedBackend(script, retry, **kw)
    sup = RecoverySupervisor(b, limits, **({"wall_clock": clock} if clock else {}))
    return sup.run(), b


class SupervisorTest(unittest.TestCase):
    def test_success_on_first_attempt(self):
        r, b = run([(OK, OK, HELD)])
        self.assertEqual(r.stop, Stop.SUCCESS)
        self.assertTrue(r.success and r.first_attempt_success)
        self.assertEqual((r.planning_calls, r.closes, len(r.attempts)), (1, 1, 1))
        self.assertFalse(r.retry_triggered)
        self.assertNotIn("retry", [c[0] for c in b.calls])

    def test_failure_then_success(self):
        r, b = run([(OK, OK, EMPTY), (OK, OK, HELD)])
        self.assertEqual(r.stop, Stop.SUCCESS)
        self.assertFalse(r.first_attempt_success)
        self.assertTrue(r.retry_triggered)
        self.assertEqual((r.planning_calls, r.closes, len(r.attempts)), (2, 2, 2))
        self.assertEqual([c[0] for c in b.calls],
                         ["observe", "plan", "execute", "check", "retry", "observe", "plan", "execute", "check"])

    def test_two_misses_exhaust_attempts(self):
        r, _ = run([(OK, OK, EMPTY), (OK, OK, EMPTY)])
        self.assertEqual(r.stop, Stop.ATTEMPTS_EXHAUSTED)
        self.assertFalse(r.success)
        self.assertEqual(r.closes, 2)

    def test_one_attempt_control_never_retries(self):
        r, b = run([(OK, OK, EMPTY)], CheckpointLimits(max_attempts=1, max_planning_calls=1))
        self.assertEqual(r.stop, Stop.ATTEMPTS_EXHAUSTED)
        self.assertNotIn("retry", [c[0] for c in b.calls])

    def test_uncertain_and_wrong_object_stop_without_retry(self):
        for check, stop in ((UNSURE, Stop.UNCERTAIN), (WRONG, Stop.WRONG_OBJECT)):
            with self.subTest(stop):
                r, b = run([(OK, OK, check), (OK, OK, HELD)])
                self.assertEqual(r.stop, stop)
                self.assertNotIn("retry", [c[0] for c in b.calls])

    def test_planning_and_infrastructure_failures(self):
        r, _ = run([(StepOutcome.fail(Stop.NO_PLAN, "cuTAMP failed"), OK, HELD)])
        self.assertEqual((r.stop, r.planning_calls, r.closes), (Stop.NO_PLAN, 1, 0))
        r, _ = run([(StepOutcome.fail(Stop.INFRASTRUCTURE, "conn refused"), OK, HELD)])
        self.assertEqual(r.stop, Stop.INFRASTRUCTURE)
        r, _ = run([(RuntimeError("boom"), OK, HELD)])
        self.assertEqual(r.stop, Stop.INFRASTRUCTURE)
        self.assertIn("RuntimeError", r.harness_exception)
        # A failed second request is a planning failure, not a free extra attempt.
        r, _ = run([(OK, OK, EMPTY), (StepOutcome.fail(Stop.NO_PLAN, "x"), OK, HELD)])
        self.assertEqual((r.stop, r.planning_calls), (Stop.NO_PLAN, 2))

    def test_planning_budget(self):
        r, _ = run([(OK, OK, EMPTY), (OK, OK, HELD)], CheckpointLimits(max_attempts=2, max_planning_calls=1))
        self.assertEqual(r.stop, Stop.PLANNING_BUDGET_EXHAUSTED)

    def test_extra_close_consumes_budget(self):
        b_script = [(OK, OK, EMPTY), (OK, OK, HELD)]
        b = ScriptedBackend(b_script)
        orig = b.execute_grasp

        def double_close(i):
            r = orig(i)
            b.n_closes += 1  # an accidental second close in the same attempt
            return r

        b.execute_grasp = double_close
        r = RecoverySupervisor(b, CheckpointLimits()).run()
        self.assertEqual((r.stop, r.closes), (Stop.ATTEMPTS_EXHAUSTED, 2))

    def test_sim_time_limit(self):
        r, b = run([(OK, OK, EMPTY), (OK, OK, HELD)], CheckpointLimits(max_sim_s=4.5))
        self.assertEqual(r.stop, Stop.SIM_TIME_LIMIT)
        self.assertNotIn(("plan", 1), b.calls)

    def test_wall_time_limit(self):
        ticks = iter(range(0, 10_000, 100))
        r, _ = run([(OK, OK, EMPTY), (OK, OK, HELD)], CheckpointLimits(max_wall_s=350), clock=lambda: next(ticks))
        self.assertEqual(r.stop, Stop.WALL_TIME_LIMIT)

    def test_return_failure_and_interrupt(self):
        r, _ = run([(OK, OK, EMPTY), (OK, OK, HELD)], retry=[StepOutcome.fail(Stop.RETURN_FAILED, "bumped")])
        self.assertEqual((r.stop, r.detail), (Stop.RETURN_FAILED, "bumped"))
        r, _ = run([(KeyboardInterrupt(), OK, HELD)])
        self.assertEqual(r.stop, Stop.INTERRUPTED)
        self.assertFalse(r.success)

    def test_limits_validation_and_mode(self):
        with self.assertRaises(ValueError):
            CheckpointLimits(max_attempts=0)
        self.assertEqual(CheckpointLimits(max_attempts=1).mode, "checkpoint_one_attempt")
        self.assertEqual(CheckpointLimits().mode, "checkpoint_recovery_2")


if __name__ == "__main__":
    unittest.main()
