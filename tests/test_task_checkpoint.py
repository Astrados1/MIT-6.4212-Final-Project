"""CPU tests for the task-level checkpoint: the lift monitor's verdicts and the trial supervisor."""

import unittest

import numpy as np

from grasp_eval.task_checkpoint import (
    LiftMonitor, Step, TaskCheckCriteria, TaskLimits, TaskStop, TaskSupervisor, Verdict,
)

DT = 1 / 15
C = TaskCheckCriteria()


def run_lift(obj_of_rise, n=60, speed=0.03, others=None, cap=None):
    """Feed a lift where the hand rises `speed` m/s; obj_of_rise(hand_rise) -> object rise. Returns monitor."""
    m = LiftMonitor(C, DT, [0.5, 0.0, 0.25], [0.5, 0.0, 0.08],
                    others_at_rest={k: [0.6, 0.1, 0.08 + f(0.0)] for k, f in (others or {}).items()})
    hand_rise = 0.0
    for i in range(n):
        if m.phase == "lift":
            hand_rise = min(hand_rise + speed * DT, cap if cap is not None else 1.0)
        hand = [0.5, 0.0, 0.25 + hand_rise]
        obj = [0.5, 0.0, 0.08 + obj_of_rise(hand_rise, i)]
        o = {k: [0.6, 0.1, 0.08 + f(hand_rise)] for k, f in (others or {}).items()}
        if m.update(hand, obj, o) is not None:
            return m
    return m


class LiftMonitorTest(unittest.TestCase):
    def test_lagging_grasp_is_held(self):
        # Nov-6 smoke: the cube trails the hand by ~2 cm while the grip settles, then follows.
        m = run_lift(lambda h, i: max(0.0, h - 0.02), n=200)
        self.assertEqual(m.verdict, Verdict.HELD, m.reason)
        self.assertGreaterEqual(m.max_obj_rise, C.held_min_object_rise_m)

    def test_clean_miss_is_empty_early(self):
        m = run_lift(lambda h, i: 0.0, n=200)
        self.assertEqual(m.verdict, Verdict.EMPTY, m.reason)
        hand_rise = m.profile[-1][1]
        self.assertLess(hand_rise, 0.08)  # decided soon after the 5 cm hand rise, not at the cap
        self.assertGreaterEqual(hand_rise, C.miss_min_hand_rise_m)

    def test_nudged_object_that_settles_is_still_empty(self):
        # baseline s1v0: a miss nudged the cube up, then it sat still
        m = run_lift(lambda h, i: min(0.008, h), n=200)
        self.assertEqual(m.verdict, Verdict.EMPTY, m.reason)

    def test_slip_after_following(self):
        def obj(h, i):
            r = max(0.0, h - 0.02)
            return r if r < 0.03 else 0.03 - (h - 0.05) * 2  # follows to 3 cm, then falls back
        m = run_lift(obj, n=200)
        self.assertEqual(m.verdict, Verdict.SLIPPED, m.reason)

    def test_nudged_but_never_following_is_uncertain_at_the_cap(self):
        # raised 1.5 cm (more than a clean miss allows) but never followed the hand
        m = run_lift(lambda h, i: min(0.015, h), n=400, cap=0.10)
        self.assertIsNone(m.verdict)
        self.assertEqual(m.at_lift_end(), Verdict.UNCERTAIN)

    def test_blocked_hand_is_uncertain_not_empty(self):
        m = LiftMonitor(C, DT, [0.5, 0.0, 0.25], [0.5, 0.0, 0.08])
        for _ in range(30):
            m.update([0.5, 0.0, 0.25], [0.5, 0.0, 0.08])  # hand stalled on the object
        self.assertEqual(m.at_lift_end(), Verdict.UNCERTAIN)
        self.assertIn("blocked", m.reason)

    def test_lost_object(self):
        m = run_lift(lambda h, i: -0.08 if i > 5 else 0.0, n=20)
        self.assertEqual(m.verdict, Verdict.LOST)

    def test_other_object_following_is_uncertain(self):
        m = run_lift(lambda h, i: 0.0, n=200, others={"mug": lambda h: max(0.0, h - 0.01)})
        self.assertEqual(m.verdict, Verdict.UNCERTAIN)
        self.assertIn("mug", m.reason)

    def test_hold_rejects_an_object_that_sinks(self):
        def obj(h, i):
            return max(0.0, h - 0.02) - (0.0005 * max(0, i - 31))  # follows; sinks ~7 mm through the hold
        m = run_lift(obj, n=400)
        self.assertEqual(m.verdict, Verdict.SLIPPED, m.reason)
        self.assertIn("during the hold", m.reason)


class FakeBackend:
    """Scripted world: verdicts[(attempt, unit)] and units per attempt."""

    def __init__(self, units_per_attempt, verdicts, plan_fail_at=None, recover_ok=True, labels=None):
        self.units = units_per_attempt
        self.verdicts = verdicts
        self.plan_fail_at = plan_fail_at
        self.recover_ok = recover_ok
        self.labels = labels or {}
        self.calls: list[tuple] = []
        self.t = 0.0

    def sim_time(self):
        return self.t

    def plan(self, attempt):
        self.calls.append(("plan", attempt))
        self.t += 1
        if attempt == self.plan_fail_at:
            return Step.fail(TaskStop.RETRY_NO_PLAN, "no plan")
        n = self.units[min(attempt, len(self.units) - 1)]
        return Step.success(units=list(range(n)), unit_labels=self.labels.get(attempt, [f"obj{u}" for u in range(n)]))

    def run_to_checkpoint(self, attempt, unit):
        self.calls.append(("approach", attempt, unit))
        self.t += 5
        return Step.success()

    def lift_and_check(self, attempt, unit):
        self.calls.append(("lift", attempt, unit))
        self.t += 3
        return {"verdict": self.verdicts.get((attempt, unit), Verdict.HELD), "reason": "scripted"}

    def resume(self, attempt, unit):
        self.calls.append(("resume", attempt, unit))
        return Step.success()

    def recover(self, attempt, unit, verdict):
        self.calls.append(("recover", attempt, unit, verdict.value))
        self.t += 4
        return Step.success() if self.recover_ok else Step.fail(TaskStop.RECOVERY_FAILED, "blocked")

    def run_to_end(self, attempt):
        self.calls.append(("end", attempt))
        self.t += 2
        return Step.success()


class TaskSupervisorTest(unittest.TestCase):
    def test_all_held_runs_the_plan_to_the_end(self):
        b = FakeBackend([3], {})
        r = TaskSupervisor(b, TaskLimits()).run()
        self.assertEqual(r.stop, TaskStop.PLAN_EXECUTED)
        self.assertEqual([c["action"] for c in r.checkpoints], ["continue"] * 3)
        self.assertEqual((r.retries, r.attempts, r.planning_calls), (0, 1, 1))
        self.assertEqual(b.calls[-1], ("end", 0))

    def test_miss_on_second_grasp_replans_the_remaining_task(self):
        b = FakeBackend([3, 2], {(0, 1): Verdict.EMPTY})
        r = TaskSupervisor(b, TaskLimits()).run()
        self.assertEqual(r.stop, TaskStop.PLAN_EXECUTED)
        self.assertIn(("recover", 0, 1, "empty"), b.calls)
        self.assertNotIn(("lift", 0, 2), b.calls)  # the old plan's third grasp never runs
        self.assertEqual([(c["attempt"], c["unit"], c["action"]) for c in r.checkpoints],
                         [(0, 0, "continue"), (0, 1, "retry"), (1, 0, "continue"), (1, 1, "continue")])
        self.assertEqual((r.retries, r.attempts, r.planning_calls), (1, 2, 2))

    def test_uncertain_never_retries(self):
        b = FakeBackend([1], {(0, 0): Verdict.UNCERTAIN})
        r = TaskSupervisor(b, TaskLimits()).run()
        self.assertEqual(r.retries, 0)
        self.assertEqual(r.checkpoints[0]["action"], "continue")

    def test_checkpoint_only_control_never_retries(self):
        b = FakeBackend([1], {(0, 0): Verdict.EMPTY})
        r = TaskSupervisor(b, TaskLimits(max_retries=0)).run()
        self.assertEqual((r.retries, r.stop), (0, TaskStop.PLAN_EXECUTED))
        self.assertEqual(r.checkpoints[0]["action"], "continue")
        self.assertEqual(TaskLimits(max_retries=0).mode, "task_checkpoint_only")

    def test_one_retry_per_object_then_the_plan_continues(self):
        b = FakeBackend([1, 1, 1], {(0, 0): Verdict.EMPTY, (1, 0): Verdict.SLIPPED},
                        labels={0: ["cube"], 1: ["cube"], 2: ["cube"]})
        r = TaskSupervisor(b, TaskLimits(max_retries=5)).run()
        self.assertEqual([c["action"] for c in r.checkpoints], ["retry", "continue"])
        self.assertEqual(r.retries, 1)

    def test_lost_object_stops_without_retry(self):
        b = FakeBackend([2], {(0, 0): Verdict.LOST})
        r = TaskSupervisor(b, TaskLimits()).run()
        self.assertEqual((r.stop, r.retries), (TaskStop.OBJECT_LOST, 0))

    def test_failed_replan_and_failed_recovery_stop(self):
        r = TaskSupervisor(FakeBackend([1], {(0, 0): Verdict.EMPTY}, plan_fail_at=1), TaskLimits()).run()
        self.assertEqual(r.stop, TaskStop.RETRY_NO_PLAN)
        r = TaskSupervisor(FakeBackend([1], {(0, 0): Verdict.EMPTY}, recover_ok=False), TaskLimits()).run()
        self.assertEqual(r.stop, TaskStop.RECOVERY_FAILED)

    def test_sim_time_limit(self):
        b = FakeBackend([5], {})
        r = TaskSupervisor(b, TaskLimits(max_sim_s=12)).run()
        self.assertEqual(r.stop, TaskStop.SIM_TIME_LIMIT)

    def test_plan_without_grasp_units_just_runs(self):
        b = FakeBackend([0], {})
        r = TaskSupervisor(b, TaskLimits()).run()
        self.assertEqual((r.stop, r.checkpoints), (TaskStop.PLAN_EXECUTED, []))


if __name__ == "__main__":
    unittest.main()
