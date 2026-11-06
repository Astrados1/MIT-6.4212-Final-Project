"""Task-level checkpoint recovery: decision logic (pure Python + numpy, no simulator).

After every grasp of a TiPToP plan the arm lifts the object slowly and a monitor watches the object
and the hand on every simulation step (docs/task_recovery.md):

    close -> LIFT (monitored) -> HOLD (confirm) -> verdict
       held      -> go back down to the plan and continue it unchanged
       empty     -> open, go to the observation pose, re-observe, new TiPToP plan (a retry)
       slipped   -> stop, lower the object until it rests ~1 cm above its start, open, retry
       uncertain -> continue the plan as the baseline would (never retry on doubt)
       lost      -> stop the trial (the object left the table)

Two lift strategies share this monitor (`lift_mode`):
  * `reverse_approach`: retrace the plan's own approach back toward its start pose (the arm's
    position before the grasp). No new motion code; the path just travelled is known to be clear.
    The lift stops as soon as the verdict is clear, well before the start pose.
  * `vertical_ik`: after the plan's 5 cm retraction, lift the hand straight up (orientation held)
    until the object is `check_object_rise_m` up or the hand reaches `hand_rise_cap_m`.

The monitor reads simulator truth (object geometry centers): results are oracle-checkpoint
results, never observation-only. All thresholds are first values to calibrate on the baseline
traces (grasp_eval/checkpoint_calibration.py) before a paired run.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np

LIFT_MODES = ("reverse_approach", "vertical_ik")


class Verdict(str, Enum):
    HELD = "held"
    EMPTY = "empty"
    SLIPPED = "slipped"
    UNCERTAIN = "uncertain"
    LOST = "object_lost"


@dataclass(frozen=True)
class TaskCheckCriteria:
    """Thresholds for the monitored lift (meters, seconds). First values; calibrate before use."""

    # empty: the hand rose this far while the object stayed down and still
    miss_min_hand_rise_m: float = 0.05
    miss_max_object_rise_m: float = 0.01
    still_window_s: float = 0.3
    still_max_motion_m: float = 0.005
    # following: the object may lag the hand by a few cm while the grip settles (observed ~2 cm)
    follow_confirm_rise_m: float = 0.02  # object rise at which the in-hand offset reference is taken
    # held: object this far up, following, then stable through a hold
    held_min_object_rise_m: float = 0.04
    max_offset_change_m: float = 0.01  # in-hand offset change allowed after following is confirmed
    hold_s: float = 1.0
    hold_max_sink_m: float = 0.003  # the object may not sink more than this while the hand holds still
    # slip: after following, the object falls back this far from its highest point
    slip_drop_m: float = 0.008
    # vertical_ik: target object rise and hand-rise cap; reverse_approach: cap on hand rise too
    check_object_rise_m: float = 0.05
    hand_rise_cap_m: float = 0.10
    # set-down after a slip: lower until the object is at most this far above its start
    set_down_rise_m: float = 0.01
    # lost: the object fell below the table or slid far from where it was grasped
    lost_below_start_m: float = 0.05
    lost_xy_m: float = 0.30
    # wrong object: another object follows the hand this closely
    other_follow_rise_m: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiftMonitor:
    """Watches one monitored lift, one sample per simulation step.

    `phase` tells the backend what to command next: "lift" (keep lifting), "hold" (stop and hold
    the current command), "done" (`verdict` is set). `at_lift_end()` reports that the lift path or
    hand-rise cap was exhausted without a verdict, which makes it uncertain.
    """

    def __init__(self, criteria: TaskCheckCriteria, dt: float, hand_at_close, object_at_rest,
                 others_at_rest: dict[str, Any] | None = None):
        self.c = criteria
        self.dt = float(dt)
        self.hand0 = np.asarray(hand_at_close, dtype=float)
        self.obj0 = np.asarray(object_at_rest, dtype=float)
        self.others0 = {k: np.asarray(v, dtype=float) for k, v in (others_at_rest or {}).items()}
        self.phase = "lift"
        self.verdict: Verdict | None = None
        self.reason: str | None = None
        self.n = 0
        self.max_obj_rise = 0.0
        self.offset_ref: np.ndarray | None = None
        self.hold_left = 0
        self.hold_obj_z: float | None = None
        self.objs: list[np.ndarray] = []
        self.profile: list[list[float]] = []  # [t, hand_rise, obj_rise] per sample (compact evidence)

    # ---------------------------------------------------------------- helpers
    def _rises(self, hand, obj) -> tuple[float, float]:
        return float(hand[2] - self.hand0[2]), float(obj[2] - self.obj0[2])

    def _still(self) -> bool:
        k = max(2, int(round(self.c.still_window_s / self.dt)))
        if len(self.objs) < k:
            return False
        w = np.asarray(self.objs[-k:])
        return float(np.max(np.linalg.norm(w - w[-1], axis=1))) <= self.c.still_max_motion_m

    def _decide(self, verdict: Verdict, reason: str) -> Verdict:
        self.phase, self.verdict, self.reason = "done", verdict, reason
        return verdict

    def evidence(self) -> dict[str, Any]:
        last = self.profile[-1] if self.profile else [None, None, None]
        return {"samples": self.n, "hand_rise_m": last[1], "object_rise_m": last[2],
                "max_object_rise_m": round(self.max_obj_rise, 4),
                "offset_ref": None if self.offset_ref is None else np.round(self.offset_ref, 4).tolist(),
                "profile_every_5": self.profile[::5][-60:]}

    # ---------------------------------------------------------------- main update
    def update(self, hand, obj, others: dict[str, Any] | None = None) -> Verdict | None:
        """Add one sample (geometry centers, world frame). Returns the verdict once decided."""
        if self.phase == "done":
            return self.verdict
        c = self.c
        hand = np.asarray(hand, dtype=float)
        obj = np.asarray(obj, dtype=float)
        self.n += 1
        self.objs.append(obj)
        hand_rise, obj_rise = self._rises(hand, obj)
        self.max_obj_rise = max(self.max_obj_rise, obj_rise)
        self.profile.append([round(self.n * self.dt, 3), round(hand_rise, 4), round(obj_rise, 4)])

        if obj_rise < -c.lost_below_start_m or float(np.linalg.norm(obj[:2] - self.obj0[:2])) > c.lost_xy_m:
            return self._decide(Verdict.LOST, "object fell below the table or slid away")
        for name, p in (others or {}).items():
            p0 = self.others0.get(name)
            if p0 is not None and float(np.asarray(p, dtype=float)[2] - p0[2]) >= c.other_follow_rise_m \
                    and obj_rise < c.miss_max_object_rise_m:
                return self._decide(Verdict.UNCERTAIN, f"another object ({name}) rose with the hand")

        if self.offset_ref is None and obj_rise >= c.follow_confirm_rise_m:
            self.offset_ref = obj - hand
        following = self.offset_ref is not None
        if following:
            drift = float(np.linalg.norm((obj - hand) - self.offset_ref))
            if self.max_obj_rise - obj_rise > c.slip_drop_m or drift > c.max_offset_change_m:
                return self._decide(Verdict.SLIPPED, f"object fell back {100 * (self.max_obj_rise - obj_rise):.1f} cm "
                                                     f"/ moved {100 * drift:.1f} cm in the hand")

        if self.phase == "lift":
            if (hand_rise >= c.miss_min_hand_rise_m and obj_rise <= c.miss_max_object_rise_m
                    and not following and self._still()):
                return self._decide(Verdict.EMPTY, "hand rose, object stayed down and still")
            if following and obj_rise >= c.held_min_object_rise_m:
                self.phase = "hold"
                self.hold_left = max(1, int(round(c.hold_s / self.dt)))
                self.hold_obj_z = float(obj[2])
            return None

        if self.hold_obj_z is not None and self.hold_obj_z - float(obj[2]) > c.hold_max_sink_m:
            return self._decide(Verdict.SLIPPED, f"object sank {1000 * (self.hold_obj_z - float(obj[2])):.1f} mm "
                                                 "during the hold")

        # hold: confirm the object stays up and steady while the hand is still
        self.hold_left -= 1
        if self.hold_left <= 0:
            if following and obj_rise >= c.held_min_object_rise_m:
                return self._decide(Verdict.HELD, "object up and steady in the hand through the hold")
            return self._decide(Verdict.UNCERTAIN, "object up but not steady through the hold")
        return None

    def at_lift_end(self, hand=None, obj=None) -> Verdict:
        """The lift path or hand-rise cap is exhausted without a verdict: decide from what is known."""
        if self.phase == "done":
            return self.verdict
        hand_rise = self.profile[-1][1] if self.profile else 0.0
        if hand_rise < self.c.miss_min_hand_rise_m:
            # the hand itself did not rise (blocked on the object, or a stall): no evidence either way
            return self._decide(Verdict.UNCERTAIN, f"hand rose only {100 * hand_rise:.1f} cm (blocked or stalled)")
        if self.offset_ref is None and self.max_obj_rise <= self.c.miss_max_object_rise_m:
            return self._decide(Verdict.EMPTY, "lift finished, object never left the table")
        return self._decide(Verdict.UNCERTAIN, f"lift finished with the object {100 * self.max_obj_rise:.1f} cm up "
                                               "(neither clearly held nor clearly empty)")


# --------------------------------------------------------------------------- trial supervisor


class TaskStop(str, Enum):
    PLAN_EXECUTED = "plan_executed"  # ran to the end of the (last) plan; task success decided by the goal check
    NO_PLAN = "planning_failure"  # first plan missing (a paired trial without a baseline plan never gets here)
    RETRY_NO_PLAN = "retry_planning_failure"
    OBJECT_LOST = "object_lost"
    RECOVERY_FAILED = "recovery_failed"  # set-down / return to the observation pose failed
    INVALID_MOTION = "invalid_motion"
    SIM_TIME_LIMIT = "sim_time_limit"
    WALL_TIME_LIMIT = "wall_time_limit"
    EPISODE_TIME_LIMIT = "episode_time_limit"
    INFRASTRUCTURE = "infrastructure"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class TaskLimits:
    """Budgets for one task-level trial. max_retries=0 is the checkpoint-only control (checks and
    lifts, never retries), which separates the effect of the check motion from that of retrying."""

    max_retries: int = 2  # new plans after a confirmed empty grasp or slip, per trial
    max_retries_per_grasp: int = 1  # retries spent on one object before it is left to the plan
    max_planning_calls: int = 3  # includes the first plan
    max_sim_s: float = 400.0
    max_wall_s: float = 1800.0
    baseline_episode_s: float = 90.0  # the baseline's episode length: task state is snapshotted there too

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode
        return d

    @property
    def mode(self) -> str:
        return "task_checkpoint_only" if self.max_retries == 0 else f"task_recovery_r{self.max_retries}"


@dataclass
class Step:
    """Result of one backend phase."""

    ok: bool
    stop: TaskStop | None = None
    detail: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, **data) -> "Step":
        return cls(True, None, None, data)

    @classmethod
    def fail(cls, stop: TaskStop, detail: str | None = None, **data) -> "Step":
        return cls(False, stop, detail, data)


class TaskBackend(Protocol):
    """World glue for the supervisor. The episode is never reset between attempts."""

    def plan(self, attempt: int) -> Step: ...  # observe + plan (attempt 0 may replay a saved plan); data["units"]
    def run_to_checkpoint(self, attempt: int, unit: int) -> Step: ...  # execute up to the lift start of `unit`
    def lift_and_check(self, attempt: int, unit: int) -> dict[str, Any]: ...  # {"verdict": Verdict, "reason", ...}
    def resume(self, attempt: int, unit: int) -> Step: ...  # back onto the plan after held/uncertain/no-retry
    def recover(self, attempt: int, unit: int, verdict: Verdict) -> Step: ...  # open/set down + observation pose
    def run_to_end(self, attempt: int) -> Step: ...  # rest of the current plan, as upstream
    def sim_time(self) -> float: ...


@dataclass
class TaskResult:
    stop: TaskStop
    detail: str | None
    checkpoints: list[dict[str, Any]]
    attempts: int  # plans executed (1 + retries that produced a plan)
    retries: int
    planning_calls: int
    sim_s: float
    wall_s: float
    events: list[dict[str, Any]]
    harness_exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stop"] = self.stop.value
        return d


class TaskSupervisor:
    """Runs a whole task: checkpoint after every grasp, retry on a confirmed miss or slip."""

    def __init__(self, backend: TaskBackend, limits: TaskLimits, wall_clock: Callable[[], float] = time.monotonic):
        self.b, self.limits, self._clock = backend, limits, wall_clock
        self.events: list[dict[str, Any]] = []

    def _event(self, what: str, **kw) -> None:
        self.events.append({"what": what, "t_sim": round(self.b.sim_time(), 3), **kw})

    def _time_stop(self, t0: float) -> TaskStop | None:
        if self.b.sim_time() >= self.limits.max_sim_s:
            return TaskStop.SIM_TIME_LIMIT
        if self._clock() - t0 >= self.limits.max_wall_s:
            return TaskStop.WALL_TIME_LIMIT
        return None

    def run(self) -> TaskResult:
        t0 = self._clock()
        checkpoints: list[dict[str, Any]] = []
        attempt, retries, planning_calls, attempts = 0, 0, 0, 0
        retries_by_object: dict[str, int] = {}
        stop, detail, exc = None, None, None
        try:
            while stop is None:
                planning_calls += 1
                r = self.b.plan(attempt)
                self._event("plan", attempt=attempt, ok=r.ok, detail=r.detail)
                if not r.ok:
                    stop = r.stop or (TaskStop.NO_PLAN if attempt == 0 else TaskStop.RETRY_NO_PLAN)
                    detail = r.detail
                    break
                attempts += 1
                units = r.data.get("units") or []
                labels = r.data.get("unit_labels") or [None] * len(units)
                restarted = False
                for u in range(len(units)):
                    t = self._time_stop(t0)
                    if t is not None:
                        stop, detail = t, f"before grasp {u} of attempt {attempt}"
                        break
                    r = self.b.run_to_checkpoint(attempt, u)
                    if not r.ok:
                        stop, detail = r.stop or TaskStop.INVALID_MOTION, r.detail
                        break
                    chk = self.b.lift_and_check(attempt, u)
                    v = chk["verdict"]
                    label = labels[u] if u < len(labels) else None
                    cp = {"attempt": attempt, "unit": u, "target_label": label, **chk,
                          "verdict": v.value, "action": None}
                    checkpoints.append(cp)
                    if v == Verdict.LOST:
                        cp["action"] = "stop"
                        stop, detail = TaskStop.OBJECT_LOST, chk.get("reason")
                        break
                    key = str(label)
                    can_retry = (v in (Verdict.EMPTY, Verdict.SLIPPED) and retries < self.limits.max_retries
                                 and retries_by_object.get(key, 0) < self.limits.max_retries_per_grasp
                                 and planning_calls < self.limits.max_planning_calls
                                 and self._time_stop(t0) is None)
                    if can_retry:
                        cp["action"] = "retry"
                        r = self.b.recover(attempt, u, v)
                        self._event("recover", attempt=attempt, unit=u, verdict=v.value, ok=r.ok, detail=r.detail)
                        if not r.ok:
                            stop, detail = r.stop or TaskStop.RECOVERY_FAILED, r.detail
                            break
                        retries += 1
                        retries_by_object[key] = retries_by_object.get(key, 0) + 1
                        attempt += 1
                        restarted = True
                        break
                    # held, uncertain, or a miss without budget: carry on exactly as the plan says
                    cp["action"] = "continue"
                    r = self.b.resume(attempt, u)
                    if not r.ok:
                        stop, detail = r.stop or TaskStop.INVALID_MOTION, r.detail
                        break
                if stop is not None or restarted:
                    continue
                r = self.b.run_to_end(attempt)
                stop = TaskStop.PLAN_EXECUTED if r.ok else (r.stop or TaskStop.INVALID_MOTION)
                detail = r.detail
        except KeyboardInterrupt:
            stop, detail = TaskStop.INTERRUPTED, "KeyboardInterrupt"
        except Exception:
            exc = traceback.format_exc()
            stop, detail = TaskStop.INFRASTRUCTURE, "harness exception"
        self._event("done", stop=stop.value)
        return TaskResult(stop=stop, detail=detail, checkpoints=checkpoints, attempts=attempts, retries=retries,
                          planning_calls=planning_calls, sim_s=self.b.sim_time(), wall_s=self._clock() - t0,
                          events=self.events, harness_exception=exc)
