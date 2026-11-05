"""Week two: grasp-checkpoint recovery logic (pure Python + numpy, no simulator).

The supervisor runs a bounded loop

    PLAN -> EXECUTE_GRASP -> CHECK_OUTCOME
        holding the target            -> DONE (success)
        confirmed empty, budget left  -> PREPARE_RETRY -> OBSERVE -> PLAN
        anything else                 -> DONE (explicit stop reason)

against a `RecoveryBackend` (the simulator glue in recovery_eval.py, or a fake in the
tests). The backend also owns the outcome detector. Two detector interfaces exist:

* `OracleStateDetector` reads privileged simulator state (object poses, finger links).
  It is the first, explicitly labelled baseline ("oracle-checkpoint recovery").
* `SensorDetector` takes only what a real robot would have (RGB-D, calibration, joint and
  gripper feedback, the target's identity). No implementation exists yet.

Neither the week-one observer nor its scorer is changed by this module. Note the difference
between the two questions asked of a trace:

* `scoring.score_first_grasp(...).lifted`: the target was held for long enough at some
  earlier point (historical, kept as the week-one metric);
* `OracleStateDetector.check(...)`: the target is held *now*, over a window that ends at
  the checkpoint. A lift followed by a drop is not "held now".

All thresholds below are first guesses. Validate them on GPU traces and freeze them
before any comparison (docs/week2_recovery.md).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np


# --------------------------------------------------------------------------- outcomes


class Outcome(str, Enum):
    HOLDING_TARGET = "holding_target"
    FAILED_EMPTY = "failed_empty"
    UNCERTAIN = "uncertain"


class State(str, Enum):
    PLAN = "plan"
    EXECUTE_GRASP = "execute_grasp"
    CHECK_OUTCOME = "check_outcome"
    PREPARE_RETRY = "prepare_retry"
    OBSERVE = "observe"
    DONE = "done"


class Stop(str, Enum):
    """Why a trial ended. Only SUCCESS is a success."""

    SUCCESS = "success"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"  # confirmed empty, no grasp attempt left
    PLANNING_BUDGET_EXHAUSTED = "planning_budget_exhausted"
    UNCERTAIN = "uncertain_outcome"
    WRONG_OBJECT = "wrong_object_held"
    NO_PLAN = "planning_failure"  # server answered without a plan
    UNSUPPORTED_PLAN = "unsupported_plan"  # plan structure the segment parser refuses
    INVALID_MOTION = "invalid_motion"  # execution did not reach the checkpoint as planned
    RETURN_FAILED = "return_failed"  # no validated empty-hand return to an observation pose
    SIM_TIME_LIMIT = "sim_time_limit"
    WALL_TIME_LIMIT = "wall_time_limit"
    EPISODE_TIME_LIMIT = "episode_time_limit"  # the simulator's own truncation
    INFRASTRUCTURE = "infrastructure"  # connection/API/harness errors
    INTERRUPTED = "interrupted"


@dataclass
class CheckResult:
    outcome: Outcome
    reason: str
    t_sim: float | None = None  # simulated time of the checkpoint decision
    source: str = ""  # detector name, e.g. "oracle-state"
    privileged: bool = True  # True when the decision used simulator truth
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


@dataclass
class StepOutcome:
    """Result of one backend phase. ok=False carries the stop reason."""

    ok: bool
    stop: Stop | None = None
    detail: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, **data) -> "StepOutcome":
        return cls(True, None, None, data)

    @classmethod
    def fail(cls, stop: Stop, detail: str | None = None, **data) -> "StepOutcome":
        return cls(False, stop, detail, data)


# --------------------------------------------------------------------------- limits


@dataclass(frozen=True)
class CheckpointLimits:
    """Per-trial budgets. Use identical values for the one-attempt control and the retry mode,
    except max_attempts / max_planning_calls. Defaults are proposals to review on the machine."""

    max_attempts: int = 2  # target closes (open->close transitions), not control ticks
    max_planning_calls: int = 2
    max_sim_s: float = 180.0  # simulated time from the first planning request
    max_wall_s: float = 600.0  # wall time from the first planning request
    confirm_hold_s: float = 1.0  # closed-gripper hold at the lift checkpoint, physics stepping

    def __post_init__(self):
        if self.max_attempts < 1 or self.max_planning_calls < 1:
            raise ValueError("max_attempts and max_planning_calls must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def mode(self) -> str:
        return "checkpoint_one_attempt" if self.max_attempts == 1 else f"checkpoint_recovery_{self.max_attempts}"


@dataclass
class Budget:
    limits: CheckpointLimits
    planning_calls: int = 0
    closes: int = 0

    def time_stop(self, sim_s: float, wall_s: float) -> Stop | None:
        if sim_s >= self.limits.max_sim_s:
            return Stop.SIM_TIME_LIMIT
        if wall_s >= self.limits.max_wall_s:
            return Stop.WALL_TIME_LIMIT
        return None

    def retry_stop(self) -> Stop | None:
        """Why no further grasp attempt is allowed, or None if one is."""
        if self.closes >= self.limits.max_attempts:
            return Stop.ATTEMPTS_EXHAUSTED
        if self.planning_calls >= self.limits.max_planning_calls:
            return Stop.PLANNING_BUDGET_EXHAUSTED
        return None


# --------------------------------------------------------------------------- backend


class RecoveryBackend(Protocol):
    """What the supervisor needs from the world. Every phase advances the same episode:
    the backend must never reset object poses, robot state or simulation time between attempts.

    `execute_grasp` is the seam where a learned approach-close-lift skill can later replace
    conventional plan execution under the same supervisor and budgets.
    """

    def observe(self, attempt: int) -> StepOutcome: ...  # capture current RGB-D + joints for planning

    def plan(self, attempt: int) -> StepOutcome: ...  # fresh TiPToP request from the latest observation

    def execute_grasp(self, attempt: int) -> StepOutcome: ...  # approach, close, retraction, confirmation hold

    def check(self, attempt: int) -> CheckResult: ...

    def prepare_retry(self, attempt: int) -> StepOutcome: ...  # empty-hand retreat to an observation pose

    def sim_time(self) -> float: ...  # simulated seconds since the supervisor started

    def closes(self) -> int: ...  # open->close transitions commanded so far in this trial


@dataclass
class AttemptLog:
    index: int
    observe: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    execute: dict[str, Any] | None = None
    check: dict[str, Any] | None = None
    retry: dict[str, Any] | None = None
    stop: str | None = None  # set on the attempt where the trial stopped


@dataclass
class TrialResult:
    stop: Stop
    detail: str | None
    success: bool
    first_attempt_success: bool
    attempts: list[AttemptLog]
    planning_calls: int
    closes: int
    sim_s: float
    wall_s: float
    transitions: list[dict[str, Any]]
    retry_triggered: bool
    harness_exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stop"] = self.stop.value
        return d


class RecoverySupervisor:
    """Bounded checkpoint-recovery loop. Pure orchestration: all side effects are in the backend."""

    def __init__(self, backend: RecoveryBackend, limits: CheckpointLimits,
                 wall_clock: Callable[[], float] = time.monotonic):
        self.backend = backend
        self.limits = limits
        self.budget = Budget(limits)
        self._clock = wall_clock
        self._t0 = None
        self.transitions: list[dict[str, Any]] = []

    def _wall(self) -> float:
        return self._clock() - self._t0

    def _enter(self, state: State, attempt: int) -> None:
        self.transitions.append({"state": state.value, "attempt": attempt,
                                 "t_sim": round(self.backend.sim_time(), 4), "t_wall": round(self._wall(), 3)})

    def _time_stop(self) -> Stop | None:
        return self.budget.time_stop(self.backend.sim_time(), self._wall())

    def run(self) -> TrialResult:
        self._t0 = self._clock()
        attempts: list[AttemptLog] = []
        stop, detail, exc = None, None, None
        first_success = False
        retry_triggered = False
        attempt = 0
        log = AttemptLog(0)
        attempts.append(log)
        state = State.OBSERVE
        try:
            while state != State.DONE:
                self._enter(state, attempt)
                t = self._time_stop()
                if t is not None and state != State.CHECK_OUTCOME:
                    stop, detail = t, f"limit reached before {state.value}"
                    break

                if state == State.OBSERVE:
                    r = self.backend.observe(attempt)
                    log.observe = {"ok": r.ok, "detail": r.detail, **r.data}
                    if not r.ok:
                        stop, detail = r.stop or Stop.INFRASTRUCTURE, r.detail
                        break
                    state = State.PLAN

                elif state == State.PLAN:
                    if self.budget.planning_calls >= self.limits.max_planning_calls:
                        stop, detail = Stop.PLANNING_BUDGET_EXHAUSTED, None
                        break
                    self.budget.planning_calls += 1  # a request counts even when it fails
                    r = self.backend.plan(attempt)
                    log.plan = {"ok": r.ok, "detail": r.detail, **r.data}
                    if not r.ok:
                        stop, detail = r.stop or Stop.NO_PLAN, r.detail
                        break
                    state = State.EXECUTE_GRASP

                elif state == State.EXECUTE_GRASP:
                    r = self.backend.execute_grasp(attempt)
                    self.budget.closes = self.backend.closes()
                    log.execute = {"ok": r.ok, "detail": r.detail, **r.data}
                    if not r.ok:
                        stop, detail = r.stop or Stop.INVALID_MOTION, r.detail
                        break
                    state = State.CHECK_OUTCOME

                elif state == State.CHECK_OUTCOME:
                    c = self.backend.check(attempt)
                    log.check = c.to_dict()
                    if c.outcome == Outcome.HOLDING_TARGET:
                        first_success = attempt == 0
                        stop, detail = Stop.SUCCESS, c.reason
                        break
                    if c.outcome == Outcome.UNCERTAIN:
                        stop = Stop.WRONG_OBJECT if c.reason == WRONG_OBJECT_REASON else Stop.UNCERTAIN
                        detail = c.reason
                        break
                    # Confirmed empty hand.
                    t = self._time_stop()
                    if t is not None:
                        stop, detail = t, "confirmed empty, but the time budget is used up"
                        break
                    b = self.budget.retry_stop()
                    if b is not None:
                        stop, detail = b, "confirmed empty"
                        break
                    state = State.PREPARE_RETRY

                elif state == State.PREPARE_RETRY:
                    retry_triggered = True
                    r = self.backend.prepare_retry(attempt)
                    self.budget.closes = self.backend.closes()
                    log.retry = {"ok": r.ok, "detail": r.detail, **r.data}
                    if not r.ok:
                        stop, detail = r.stop or Stop.RETURN_FAILED, r.detail
                        break
                    attempt += 1
                    log = AttemptLog(attempt)
                    attempts.append(log)
                    state = State.OBSERVE
        except KeyboardInterrupt:
            stop, detail = Stop.INTERRUPTED, "KeyboardInterrupt"
        except Exception:
            exc = traceback.format_exc()
            stop, detail = Stop.INFRASTRUCTURE, "harness exception"

        self.budget.closes = self._safe_closes()
        log.stop = stop.value
        self._enter(State.DONE, attempt)
        return TrialResult(
            stop=stop, detail=detail, success=stop == Stop.SUCCESS, first_attempt_success=first_success,
            attempts=attempts, planning_calls=self.budget.planning_calls, closes=self.budget.closes,
            sim_s=self.backend.sim_time(), wall_s=self._wall(), transitions=self.transitions,
            retry_triggered=retry_triggered, harness_exception=exc,
        )

    def _safe_closes(self) -> int:
        try:
            return self.backend.closes()
        except Exception:
            return self.budget.closes


# --------------------------------------------------------------------------- detectors

WRONG_OBJECT_REASON = "wrong_object_held"


class OutcomeDetector(Protocol):
    name: str
    privileged: bool

    def check(self, packet: Any) -> CheckResult: ...


@dataclass(frozen=True)
class OracleCheckCriteria:
    """Thresholds for the simulator-state checkpoint. Pilot values: validate against video.

    The pinned cuTAMP retraction moves the hand only ~5 cm along the grasp axis, so the
    week-one absolute rule (target >= 5 cm above its start) is borderline at this checkpoint.
    "Held" is therefore judged by the target following the hand during the retraction.
    """

    max_obj_to_gripper_m: float = 0.10  # same meaning as LiftCriteria.max_obj_to_gripper_m
    close_threshold: float = 0.5
    min_gripper_rise_m: float = 0.03  # the retraction must raise the hand at least this much
    held_min_follow_frac: float = 0.6  # target rise >= frac * hand rise (close -> checkpoint)
    held_max_rel_drift_m: float = 0.01  # target-in-hand offset spread over the hold window
    empty_max_target_rise_m: float = 0.01  # |target dz| since the close started
    empty_max_target_motion_m: float = 0.005  # target displacement over the hold window (settled)
    other_near_m: float = 0.08  # another object this close to the hand makes "empty" ambiguous
    other_moved_m: float = 0.01  # ... if it also moved by more than this since the close
    # Optional finger evidence (privileged finger joint, 0 open .. 1 closed). None = unused
    # until validated: fingers stopping short of closed suggest something between them.
    empty_min_finger_closure: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OracleStatePacket:
    """Simulator-state window for one attempt, from the start of the close to the checkpoint.

    Arrays are sampled after each env.step. `hold_start` indexes the first sample of the
    confirmation hold; `retract_start` the first sample of the post-close retraction.
    """

    t: np.ndarray  # (N,)
    target_pos: np.ndarray  # (N, 3)
    gripper_pos: np.ndarray  # (N, 3) gripper reference point (finger midpoint)
    gripper_cmd: np.ndarray  # (N,)
    retract_start: int
    hold_start: int
    other_pos: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N, 3)
    finger_closure: np.ndarray | None = None  # (N,) 0 open .. 1 closed, privileged


class OracleStateDetector:
    """Ground-truth-state checkpoint ("oracle-checkpoint"). Changes control using privileged
    simulator state; never describe its results as observation-only or image-based."""

    name = "oracle-state"
    privileged = True

    def __init__(self, criteria: OracleCheckCriteria = OracleCheckCriteria()):
        self.criteria = criteria

    def check(self, p: OracleStatePacket) -> CheckResult:
        c = self.criteria
        t = np.asarray(p.t, dtype=float)
        obj = np.asarray(p.target_pos, dtype=float).reshape(-1, 3)
        grip = np.asarray(p.gripper_pos, dtype=float).reshape(-1, 3)
        cmd = np.asarray(p.gripper_cmd, dtype=float)
        n = len(t)
        ev: dict[str, Any] = {"n_samples": n, "criteria": c.to_dict()}

        def result(outcome: Outcome, reason: str) -> CheckResult:
            return CheckResult(outcome, reason, float(t[-1]) if n else None, self.name, True, ev)

        if n == 0 or not (0 <= p.retract_start <= p.hold_start < n):
            ev["retract_start"], ev["hold_start"] = p.retract_start, p.hold_start
            return result(Outcome.UNCERTAIN, "no_checkpoint_window")
        if not (len(obj) == len(grip) == len(cmd) == n):
            raise ValueError("packet arrays must have the same length")

        hold = slice(p.hold_start, n)
        ev["hold_s"] = round(float(t[-1] - t[p.hold_start]), 4)
        if not np.all(cmd[hold] > c.close_threshold):
            return result(Outcome.UNCERTAIN, "gripper_not_commanded_closed_during_hold")

        grip_rise = float(grip[-1, 2] - grip[0, 2])
        obj_rise = float(obj[-1, 2] - obj[0, 2])
        dist = np.linalg.norm(obj - grip, axis=1)
        rel = obj[hold] - grip[hold]
        rel_drift = float(np.max(np.linalg.norm(rel - rel.mean(axis=0), axis=1)))
        obj_motion_hold = float(np.max(np.linalg.norm(obj[hold] - obj[-1], axis=1)))
        ev.update({
            "gripper_rise_m": round(grip_rise, 4),
            "target_rise_m": round(obj_rise, 4),
            "target_to_gripper_m_end": round(float(dist[-1]), 4),
            "target_to_gripper_m_hold_max": round(float(np.max(dist[hold])), 4),
            "target_in_hand_drift_m": round(rel_drift, 4),
            "target_motion_in_hold_m": round(obj_motion_hold, 4),
        })

        # Other objects: wrong-object grasps and ambiguous neighbours.
        others = {}
        for name, pos in (p.other_pos or {}).items():
            pos = np.asarray(pos, dtype=float).reshape(-1, 3)
            d_end = float(np.linalg.norm(pos[-1] - grip[-1]))
            rise = float(pos[-1, 2] - pos[0, 2])
            moved = float(np.linalg.norm(pos[-1] - pos[0]))
            followed = (grip_rise > 0 and rise >= c.held_min_follow_frac * grip_rise
                        and bool(np.all(np.linalg.norm(pos[hold] - grip[hold], axis=1) <= c.max_obj_to_gripper_m)))
            others[name] = {"to_gripper_m_end": round(d_end, 4), "rise_m": round(rise, 4),
                            "moved_m": round(moved, 4), "followed_hand": followed}
        ev["other_objects"] = others

        if grip_rise < c.min_gripper_rise_m:
            return result(Outcome.UNCERTAIN, "retraction_did_not_raise_gripper")

        if p.finger_closure is not None:
            fc = np.asarray(p.finger_closure, dtype=float)
            ev["finger_closure_end"] = round(float(fc[-1]), 4)

        held_now = (obj_rise >= c.held_min_follow_frac * grip_rise
                    and bool(np.all(dist[hold] <= c.max_obj_to_gripper_m))
                    and rel_drift <= c.held_max_rel_drift_m)
        if held_now:
            if any(o["followed_hand"] for o in others.values()):
                return result(Outcome.UNCERTAIN, "target_and_another_object_both_follow_hand")
            return result(Outcome.HOLDING_TARGET, "target_followed_retraction_and_stable_in_hold")

        if any(o["followed_hand"] for o in others.values()):
            return result(Outcome.UNCERTAIN, WRONG_OBJECT_REASON)

        if obj_rise >= c.held_min_follow_frac * grip_rise:
            # Rose with the hand but not stable/near during the hold: slipping or swinging.
            return result(Outcome.UNCERTAIN, "target_rose_but_not_stably_held")

        empty = (abs(obj_rise) <= c.empty_max_target_rise_m and obj_motion_hold <= c.empty_max_target_motion_m)
        if not empty:
            if obj_motion_hold > c.empty_max_target_motion_m:
                return result(Outcome.UNCERTAIN, "target_still_moving_at_checkpoint")
            return result(Outcome.UNCERTAIN, "target_displaced_vertically_but_not_held")

        ambiguous = [n for n, o in others.items()
                     if o["to_gripper_m_end"] <= c.other_near_m and o["moved_m"] > c.other_moved_m]
        if ambiguous:
            ev["ambiguous_neighbours"] = ambiguous
            return result(Outcome.UNCERTAIN, "disturbed_object_near_hand")

        if c.empty_min_finger_closure is not None:
            if p.finger_closure is None:
                return result(Outcome.UNCERTAIN, "finger_evidence_required_but_missing")
            if float(np.asarray(p.finger_closure)[-1]) < c.empty_min_finger_closure:
                return result(Outcome.UNCERTAIN, "fingers_blocked_but_target_not_held")

        return result(Outcome.FAILED_EMPTY, "target_stayed_down_while_hand_rose")


@dataclass
class SensorPacket:
    """What a real robot has at the checkpoint. Ground-truth object poses never go in here;
    they stay in the evaluator and the oracle adapter."""

    t: float
    rgb: np.ndarray | None
    depth: np.ndarray | None
    intrinsics: np.ndarray | None
    world_from_cam: np.ndarray | None
    joint_pos: np.ndarray
    gripper_obs: float  # measured gripper position, 0 open .. 1 closed (noisy in the sim)
    target_label: str  # original target identity as TiPToP perceived it
    instruction: str


class SensorDetector:
    """Observation-based checkpoint detector interface. Not implemented in week two.

    A candidate narrow-domain implementation: run TiPToP's perception only (no TAMP call on
    a held-object image) to segment the target, lift its pixels to 3D with depth, compare the
    point set with the gripper frame from FK, combine with gripper feedback, and return
    UNCERTAIN (or request another view) under occlusion. Gripper width alone is not evidence
    of success. Validate against the oracle (false-success and false-empty rates) before
    calling it reliable.
    """

    name = "sensor"
    privileged = False

    def check(self, packet: SensorPacket) -> CheckResult:  # pragma: no cover - interface only
        raise NotImplementedError("no observation-based checkpoint detector exists yet; use oracle-state")
