"""One week-two checkpoint-recovery trial in droid-sim-evals (oracle-state detector).

Run from the droid-sim-evals checkout, inside its uv environment (same as observed_eval.py):

    cd $TIPTOP_DIR/droid-sim-evals
    uv run python $PROJECT/grasp_eval/recovery_eval.py --tiptop-dir $TIPTOP_DIR/tiptop \
        --scene 1 --variant 0 --trial 0 --target-object <name from observed_eval --inspect-only> \
        --instruction "Put the Rubik's cube in the bowl." \
        --detector oracle-state --max-attempts 2 --out-dir $PROJECT/results/week2_checkpoint_oracle

--max-attempts 1 is the matched one-attempt checkpoint control. Neither mode is "unchanged
upstream TiPToP": both stop after the post-grasp retraction and hold for --confirm-hold-s.
The week-one observer (observed_eval.py) remains the unmodified-baseline condition.

What one trial does, in one continuous episode (no resets between attempts):
  1. settle as upstream; start the trial clock;
  2. ask TiPToP for a plan from the current observation; locate the target's
     approach/close/retraction (plan_segments.py); refuse other structures;
  3. execute through the final retraction waypoint (applied via env.step), then hold the
     final joint target with the gripper closed, stepping physics;
  4. decide HOLDING_TARGET / FAILED_EMPTY / UNCERTAIN from simulator state (privileged);
  5. on FAILED_EMPTY with budget left: drop the old plan, open the empty gripper in place,
     move in joint space to the observation pose while checking that no object is disturbed,
     settle, and go back to 2 with a reconnected client (fresh server handler), new RGB-D
     and the current joints. The same instruction and target are used throughout.

Outputs (never runs.jsonl): <out>/recovery_trials.jsonl, <out>/attempts.jsonl and
<out>/trials/<trial_id>/{record.json, attempts.json, trace.npz, video.mp4, recovery_eval.log}.

The simulator-facing glue is the `IsaacWorld` adapter built in main(). Everything else here
(SimRecoveryBackend) only talks to a small `World` interface and the upstream client, so it is
unit-tested on CPU with stand-ins (tests/test_recovery_backend.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

PROCESS_T0 = time.monotonic()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_eval.checkpoint import (  # noqa: E402
    CheckpointLimits, CheckResult, OracleCheckCriteria, OracleStateDetector, OracleStatePacket, Outcome,
    RecoverySupervisor, StepOutcome, Stop, TrialResult, is_api_quota_error,
)
from grasp_eval.plan_segments import (  # noqa: E402
    ClientCursor, UnsupportedPlanError, clear_client_plan, find_grasp_segment, run_segment, tag_action,
)
from grasp_eval.probe_math import displaced_pose  # noqa: E402
from grasp_eval.records import append_jsonl  # noqa: E402
from grasp_eval.recovery_records import ATTEMPTS_FILE, TRIALS_FILE, AttemptRecord, RecoveryTrialRecord  # noqa: E402
from grasp_eval.scoring import LiftCriteria, count_close_commands, score_first_grasp  # noqa: E402
from grasp_eval.task_goal import geometry_center_positions  # noqa: E402

logger = logging.getLogger("recovery_eval")

DETECTORS = ("oracle-state",)
DEFAULT_GRIPPER_BODIES = ["left_inner_finger", "right_inner_finger"]  # as observed_eval; CHECK with --inspect-only
FINGER_JOINT = "finger_joint"
FINGER_CLOSED_RAD = math.pi / 4  # simulator's closed target for the Robotiq finger joint
VIDEO_FPS = 15
PHASES = ("approach", "close", "retract", "hold", "retry_open", "retry_return", "retry_settle")
TRACE_KEYS = ("t", "objects_pos", "objects_quat", "objects_center", "gripper_pos", "gripper_quat", "gripper_cmd", "gripper_obs",
              "joint_pos", "joint_cmd", "finger_closure", "attempt", "phase", "plan_step")


@dataclass(frozen=True)
class ReturnConfig:
    """Empty-hand return to an observation pose after a confirmed miss. First-version strategy:
    open in place, cosine joint-space interpolation, settle. It is not a collision-free plan; it is
    accepted only if no object moves more than max_object_motion_m (privileged monitor) and the
    arm ends within max_final_joint_error_rad of the target. Validate on video before trusting it."""

    max_joint_speed_rad_s: float = 0.5  # peak joint speed of the interpolation
    open_steps: int = 20  # same dwell as the upstream client's gripper actions
    settle_s: float = 0.5
    max_object_motion_m: float = 0.01
    max_final_joint_error_rad: float = 0.05
    observation_joints: tuple[float, ...] | None = None  # None: the trial's settled start joints

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["strategy"] = "open_in_place + cosine_joint_interpolation + settle; oracle object-disturbance monitor"
        return d


def joint_interpolation(q_from, q_to, max_speed_rad_s: float, dt: float) -> list[np.ndarray]:
    """Cosine-profile joint path from q_from to q_to (excluding q_from), peak speed <= max_speed."""
    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    span = float(np.max(np.abs(q_to - q_from))) if q_from.size else 0.0
    n = max(1, int(math.ceil((math.pi / 2) * span / (max_speed_rad_s * dt))))
    return [q_from + (q_to - q_from) * (0.5 - 0.5 * math.cos(math.pi * i / n)) for i in range(1, n + 1)]


def request_fingerprint(request: dict) -> dict[str, Any]:
    """Compact proof of what was sent to TiPToP (the arrays themselves are not stored)."""
    def digest(x):
        return None if x is None else sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    return {
        "q_init": [round(float(v), 5) for v in np.asarray(request.get("q_init")).ravel()],
        "rgb_sha256_16": digest(request.get("rgb")),
        "depth_sha256_16": digest(request.get("depth")),
        "world_from_cam": np.round(np.asarray(request.get("world_from_cam"), dtype=float), 5).tolist(),
        "task": request.get("task"),
    }


def classify_planning_exception(e: Exception, server_error: str | None = None) -> Stop:
    """PlanningError = the server answered without a plan; anything else = infrastructure.
    An API quota/rate-limit rejection is infrastructure even when it arrives as a PlanningError."""
    if is_api_quota_error(server_error) or is_api_quota_error(str(e)):
        return Stop.INFRASTRUCTURE
    return Stop.NO_PLAN if type(e).__name__ == "PlanningError" else Stop.INFRASTRUCTURE


class World(Protocol):
    step_dt: float
    object_names: list[str]
    # Optional: name -> [[min xyz], [max xyz]] geometry box in the body frame. Objects listed here
    # are scored at their geometry-box center; others fall back to the rigid-body origin.
    object_local_bounds: dict[str, Any]

    def step(self, action: np.ndarray) -> tuple[Any, bool, bool]: ...
    def object_positions(self) -> dict[str, list[float]]: ...
    def object_quats(self) -> dict[str, list[float]]: ...
    def gripper_ref(self) -> list[float]: ...
    def gripper_quat(self) -> list[float]: ...
    def finger_closure(self) -> float | None: ...
    def joint_pos(self, obs) -> np.ndarray: ...
    def gripper_obs(self, obs) -> float: ...
    def set_object_pose(self, name: str, pos, quat) -> None: ...


class SimRecoveryBackend:
    """RecoveryBackend over a World and the upstream TiPToP websocket client."""

    def __init__(self, world: World, client, obs, *, trial_id: str, instruction: str, target: str,
                 target_label: str | None, limits: CheckpointLimits, detector: OracleStateDetector,
                 lift_criteria: LiftCriteria, return_cfg: ReturnConfig, displacement: dict | None = None,
                 replay_plan: str | None = None, on_step: Callable[..., None] | None = None,
                 wall_clock: Callable[[], float] = time.monotonic):
        self.world = world
        self.client = client
        self.obs = obs
        self.trial_id = trial_id
        self.instruction = instruction
        self.target = target
        self.target_label = target_label
        self.limits = limits
        self.detector = detector
        self.lift = lift_criteria
        self.return_cfg = return_cfg
        self.displacement = displacement
        self.replay_plan = replay_plan
        self.on_step = on_step
        self._clock = wall_clock
        self._wall0 = wall_clock()
        self.dt = float(world.step_dt)
        self.names = list(world.object_names)
        self.trace: dict[str, list] = {k: [] for k in TRACE_KEYS}
        self.steps = 0
        self.attempts: list[AttemptRecord] = []
        self.start_joints = np.asarray(world.joint_pos(obs), dtype=float)
        self.bounds = dict(getattr(world, "object_local_bounds", None) or {})
        self.initial_positions = world.object_positions()
        self.initial_centers = self.object_centers()
        self._pending = None
        self._segment = None
        self._exec: dict[int, dict[str, int]] = {}
        self._displaced = False
        if target not in self.names:
            raise KeyError(f"target {target!r} not in {self.names}")

    # ---------------------------------------------------------------- helpers
    def sim_time(self) -> float:
        return self.steps * self.dt

    def wall_time(self) -> float:
        return self._clock() - self._wall0

    def object_centers(self, pos: dict | None = None, quat: dict | None = None) -> dict[str, list[float]]:
        """Geometry-box centers (the scored position); body origin when no box is known."""
        pos = self.world.object_positions() if pos is None else pos
        quat = self.world.object_quats() if quat is None else quat
        return {n: (geometry_center_positions(pos[n], quat[n], self.bounds[n]).tolist() if n in self.bounds
                    else list(pos[n])) for n in self.names}

    def position_reference(self, name: str) -> str:
        return "geometry_box_center" if name in self.bounds else "rigid_body_origin_fallback"

    def closes(self) -> int:
        return count_close_commands(np.asarray(self.trace["gripper_cmd"]), self.lift.close_threshold)

    def _attempt(self, i: int) -> AttemptRecord:
        while len(self.attempts) <= i:
            self.attempts.append(AttemptRecord(self.trial_id, len(self.attempts)))
        return self.attempts[i]

    def _step(self, action, phase: str, attempt: int, plan_step: int | None = None):
        action = np.asarray(action, dtype=float)
        obs, term, trunc = self.world.step(action)
        self.steps += 1
        self.obs = obs
        w, tr = self.world, self.trace
        pos, quat = w.object_positions(), w.object_quats()
        centers = self.object_centers(pos, quat)
        fc = w.finger_closure()
        tr["t"].append(self.steps * self.dt)
        tr["objects_pos"].append([pos[n] for n in self.names])
        tr["objects_quat"].append([quat[n] for n in self.names])
        tr["objects_center"].append([centers[n] for n in self.names])
        tr["gripper_pos"].append(list(w.gripper_ref()))
        tr["gripper_quat"].append(list(w.gripper_quat()))
        tr["gripper_cmd"].append(float(action[-1]))
        tr["gripper_obs"].append(float(w.gripper_obs(obs)))
        tr["joint_pos"].append([float(v) for v in w.joint_pos(obs)])
        tr["joint_cmd"].append([float(v) for v in action[:7]])
        tr["finger_closure"].append(float("nan") if fc is None else float(fc))
        tr["attempt"].append(attempt)
        tr["phase"].append(PHASES.index(phase))
        tr["plan_step"].append(-1 if plan_step is None else int(plan_step))
        if self.on_step is not None:
            self.on_step(obs, action, phase, attempt, plan_step)
        return obs, bool(term), bool(trunc)

    def _steps_left(self) -> int:
        return int(math.floor((self.limits.max_sim_s - self.sim_time()) / self.dt + 1e-9))

    def _read_response(self, a: AttemptRecord) -> dict | None:
        c = self.client
        a.planning_time_s = getattr(c, "last_planning_time", None)
        text = getattr(c, "last_response_text", None)
        if text is None:
            return None
        a.server_response_received = True
        try:
            resp = json.loads(text)
        except json.JSONDecodeError:
            a.server_success = False
            a.server_error = f"non-JSON server reply (server-side exception): {text[:500]}"
            return None
        a.server_success = bool(resp.get("success"))
        a.server_error = resp.get("error")
        a.server_save_dir = resp.get("save_dir")
        plan = resp.get("plan") or None
        if a.server_success and plan:
            a.plan_num_steps = len([s for s in plan.get("steps", []) if s.get("type") != "metadata"])
        if a.server_save_dir:
            a.artifacts["server_outputs"] = a.server_save_dir
            a.artifacts["server_plan"] = (self.replay_plan if a.plan_source == "replay"
                                          else str(Path(a.server_save_dir) / "tiptop_plan.json"))
            a.artifacts["server_log"] = str(Path(a.server_save_dir) / "tiptop_run.log")
        return plan

    def _apply_displacement(self) -> dict:
        d = dict(self.displacement)
        pos = np.asarray(self.world.object_positions()[self.target], dtype=float)
        quat = np.asarray(self.world.object_quats()[self.target], dtype=float)
        new_pos, new_quat = displaced_pose(pos, quat, d["dx_m"], d["dy_m"], d["yaw_deg"])
        self.world.set_object_pose(self.target, new_pos, new_quat)
        self._displaced = True
        d.update({"applied": True, "when": "attempt_0_after_plan_before_motion", "pos_before": pos.tolist(),
                  "pos_after": new_pos.tolist(), "quat_before": quat.tolist(), "quat_after": new_quat.tolist()})
        return d

    # ---------------------------------------------------------------- phases
    def observe(self, attempt: int) -> StepOutcome:
        a = self._attempt(attempt)
        a.started_sim_s = round(self.sim_time(), 4)
        a.started_wall_s = round(self.wall_time(), 3)
        a.start_joint_pos = [float(v) for v in self.world.joint_pos(self.obs)]
        a.object_positions = self.world.object_positions()
        a.object_quats = self.world.object_quats()
        a.object_centers = self.object_centers(a.object_positions, a.object_quats)
        a.trace_slice = [len(self.trace["t"]), len(self.trace["t"])]
        return StepOutcome.success(start_joint_pos=a.start_joint_pos, sim_s=a.started_sim_s)

    def plan(self, attempt: int) -> StepOutcome:
        a = self._attempt(attempt)
        c = self.client
        self._pending = self._segment = None
        if attempt == 0 and self.replay_plan:
            a.plan_source, a.replayed_plan = "replay", self.replay_plan
        else:
            a.plan_source = "server"
        if attempt > 0:
            # A genuinely fresh request: no stale plan/counters, no replay, a new server handler.
            # (A new connection is not proof that every server cache was cleared; see the runbook.)
            clear_client_plan(c)
            c.replay_plan_path = None
            c.last_response_text = None
            c.last_request_summary = None
            try:
                c.reset()
            except Exception as e:
                return StepOutcome.fail(Stop.INFRASTRUCTURE, f"reconnect to TiPToP failed: {e}")
        before = ClientCursor.of(c)
        t0 = self._clock()
        try:
            ret = c.infer(self.obs, self.instruction)
        except Exception as e:
            a.planning_wall_s = round(self._clock() - t0, 3)
            self._read_response(a)
            a.server_error = a.server_error or f"{type(e).__name__}: {e}"
            return StepOutcome.fail(classify_planning_exception(e, a.server_error), a.server_error)
        a.planning_wall_s = round(self._clock() - t0, 3)
        self._read_response(a)
        a.request = dict(getattr(c, "last_request_summary", None) or {})
        if a.plan_source == "replay":
            a.request["replayed"] = True
            a.request["q_actual"] = a.start_joint_pos
            q_plan = getattr(c, "replay_q_init", None)
            if q_plan is not None and a.start_joint_pos is not None:
                a.request["plan_q_init_max_abs_diff_rad"] = round(
                    float(np.max(np.abs(np.asarray(q_plan) - np.asarray(a.start_joint_pos)))), 5)
        if not c._plan:
            stop = Stop.INFRASTRUCTURE if is_api_quota_error(a.server_error) else Stop.NO_PLAN
            return StepOutcome.fail(stop, a.server_error or "empty plan")
        earlier = {x.server_save_dir for x in self.attempts[:attempt] if x.server_save_dir}
        if a.plan_source == "server" and a.server_save_dir and a.server_save_dir in earlier:
            return StepOutcome.fail(Stop.INFRASTRUCTURE, f"server reused output dir {a.server_save_dir}")
        try:
            seg = find_grasp_segment(c._plan, self.target_label)
        except UnsupportedPlanError as e:
            a.segment_error = str(e)
            return StepOutcome.fail(Stop.UNSUPPORTED_PLAN, str(e))
        a.segment = seg.to_dict()
        self._segment = seg
        self._pending = (ret, tag_action(before, ClientCursor.of(c)))
        return StepOutcome.success(segment=a.segment, server_save_dir=a.server_save_dir,
                                   planning_wall_s=a.planning_wall_s, plan_source=a.plan_source)

    def execute_grasp(self, attempt: int) -> StepOutcome:
        a = self._attempt(attempt)
        seg = self._segment
        if seg is None or self._pending is None:
            return StepOutcome.fail(Stop.INVALID_MOTION, "no validated plan to execute")
        if attempt == 0 and self.displacement and not self._displaced:
            a.displacement = self._apply_displacement()  # once per trial, before any motion
        closes_before = self.closes()
        start = len(self.trace["t"])
        hold_steps = max(1, int(round(self.limits.confirm_hold_s / self.dt)))
        max_actions = self._steps_left() - hold_steps
        if max_actions <= 0:
            return StepOutcome.fail(Stop.SIM_TIME_LIMIT, "no simulated time left for the grasp")

        def step_env(action, tag):
            if tag.kind == "gripper":
                phase = "close"
            elif tag.plan_step == seg.retraction_index:
                phase = "retract"
            else:
                phase = "approach"
            return self._step(action, phase, attempt, tag.plan_step)

        res = run_segment(self.client, self.obs, self.instruction, seg, step_env, max_actions, first=self._pending)
        self._pending = None
        a.actions_applied = res.actions_applied
        a.retraction_applied = res.retraction_applied
        a.final_joint_cmd = res.final_joint_cmd
        a.trace_slice = [a.trace_slice[0] if a.trace_slice else start, len(self.trace["t"])]
        if not res.ok:
            a.closes_in_attempt = self.closes() - closes_before
            if res.truncated:
                stop = Stop.EPISODE_TIME_LIMIT
            elif res.actions_applied >= max_actions:
                stop = Stop.SIM_TIME_LIMIT
            else:
                stop = Stop.INVALID_MOTION
            return StepOutcome.fail(stop, res.reason, actions_applied=res.actions_applied)
        if res.final_gripper_cmd is None or res.final_gripper_cmd <= self.lift.close_threshold:
            return StepOutcome.fail(Stop.INVALID_MOTION, "gripper not commanded closed at the end of the retraction")

        # Confirmation hold: keep the final commanded joint target and the closed gripper, stepping physics.
        hold_action = np.array(list(res.final_joint_cmd) + [float(res.final_gripper_cmd)])
        hold_start = len(self.trace["t"])
        for _ in range(hold_steps):
            _, term, trunc = self._step(hold_action, "hold", attempt)
            if term or trunc:
                a.trace_slice[1] = len(self.trace["t"])
                return StepOutcome.fail(Stop.EPISODE_TIME_LIMIT, "episode ended during the confirmation hold")
        end = len(self.trace["t"])
        q = np.asarray(self.world.joint_pos(self.obs), dtype=float)
        a.hold_steps, a.hold_s = hold_steps, round(hold_steps * self.dt, 4)
        a.final_joint_pos = [float(v) for v in q]
        a.final_joint_error_max_rad = round(float(np.max(np.abs(q - np.asarray(res.final_joint_cmd)))), 5)
        a.closes_in_attempt = self.closes() - closes_before
        a.trace_slice[1] = end
        self._exec[attempt] = {"start": start, "hold_start": hold_start, "end": end}
        return StepOutcome.success(actions_applied=res.actions_applied, hold_steps=hold_steps,
                                   retraction_step=seg.retraction_index, closes_in_attempt=a.closes_in_attempt,
                                   final_joint_error_max_rad=a.final_joint_error_max_rad)

    def check(self, attempt: int) -> CheckResult:
        a = self._attempt(attempt)
        ex = self._exec[attempt]
        tr = self.trace
        t = np.asarray(tr["t"], dtype=float)
        objs = np.asarray(tr["objects_center"], dtype=float)  # geometry centers, not body origins
        grip = np.asarray(tr["gripper_pos"], dtype=float)
        cmd = np.asarray(tr["gripper_cmd"], dtype=float)
        phase = np.asarray(tr["phase"])
        fc = np.asarray(tr["finger_closure"], dtype=float)
        k = self.names.index(self.target)
        idx = np.arange(ex["start"], ex["end"])
        closed = idx[cmd[idx] > self.lift.close_threshold]
        retract = idx[phase[idx] == PHASES.index("retract")]
        if closed.size == 0 or retract.size == 0:
            res = CheckResult(Outcome.UNCERTAIN, "no_close_or_retraction_in_attempt", float(t[ex["end"] - 1]),
                              self.detector.name, self.detector.privileged, {})
        else:
            c0 = int(closed[0])
            sl = slice(c0, ex["end"])
            fcs = fc[sl]
            packet = OracleStatePacket(
                t=t[sl], target_pos=objs[sl, k], gripper_pos=grip[sl], gripper_cmd=cmd[sl],
                retract_start=int(retract[0]) - c0, hold_start=ex["hold_start"] - c0,
                other_pos={n: objs[sl, j] for j, n in enumerate(self.names) if n != self.target},
                finger_closure=None if np.all(np.isnan(fcs)) else fcs,
            )
            res = self.detector.check(packet)
        att = slice(ex["start"], ex["end"])
        a.historical_lift = score_first_grasp(t[att], objs[att, k], grip[att], cmd[att],
                                              a.object_centers[self.target][2], self.lift).to_dict()
        a.historical_lift["position_reference"] = self.position_reference(self.target)
        a.checkpoint = res.to_dict()
        a.outcome, a.outcome_source, a.outcome_privileged = res.outcome.value, res.source, res.privileged
        return res

    def prepare_retry(self, attempt: int) -> StepOutcome:
        cfg = self.return_cfg
        a = self._attempt(attempt)
        clear_client_plan(self.client)  # the old transport/place actions are never executed
        info: dict[str, Any] = {"config": cfg.to_dict(), "privileged_monitor": True,
                                "start_sim_s": round(self.sim_time(), 4)}
        a.retry = info
        q_hold = np.asarray(a.final_joint_cmd, dtype=float)
        q_to = (np.asarray(cfg.observation_joints, dtype=float) if cfg.observation_joints is not None
                else self.start_joints)
        q_from = np.asarray(self.world.joint_pos(self.obs), dtype=float)
        path = joint_interpolation(q_from, q_to, cfg.max_joint_speed_rad_s, self.dt)
        settle_steps = max(1, int(round(cfg.settle_s / self.dt)))
        needed = cfg.open_steps + len(path) + settle_steps
        info.update({"observation_joints": [float(v) for v in q_to], "return_steps": len(path),
                     "planned_steps": needed})
        if needed > self._steps_left():
            return StepOutcome.fail(Stop.SIM_TIME_LIMIT, "not enough simulated time for the return motion", **info)
        p0 = {n: np.asarray(v, dtype=float) for n, v in self.object_centers().items()}

        def disturbed() -> tuple[str, float]:
            now = self.object_centers()
            return max(((n, float(np.linalg.norm(np.asarray(now[n]) - p0[n]))) for n in self.names),
                       key=lambda x: x[1])

        sequence = ([(np.r_[q_hold, 0.0], "retry_open")] * cfg.open_steps
                    + [(np.r_[q, 0.0], "retry_return") for q in path]
                    + [(np.r_[q_to, 0.0], "retry_settle")] * settle_steps)
        for action, phase in sequence:
            _, term, trunc = self._step(action, phase, attempt)
            name, moved = disturbed()
            if moved > cfg.max_object_motion_m:
                info.update({"disturbed_object": name, "disturbance_m": round(moved, 4), "aborted_in": phase})
                return StepOutcome.fail(Stop.RETURN_FAILED, f"{name} moved {moved:.3f} m during {phase}", **info)
            if term or trunc:
                return StepOutcome.fail(Stop.EPISODE_TIME_LIMIT, f"episode ended during {phase}", **info)
        err = float(np.max(np.abs(np.asarray(self.world.joint_pos(self.obs)) - q_to)))
        name, moved = disturbed()
        info.update({"final_joint_error_max_rad": round(err, 5), "max_object_motion_m": round(moved, 4),
                     "end_sim_s": round(self.sim_time(), 4)})
        if err > cfg.max_final_joint_error_rad:
            return StepOutcome.fail(Stop.RETURN_FAILED, f"observation pose not reached (error {err:.3f} rad)", **info)
        return StepOutcome.success(**info)


# --------------------------------------------------------------------------- trial assembly


def fill_trial_record(rec: RecoveryTrialRecord, result: TrialResult, backend: SimRecoveryBackend) -> None:
    rec.final_success = result.success
    rec.stop_reason = result.stop.value
    rec.stop_detail = result.detail
    rec.num_planning_calls = result.planning_calls
    rec.num_closes = result.closes
    rec.retry_triggered = result.retry_triggered
    rec.num_attempts = sum(1 for a in backend.attempts if a.actions_applied > 0)
    rec.sim_duration_s = round(result.sim_s, 4)
    rec.wall_duration_s = round(result.wall_s, 3)
    rec.transitions = result.transitions
    rec.harness_exception = result.harness_exception
    a0 = backend.attempts[0] if backend.attempts else None
    rec.first_attempt_success = bool(a0 and a0.outcome == Outcome.HOLDING_TARGET.value)
    rec.first_attempt_historical_lift = bool(a0 and (a0.historical_lift or {}).get("lifted"))
    if result.success:
        rec.time_to_success_sim_s = rec.sim_duration_s
        rec.time_to_success_wall_s = rec.wall_duration_s
    for i, log in enumerate(result.attempts):
        if i < len(backend.attempts) and log.stop:
            backend.attempts[i].stop = log.stop


# --------------------------------------------------------------------------- CLI / simulator glue


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="One checkpoint-recovery trial (week two)")
    p.add_argument("--sim-root", type=Path, default=Path.cwd())
    p.add_argument("--tiptop-dir", type=Path, default=None, help="tiptop checkout, for recording its SHA")
    p.add_argument("--scene", type=int, required=True)
    p.add_argument("--variant", type=int, required=True)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--instruction", required=True, help="original task instruction, used for every attempt")
    p.add_argument("--target-object", required=True, help="simulator rigid-object name (observed_eval --inspect-only)")
    p.add_argument("--target-label", default=None,
                   help="TiPToP's perceived name for the target; if given, the plan's Pick must match it")
    p.add_argument("--trial-id", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--headless", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--ws-host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--provenance-file", type=Path, default=None)
    p.add_argument("--gripper-bodies", nargs="+", default=DEFAULT_GRIPPER_BODIES)
    p.add_argument("--finger-joint", default=FINGER_JOINT, help="joint used for the (unvalidated) finger evidence")
    p.add_argument("--probe-label", default=None)

    g = p.add_argument_group("checkpoint and budgets")
    g.add_argument("--detector", choices=DETECTORS, default="oracle-state")
    d = CheckpointLimits()
    g.add_argument("--max-attempts", type=int, default=d.max_attempts, help="target closes (1 = matched control)")
    g.add_argument("--max-planning-calls", type=int, default=None, help="default: --max-attempts")
    g.add_argument("--max-sim-s", type=float, default=d.max_sim_s)
    g.add_argument("--max-wall-s", type=float, default=d.max_wall_s)
    g.add_argument("--confirm-hold-s", type=float, default=d.confirm_hold_s)
    o = OracleCheckCriteria()
    for f in ("min_gripper_rise_m", "held_min_follow_frac", "held_max_rel_drift_m", "empty_max_target_rise_m",
              "empty_max_target_motion_m", "other_near_m", "other_moved_m"):
        g.add_argument("--" + f.replace("_", "-"), type=float, default=getattr(o, f))
    g.add_argument("--empty-min-finger-closure", type=float, default=None)

    s = p.add_argument_group("week-one stable-lift criterion (historical score per attempt)")
    c = LiftCriteria()
    s.add_argument("--min-rise-m", type=float, default=c.min_rise_m)
    s.add_argument("--min-hold-s", type=float, default=c.min_hold_s)
    # The frozen 0.20 m shared with the oracle checkpoint, not the generic LiftCriteria default (0.10).
    s.add_argument("--max-obj-to-gripper-m", type=float, default=o.max_obj_to_gripper_m)

    r = p.add_argument_group("empty-hand return before a retry")
    rc = ReturnConfig()
    r.add_argument("--return-max-joint-speed", type=float, default=rc.max_joint_speed_rad_s)
    r.add_argument("--return-settle-s", type=float, default=rc.settle_s)
    r.add_argument("--return-max-object-motion-m", type=float, default=rc.max_object_motion_m)
    r.add_argument("--return-max-joint-error-rad", type=float, default=rc.max_final_joint_error_rad)
    r.add_argument("--observation-joints", type=float, nargs=7, default=None,
                   help="observation pose (default: the trial's settled start joints)")

    pr = p.add_argument_group("probes / paired controls (attempt 0 only)")
    pr.add_argument("--plan-json", type=Path, default=None,
                    help="replay this saved tiptop_plan.json for attempt 0; a retry always plans fresh")
    pr.add_argument("--displace-dx-m", type=float, default=0.0)
    pr.add_argument("--displace-dy-m", type=float, default=0.0)
    pr.add_argument("--displace-yaw-deg", type=float, default=0.0)

    args, _ = p.parse_known_args(argv)  # leave AppLauncher flags for Isaac Lab
    if args.max_planning_calls is None:
        args.max_planning_calls = args.max_attempts
    args.displace = any(v != 0 for v in (args.displace_dx_m, args.displace_dy_m, args.displace_yaw_deg))
    return args


def build_config(args) -> tuple[CheckpointLimits, OracleCheckCriteria, LiftCriteria, ReturnConfig]:
    limits = CheckpointLimits(args.max_attempts, args.max_planning_calls, args.max_sim_s, args.max_wall_s,
                              args.confirm_hold_s)
    check = OracleCheckCriteria(
        max_obj_to_gripper_m=args.max_obj_to_gripper_m, min_gripper_rise_m=args.min_gripper_rise_m,
        held_min_follow_frac=args.held_min_follow_frac, held_max_rel_drift_m=args.held_max_rel_drift_m,
        empty_max_target_rise_m=args.empty_max_target_rise_m,
        empty_max_target_motion_m=args.empty_max_target_motion_m, other_near_m=args.other_near_m,
        other_moved_m=args.other_moved_m, empty_min_finger_closure=args.empty_min_finger_closure)
    lift = LiftCriteria(min_rise_m=args.min_rise_m, min_hold_s=args.min_hold_s,
                        max_obj_to_gripper_m=args.max_obj_to_gripper_m)
    ret = ReturnConfig(max_joint_speed_rad_s=args.return_max_joint_speed, settle_s=args.return_settle_s,
                       max_object_motion_m=args.return_max_object_motion_m,
                       max_final_joint_error_rad=args.return_max_joint_error_rad,
                       observation_joints=tuple(args.observation_joints) if args.observation_joints else None)
    return limits, check, lift, ret


def make_recovery_client_class(observed_base):
    """The week-one ObservedTiptopClient plus a fingerprint of each outgoing request."""

    class RecoveryTiptopClient(observed_base):
        last_request_summary: dict | None = None

        def _build_request(self, raw_obs, curr_obs, instruction):
            request = super()._build_request(raw_obs, curr_obs, instruction)
            self.last_request_summary = request_fingerprint(request)
            return request

    return RecoveryTiptopClient


class _StreamingVideo:
    """Writes frames as they come (a 180 s trial is too long to buffer like week one)."""

    def __init__(self, path: Path, fps: int):
        self.path, self.fps, self._w, self.frames = path, fps, None, 0

    def add(self, frame: np.ndarray) -> None:
        import mediapy

        if self._w is None:
            self._w = mediapy.VideoWriter(self.path, shape=frame.shape[:2], fps=self.fps)
            self._w.__enter__()
        self._w.add_image(frame)
        self.frames += 1

    def close(self) -> None:
        if self._w is not None:
            self._w.__exit__(None, None, None)
            self._w = None


def main(argv=None) -> int:  # pragma: no cover - needs Isaac Sim
    args = parse_args(argv)
    sim_root = args.sim_root.resolve()
    sys.path.insert(0, str(sim_root))
    os.chdir(sim_root)

    from grasp_eval.observed_eval import (_file_sha256, _git_diff_sha256, _git_dirty, _git_sha, _utc_now,
                                          make_observed_client_class, usd_local_bounds)

    trial_id = args.trial_id or (f"s{args.scene}v{args.variant}t{args.trial}_a{args.max_attempts}_"
                                 f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out = args.out_dir.resolve()
    trial_dir = out / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(trial_dir / "recovery_eval.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    import gymnasium as gym
    import torch

    if args.seed is not None:
        torch.manual_seed(args.seed)
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(argv)
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    simulation_app = AppLauncher(args_cli).app

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.inference.tiptop_websocket import TiptopWebsocketClient
    from src.sim_evals.sim_utils import settle_sim
    from src.visual_utils import add_top_padding, overlay_timer_ms

    from grasp_eval.video_overlay import add_caption

    limits, check_criteria, lift, return_cfg = build_config(args)
    probe = {"type": "baseline", "label": args.probe_label}
    if args.plan_json or args.displace:
        probe = {"type": "+".join(k for k, on in (("replay", args.plan_json), ("displacement", args.displace)) if on),
                 "label": args.probe_label}
    displacement = ({"dx_m": args.displace_dx_m, "dy_m": args.displace_dy_m, "yaw_deg": args.displace_yaw_deg}
                    if args.displace else None)
    if displacement:
        probe["displacement"] = dict(displacement, when="attempt_0_after_plan_before_motion")
    episode_length_s = limits.max_sim_s + 10.0  # our own limit fires first; the env's is the backstop
    rec = RecoveryTrialRecord(
        trial_id=trial_id, scene=args.scene, variant=args.variant, trial=args.trial, instruction=args.instruction,
        target_object=args.target_object, target_label=args.target_label, mode=limits.mode,
        detector=args.detector, detector_privileged=True, limits=limits.to_dict(), criteria=lift.to_dict(),
        check_criteria=check_criteria.to_dict(), return_config=return_cfg.to_dict(), probe=probe, seed=args.seed,
        started_utc=_utc_now(), episode_length_s=episode_length_s,
        versions={
            "project_sha": _git_sha(PROJECT_ROOT), "project_dirty": _git_dirty(PROJECT_ROOT),
            "sim_sha": _git_sha(sim_root), "sim_dirty": _git_dirty(sim_root),
            "tiptop_sha": _git_sha(args.tiptop_dir) if args.tiptop_dir else None,
            "tiptop_diff_sha256": _git_diff_sha256(args.tiptop_dir) if args.tiptop_dir else None,
            "provenance_file": str(args.provenance_file) if args.provenance_file else None,
            "provenance_sha256": _file_sha256(args.provenance_file) if args.provenance_file else None,
        },
    )
    env = client = backend = None
    video = _StreamingVideo(trial_dir / "video.mp4", VIDEO_FPS)
    try:
        env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
        env_cfg.set_scene(str(args.scene), args.variant)
        env_cfg.episode_length_s = episode_length_s
        if args.seed is not None:
            env_cfg.seed = args.seed
        env = gym.make("DROID", cfg=env_cfg)
        scene = env.unwrapped.scene
        robot = scene["robot"]
        device = env.unwrapped.device
        names = sorted(scene.rigid_objects.keys())
        rec.object_names = names
        if args.target_object not in names:
            raise KeyError(f"target {args.target_object!r} not in {names}; run observed_eval.py --inspect-only")
        body_ids, body_names = robot.find_bodies(args.gripper_bodies, preserve_order=True)
        if len(body_ids) != len(args.gripper_bodies):
            raise KeyError(f"gripper bodies {args.gripper_bodies} matched {body_names}")
        finger_ids, _ = robot.find_joints([args.finger_joint])
        try:  # before the first physics step (Fabric stops USD transform updates); as observed_eval
            rec.object_local_bounds, rec.object_bounds_notes = usd_local_bounds(scene, names)
        except Exception as e:  # fall back to body origins rather than lose the trial
            rec.object_bounds_notes = {"_error": f"{type(e).__name__}: {e}"}

        class IsaacWorld:
            step_dt = float(env.unwrapped.step_dt)
            object_names = names
            object_local_bounds = rec.object_local_bounds

            def step(self, action):
                o, _, term, trunc, _ = env.step(torch.tensor(action, dtype=torch.float32)[None])
                return o, bool(term), bool(trunc)

            def object_positions(self):
                return {n: scene.rigid_objects[n].data.root_pos_w[0].tolist() for n in names}

            def object_quats(self):
                return {n: scene.rigid_objects[n].data.root_quat_w[0].tolist() for n in names}

            def gripper_ref(self):
                return robot.data.body_pos_w[0, body_ids].mean(dim=0).tolist()

            def gripper_quat(self):
                return robot.data.body_quat_w[0, body_ids[0]].tolist()

            def finger_closure(self):
                if not finger_ids:
                    return None
                return float(robot.data.joint_pos[0, finger_ids[0]]) / FINGER_CLOSED_RAD

            def joint_pos(self, o):
                return o["policy"]["arm_joint_pos"][0].cpu().numpy()

            def gripper_obs(self, o):
                return float(o["policy"]["gripper_pos"][0, 0])

            def set_object_pose(self, name, pos, quat):
                obj = scene[name]
                obj.write_root_pose_to_sim(torch.tensor([[*pos, *quat]], dtype=torch.float32, device=device))
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))

        world = IsaacWorld()
        caption_ok = [True]

        def on_step(o, action, phase, attempt, plan_step):
            try:
                pol = o["policy"]
                viz = np.concatenate([pol["external_cam"][0].cpu().numpy(), pol["wrist_cam"][0].cpu().numpy()], axis=1)
                viz = add_top_padding(viz, pad_px=40)
                overlay_timer_ms(viz, int(backend.steps * 1000 / VIDEO_FPS))
                if caption_ok[0]:
                    rise = backend.object_centers()[args.target_object][2] - \
                        backend.initial_centers[args.target_object][2]
                    step = "" if plan_step is None else f" step {plan_step + 1}"
                    viz = add_caption(viz, f"attempt {attempt + 1}/{limits.max_attempts} | {phase}{step} | gripper "
                                           f"{'CLOSE' if action[-1] > 0.5 else 'open'} | target {100 * rise:+.1f} cm")
                video.add(viz)
            except Exception:
                caption_ok[0] = False
                logger.warning("video frame skipped:\n%s", traceback.format_exc())

        ClientCls = make_recovery_client_class(make_observed_client_class(TiptopWebsocketClient))
        if args.plan_json:
            ClientCls.replay_plan_path = args.plan_json.resolve()  # attempt 0 only; cleared before a retry
        with torch.no_grad():
            obs, _ = env.reset()
            obs, _ = env.reset()  # upstream: second render cycle for materials
            obs, _ = env.reset()
            obs = settle_sim(env, obs, reset_episode_buf=True)
            rec.initial_object_positions = world.object_positions()
            client = ClientCls(host=args.ws_host, port=args.ws_port)
            backend = SimRecoveryBackend(
                world, client, obs, trial_id=trial_id, instruction=args.instruction, target=args.target_object,
                target_label=args.target_label, limits=limits, detector=OracleStateDetector(check_criteria),
                lift_criteria=lift, return_cfg=return_cfg, displacement=displacement,
                replay_plan=str(args.plan_json.resolve()) if args.plan_json else None, on_step=on_step)
            rec.initial_object_centers = backend.initial_centers
            rec.process_startup_s = round(time.monotonic() - PROCESS_T0, 2)
            result = RecoverySupervisor(backend, limits).run()
        fill_trial_record(rec, result, backend)
        rec.final_object_positions = world.object_positions()
        rec.final_object_centers = backend.object_centers()
    except Exception:
        rec.harness_exception = traceback.format_exc()
        rec.stop_reason = rec.stop_reason or Stop.INFRASTRUCTURE.value
        logger.error("Harness exception:\n%s", rec.harness_exception)
    finally:
        try:
            video.close()
            if video.frames:
                rec.artifacts["video"] = str(video.path)
            if backend is not None and backend.trace["t"]:
                arrays = {k: np.asarray(v) for k, v in backend.trace.items()}
                np.savez_compressed(trial_dir / "trace.npz", object_names=np.asarray(backend.names),
                                    phase_names=np.asarray(PHASES), **arrays)
                rec.artifacts["trace"] = str(trial_dir / "trace.npz")
        except Exception:
            rec.artifacts["artifact_error"] = traceback.format_exc()
        rec.artifacts["log"] = str(trial_dir / "recovery_eval.log")
        rec.finished_utc = _utc_now()
        attempts = [a.to_dict() for a in (backend.attempts if backend is not None else [])]
        (trial_dir / "attempts.json").write_text(json.dumps(attempts, indent=2, sort_keys=True))
        (trial_dir / "record.json").write_text(json.dumps(rec.to_dict(), indent=2, sort_keys=True))
        for a in attempts:
            append_jsonl(out / ATTEMPTS_FILE, a)
        append_jsonl(out / TRIALS_FILE, rec.to_dict())
        logger.info("Trial %s: %s (%s), attempts=%d closes=%d plans=%d sim=%.1fs", trial_id, rec.stop_reason,
                    rec.stop_detail, rec.num_attempts, rec.num_closes, rec.num_planning_calls,
                    rec.sim_duration_s or 0.0)
        if client is not None:
            client.close()
        if env is not None:
            env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
