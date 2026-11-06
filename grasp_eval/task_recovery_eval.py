"""One task-level checkpoint-recovery trial in droid-sim-evals (docs/task_recovery.md).

Run from the droid-sim-evals checkout, inside its uv environment (UV_NO_SYNC=1):

    uv run python $PROJECT/grasp_eval/task_recovery_eval.py --scene 1 --variant 0 --trial 0 --seed 2000 \
        --instruction "Put the Rubik's cube in the bowl." --goal-json '<scenes.json goal>' \
        --target-object rubiks_cube --lift-mode reverse_approach \
        --plan-json <baseline server_save_dir>/tiptop_plan.json --paired-baseline-run 000_s1v0_scene1_t0 \
        --out-dir $PROJECT/results/task_smoke

The whole task runs: after every grasp the arm lifts the object slowly while a monitor watches it
(grasp_eval/task_checkpoint.py). Held or uncertain: back down to the plan, continue unchanged.
Confirmed empty or slipped: open (setting a slipping object down first), go to the observation pose,
new TiPToP plan for the same instruction, continue with that plan. Task success is scored with the
baseline's goal check (grasp_eval/task_goal.py) on the final state, and again on the state at the
baseline's 90 s episode end for a like-for-like paired comparison.

Lift modes: `reverse_approach` retraces the plan's approach toward its start pose (no new motion code;
the plan's retraction end is on that path, so the arm rejoins the plan exactly). `vertical_ik` runs
the plan's retraction, then lifts straight up with damped least-squares IK on the simulator's Jacobian.
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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

PROCESS_T0 = time.monotonic()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_eval.checkpoint import is_api_quota_error  # noqa: E402
from grasp_eval.plan_segments import (  # noqa: E402
    ClientCursor, GraspUnit, UnsupportedPlanError, _norm, clear_client_plan, find_grasp_units, reverse_approach_path,
    run_actions, skip_client_to, tag_action,
)
from grasp_eval.recovery_eval import ReturnConfig, joint_interpolation  # noqa: E402
from grasp_eval.recovery_records import TASK_TRIALS_FILE, TaskTrialRecord  # noqa: E402
from grasp_eval.records import append_jsonl  # noqa: E402
from grasp_eval.task_checkpoint import (  # noqa: E402
    LIFT_MODES, LiftMonitor, Step, TaskCheckCriteria, TaskLimits, TaskResult, TaskStop, TaskSupervisor, Verdict,
)
from grasp_eval.task_goal import check_goal, geometry_center_positions  # noqa: E402

logger = logging.getLogger("task_recovery_eval")

PHASES = ("approach", "close", "retract", "lift", "hold", "descend", "transport", "open", "return", "set_down",
          "continue")
TRACE_KEYS = ("t", "objects_pos", "objects_quat", "objects_center", "gripper_pos", "gripper_cmd", "gripper_obs",
              "joint_pos", "joint_cmd", "attempt", "unit", "phase", "plan_step")
GRIPPER_DWELL_STEPS = 20  # the upstream client's gripper action length
DEFAULT_REVERSE_STRIDE = 1.5  # plan waypoints per sim step: half the client's speed (it uses 3)
DEFAULT_IK_SPEED = 0.03  # m/s for the vertical lift


class TaskWorld(Protocol):
    step_dt: float
    object_names: list[str]
    object_local_bounds: dict[str, Any]

    def step(self, action: np.ndarray) -> tuple[Any, bool, bool]: ...
    def object_positions(self) -> dict[str, list[float]]: ...
    def object_quats(self) -> dict[str, list[float]]: ...
    def gripper_ref(self) -> list[float]: ...
    def joint_pos(self, obs) -> np.ndarray: ...
    def gripper_obs(self, obs) -> float: ...
    def vertical_lift_step(self, q_cmd: np.ndarray, dz: float) -> np.ndarray: ...  # vertical_ik only
    def set_object_pose(self, name: str, pos, quat) -> None: ...  # displacement fixture only


def map_label(label: str | None, names: list[str], aliases: dict[str, str]) -> str | None:
    """TiPToP's perceived object label -> simulator object name (aliases, then name match)."""
    if not label:
        return None
    n = _norm(label)
    for k, v in aliases.items():
        if _norm(k) == n and v in names:
            return v
    for name in names:
        if _norm(name) == n:
            return name
    return None


class TaskSimBackend:
    """TaskBackend over a TaskWorld and the upstream TiPToP websocket client."""

    def __init__(self, world: TaskWorld, client, obs, *, instruction: str, lift_mode: str, limits: TaskLimits,
                 criteria: TaskCheckCriteria, return_cfg: ReturnConfig, goal: list[dict] | None = None,
                 target_object: str | None = None, aliases: dict[str, str] | None = None,
                 replay_plan: str | None = None, displacement: dict | None = None,
                 reverse_stride: float = DEFAULT_REVERSE_STRIDE, ik_speed: float = DEFAULT_IK_SPEED,
                 on_step: Callable[..., None] | None = None, wall_clock: Callable[[], float] = time.monotonic):
        if lift_mode not in LIFT_MODES:
            raise ValueError(f"lift_mode must be one of {LIFT_MODES}")
        self.world, self.client, self.obs = world, client, obs
        self.instruction, self.lift_mode, self.limits, self.c = instruction, lift_mode, limits, criteria
        self.return_cfg, self.goal, self.target_object = return_cfg, goal, target_object
        self.aliases = aliases or {}
        self.replay_plan, self.displacement = replay_plan, displacement
        self.reverse_stride, self.ik_speed, self.on_step = reverse_stride, ik_speed, on_step
        self._clock = wall_clock
        self.dt = float(world.step_dt)
        self.names = list(world.object_names)
        self.bounds = dict(getattr(world, "object_local_bounds", None) or {})
        self.trace: dict[str, list] = {k: [] for k in TRACE_KEYS}
        self.steps = 0
        self.start_joints = np.asarray(world.joint_pos(obs), dtype=float)
        self.initial_positions = world.object_positions()
        self.initial_quats = world.object_quats()
        self.plans: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.units: list[GraspUnit] = []
        self.steps_plan: list[dict] = []
        self._pending = None
        self._displaced = False
        self._attempt = 0
        self._unit = -1
        self._lift: dict[str, Any] = {}
        self._carry: dict[str, Any] | None = None
        self._snapshot: dict[str, Any] | None = None
        self.last_cmd = np.r_[self.start_joints, 0.0]

    # ---------------------------------------------------------------- state and stepping
    def sim_time(self) -> float:
        return self.steps * self.dt

    def centers(self, pos=None, quat=None) -> dict[str, np.ndarray]:
        pos = self.world.object_positions() if pos is None else pos
        quat = self.world.object_quats() if quat is None else quat
        return {n: (geometry_center_positions(pos[n], quat[n], self.bounds[n]) if n in self.bounds
                    else np.asarray(pos[n], dtype=float)) for n in self.names}

    def _event(self, what: str, **kw) -> None:
        self.events.append({"what": what, "t_sim": round(self.sim_time(), 3), "attempt": self._attempt, **kw})

    def _step(self, action, phase: str, plan_step: int | None = None):
        action = np.asarray(action, dtype=float)
        obs, term, trunc = self.world.step(action)
        self.steps += 1
        self.obs = obs
        self.last_cmd = action
        pos, quat = self.world.object_positions(), self.world.object_quats()
        cen = self.centers(pos, quat)
        tr = self.trace
        tr["t"].append(self.steps * self.dt)
        tr["objects_pos"].append([list(pos[n]) for n in self.names])
        tr["objects_quat"].append([list(quat[n]) for n in self.names])
        tr["objects_center"].append([cen[n].tolist() for n in self.names])
        tr["gripper_pos"].append(list(self.world.gripper_ref()))
        tr["gripper_cmd"].append(float(action[-1]))
        tr["gripper_obs"].append(float(self.world.gripper_obs(obs)))
        tr["joint_pos"].append([float(v) for v in self.world.joint_pos(obs)])
        tr["joint_cmd"].append([float(v) for v in action[:7]])
        tr["attempt"].append(self._attempt)
        tr["unit"].append(self._unit)
        tr["phase"].append(PHASES.index(phase))
        tr["plan_step"].append(-1 if plan_step is None else int(plan_step))
        if self._snapshot is None and self.sim_time() >= self.limits.baseline_episode_s:
            self._snapshot = {"positions": pos, "quats": quat, "gripper_open": bool(action[-1] <= 0.5)}
        self._watch_carry(cen, action)
        if self.on_step is not None:
            self.on_step(obs, action, phase, self._attempt, plan_step)
        return obs, bool(term), bool(trunc)

    def _watch_carry(self, cen, action) -> None:
        """Record (never act on) an object that leaves the hand while the gripper is still closed."""
        c = self._carry
        if c is None:
            return
        if action[-1] <= 0.5:
            self._carry = None  # released on purpose by the plan
            return
        off = cen[c["object"]] - np.asarray(self.world.gripper_ref(), dtype=float)
        if left_hand(c["offset"], off):
            self._event("dropped_during_carry", object=c["object"], unit=c["unit"])
            self._carry = None

    def _steps_left(self) -> int:
        return int(math.floor((self.limits.max_sim_s - self.sim_time()) / self.dt + 1e-9))

    # ---------------------------------------------------------------- planning
    def _read_response(self, rec: dict[str, Any]) -> None:
        c = self.client
        rec["planning_time_s"] = getattr(c, "last_planning_time", None)
        text = getattr(c, "last_response_text", None)
        if text is None:
            return
        try:
            resp = json.loads(text)
        except json.JSONDecodeError:
            rec["server_success"], rec["server_error"] = False, f"non-JSON server reply: {text[:300]}"
            return
        rec["server_success"] = bool(resp.get("success"))
        rec["server_error"] = resp.get("error")
        rec["server_save_dir"] = resp.get("save_dir")

    def plan(self, attempt: int) -> Step:
        self._attempt, self._unit = attempt, -1
        c = self.client
        rec: dict[str, Any] = {"attempt": attempt, "source": "replay" if attempt == 0 and self.replay_plan else "server",
                               "start_sim_s": round(self.sim_time(), 3),
                               "start_joint_pos": [round(float(v), 5) for v in self.world.joint_pos(self.obs)]}
        self.plans.append(rec)
        if attempt > 0:
            clear_client_plan(c)
            c.replay_plan_path = None
            c.last_response_text = None
            try:
                c.reset()
            except Exception as e:
                return Step.fail(TaskStop.INFRASTRUCTURE, f"reconnect to TiPToP failed: {e}")
        before = ClientCursor.of(c)
        t0 = self._clock()
        try:
            ret = c.infer(self.obs, self.instruction)
        except Exception as e:
            rec["planning_wall_s"] = round(self._clock() - t0, 3)
            self._read_response(rec)
            err = rec.get("server_error") or f"{type(e).__name__}: {e}"
            rec["server_error"] = err
            infra = is_api_quota_error(err) or type(e).__name__ != "PlanningError"
            stop = TaskStop.INFRASTRUCTURE if infra else (TaskStop.NO_PLAN if attempt == 0 else TaskStop.RETRY_NO_PLAN)
            return Step.fail(stop, err)
        rec["planning_wall_s"] = round(self._clock() - t0, 3)
        self._read_response(rec)
        if not c._plan:
            err = rec.get("server_error") or "empty plan"
            stop = TaskStop.INFRASTRUCTURE if is_api_quota_error(err) else (
                TaskStop.NO_PLAN if attempt == 0 else TaskStop.RETRY_NO_PLAN)
            return Step.fail(stop, err)
        self.steps_plan = list(c._plan)
        try:
            self.units = find_grasp_units(self.steps_plan)
        except UnsupportedPlanError as e:
            self.units = []
            rec["units_error"] = str(e)  # executed unchanged, without checkpoints
        labels = [u.target_label for u in self.units]
        rec.update({"num_steps": len(self.steps_plan), "units": [u.to_dict() for u in self.units],
                    "unit_objects": [map_label(lb, self.names, self.aliases) for lb in labels]})
        self._pending = (ret, tag_action(before, ClientCursor.of(c)))
        return Step.success(units=list(range(len(self.units))), unit_labels=labels)

    # ---------------------------------------------------------------- execution up to a checkpoint
    def _phase_of(self, tag, unit: GraspUnit | None) -> str:
        if tag.kind == "gripper":
            return "close" if unit is not None and tag.plan_step == unit.close_index else "open"
        if tag.kind == "hold" or tag.plan_step is None:
            return "continue"
        if unit is None:
            return "continue"
        if tag.plan_step < unit.first_step:
            return "transport"
        if tag.plan_step < unit.close_index:
            return "approach"
        if tag.plan_step == unit.retraction_index:
            return "retract"
        return "continue"

    def run_to_checkpoint(self, attempt: int, unit: int) -> Step:
        self._unit = unit
        u = self.units[unit]
        if attempt == 0 and unit == 0 and self.displacement and not self._displaced:
            self._apply_displacement(u)
        boundary = u.close_index if self.lift_mode == "reverse_approach" else u.retraction_index
        start = len(self.trace["t"])

        def step_env(action, tag):
            return self._step(action, self._phase_of(tag, u), tag.plan_step)

        res = run_actions(self.client, self.obs, self.instruction, step_env, max(1, self._steps_left()), boundary,
                          first=self._pending)
        self._pending = None
        if not res.ok:
            stop = (TaskStop.EPISODE_TIME_LIMIT if res.truncated else
                    TaskStop.SIM_TIME_LIMIT if self._steps_left() <= 0 else TaskStop.INVALID_MOTION)
            return Step.fail(stop, res.reason)
        # this grasp's close, by plan step: the gripper may still be closed from carrying the previous object
        ph = np.asarray(self.trace["phase"])
        idx = np.arange(start, len(ph))
        closing = idx[ph[idx] == PHASES.index("close")]
        if closing.size == 0:
            return Step.fail(TaskStop.INVALID_MOTION, f"grasp {unit}: the gripper was never commanded closed")
        c0 = int(closing[0])
        self._lift = {"close_sample": c0, "lift_start_sample": len(self.trace["t"]),
                      "q_lift_start": np.asarray(res.final_joint_cmd, dtype=float)}
        return Step.success(actions=res.actions_applied)

    def _apply_displacement(self, u: GraspUnit) -> None:
        from grasp_eval.probe_math import displaced_pose

        d = dict(self.displacement)
        target = map_label(u.target_label, self.names, self.aliases) or self.target_object
        pos = np.asarray(self.world.object_positions()[target], dtype=float)
        quat = np.asarray(self.world.object_quats()[target], dtype=float)
        new_pos, new_quat = displaced_pose(pos, quat, d.get("dx_m", 0.0), d.get("dy_m", 0.0), d.get("yaw_deg", 0.0))
        self.world.set_object_pose(target, new_pos, new_quat)
        self._displaced = True
        self._event("displacement_fixture", object=target, before=pos.tolist(), after=np.asarray(new_pos).tolist())

    # ---------------------------------------------------------------- the monitored lift
    def _target_of(self, unit: GraspUnit, close_sample: int) -> tuple[str, str]:
        name = map_label(unit.target_label, self.names, self.aliases)
        if name is not None:
            return name, "plan_label"
        g = np.asarray(self.trace["gripper_pos"][close_sample])
        cen = np.asarray(self.trace["objects_center"][close_sample])
        k = int(np.argmin(np.linalg.norm(cen - g, axis=1)))
        return self.names[k], "nearest_at_close"

    def lift_and_check(self, attempt: int, unit: int) -> dict[str, Any]:
        u = self.units[unit]
        L = self._lift
        c0 = L["close_sample"]
        target, source = self._target_of(u, c0)
        k = self.names.index(target)
        cen = np.asarray(self.trace["objects_center"], dtype=float)
        grip = np.asarray(self.trace["gripper_pos"], dtype=float)
        r0 = max(0, c0 - 1)  # resting state: the sample before the close (the close may already move things)
        others0 = {n: cen[r0, j] for j, n in enumerate(self.names) if n != target}
        mon = LiftMonitor(self.c, self.dt, grip[c0], cen[r0, k], others0)
        # vertical_ik: the plan's retraction already ran; let the monitor see it first
        for i in range(c0, L["lift_start_sample"]):  # the close dwell (and, for vertical_ik, the retraction)
            if self.trace["gripper_cmd"][i] > 0.5:
                mon.update(grip[i], cen[i, k], {n: cen[i, j] for j, n in enumerate(self.names) if n != target})
        q = np.asarray(L["q_lift_start"], dtype=float)
        path = (iter(reverse_approach_path(self.steps_plan, u, self.reverse_stride))
                if self.lift_mode == "reverse_approach" else None)
        applied: list[np.ndarray] = []
        hand0_z = grip[c0, 2]
        verdict = mon.verdict
        trunc_stop = False
        while verdict is None:
            if self._steps_left() <= 0:
                verdict = mon.at_lift_end()
                break
            if mon.phase == "lift":
                hand_rise = float(self.world.gripper_ref()[2] - hand0_z)
                nxt = None
                if hand_rise < self.c.hand_rise_cap_m:
                    if path is not None:
                        nxt = next(path, None)
                    else:
                        nxt = np.asarray(self.world.vertical_lift_step(q, self.ik_speed * self.dt), dtype=float)
                if nxt is None:
                    verdict = mon.at_lift_end()
                    break
                q = np.asarray(nxt, dtype=float)
                applied.append(q.copy())
                phase = "lift"
            else:
                phase = "hold"
            _, term, trunc = self._step(np.r_[q, 1.0], phase)
            cen_now = self.centers()
            verdict = mon.update(self.world.gripper_ref(), cen_now[target],
                                 {n: cen_now[n] for n in self.names if n != target})
            if (term or trunc) and verdict is None:
                verdict, trunc_stop = mon.at_lift_end(), True
        self._lift.update({"applied": applied, "q_top": q.copy(), "target": target, "verdict": verdict,
                           "offset": cen[c0, k] - grip[c0]})
        ev = mon.evidence()
        out = {"verdict": verdict, "reason": mon.reason, "sim_object": target, "target_source": source,
               "lift_mode": self.lift_mode, "lift_steps": len(applied), "evidence": ev,
               "t_sim": round(self.sim_time(), 3)}
        if trunc_stop:
            out["episode_ended"] = True
        return out

    # ---------------------------------------------------------------- after the verdict
    def _replay(self, cmds: list[np.ndarray], gripper: float, phase: str, until: Callable[[], bool] | None = None) -> bool:
        """Apply joint commands in order; returns False if the episode ended."""
        for q in cmds:
            _, term, trunc = self._step(np.r_[q, gripper], phase)
            if term or trunc:
                return False
            if until is not None and until():
                break
        return True

    def resume(self, attempt: int, unit: int) -> Step:
        """Back down onto the plan (gripper unchanged: closed) and let the client continue it."""
        u = self.units[unit]
        L = self._lift
        applied = L.get("applied") or []
        down = applied[::-1]
        if self.lift_mode == "reverse_approach":
            # the plan's retraction ends at the pre-grasp, which is on the reversed approach path
            q_ret_end = np.asarray(self.steps_plan[u.retraction_index]["positions"][-1], dtype=float)
            if applied:
                dists = [float(np.max(np.abs(np.asarray(a) - q_ret_end))) for a in applied]
                j = int(np.argmin(dists))
                down = applied[j:][::-1]  # from the top back down to the waypoint nearest the retraction end
            if not self._replay(down, 1.0, "descend"):
                return Step.fail(TaskStop.EPISODE_TIME_LIMIT, "episode ended while descending to the plan")
            here = np.asarray(self.last_cmd[:7], dtype=float)
            for q in joint_interpolation(here, q_ret_end, 0.25, self.dt):
                _, term, trunc = self._step(np.r_[q, 1.0], "descend")
                if term or trunc:
                    return Step.fail(TaskStop.EPISODE_TIME_LIMIT, "episode ended while descending to the plan")
            skip_client_to(self.client, u.retraction_index + 1, 1.0)
        else:
            if not self._replay(down, 1.0, "descend"):
                return Step.fail(TaskStop.EPISODE_TIME_LIMIT, "episode ended while descending to the plan")
        if L.get("verdict") == Verdict.HELD:
            g = np.asarray(self.world.gripper_ref(), dtype=float)
            target = L["target"]
            self._carry = {"object": target, "unit": unit, "offset": self.centers()[target] - g}
        return Step.success()

    def recover(self, attempt: int, unit: int, verdict: Verdict) -> Step:
        """Empty: open in place. Slipped: lower until the object rests ~1 cm above its start, open.
        Then go to the observation pose (the trial's start joints) for a fresh plan."""
        L = self._lift
        applied = L.get("applied") or []
        target = L.get("target")
        info: dict[str, Any] = {"verdict": verdict.value}
        if verdict == Verdict.SLIPPED and applied:
            k = self.names.index(target)
            rest_z = float(self.trace["objects_center"][max(0, L["close_sample"] - 1)][k][2])

            def low_enough() -> bool:
                return float(self.centers()[target][2]) - rest_z <= self.c.set_down_rise_m

            if not low_enough() and not self._replay(applied[::-1], 1.0, "set_down", until=low_enough):
                return Step.fail(TaskStop.EPISODE_TIME_LIMIT, "episode ended during set-down")
            info["set_down_object_rise_m"] = round(float(self.centers()[target][2]) - rest_z, 4)
        hold_q = np.asarray(self.last_cmd[:7], dtype=float)
        for _ in range(GRIPPER_DWELL_STEPS):
            _, term, trunc = self._step(np.r_[hold_q, 0.0], "open")
            if term or trunc:
                return Step.fail(TaskStop.EPISODE_TIME_LIMIT, "episode ended while opening")
        clear_client_plan(self.client)
        self._carry = None
        # empty-handed route to the observation pose
        p0 = {n: v.copy() for n, v in self.centers().items()}

        def disturbed() -> tuple[str, float]:
            now = self.centers()
            return max(((n, float(np.linalg.norm(now[n] - p0[n]))) for n in self.names), key=lambda x: x[1])

        route: list[np.ndarray] = []
        if self.lift_mode == "reverse_approach":
            u = self.units[unit]
            path = reverse_approach_path(self.steps_plan, u, self.reverse_stride)
            here = np.asarray(self.last_cmd[:7], dtype=float)
            j = int(np.argmin([float(np.max(np.abs(p - here))) for p in path]))
            route += [np.asarray(p) for p in path[j:]]  # carry on up the approach, empty
        else:
            route += [np.asarray(q) for q in applied]  # back up the vertical lift, empty
        start = route[-1] if route else np.asarray(self.last_cmd[:7], dtype=float)
        route += joint_interpolation(start, self.start_joints, self.return_cfg.max_joint_speed_rad_s, self.dt)
        settle = max(1, int(round(self.return_cfg.settle_s / self.dt)))
        route += [self.start_joints] * settle
        if len(route) > self._steps_left():
            return Step.fail(TaskStop.SIM_TIME_LIMIT, "not enough simulated time to return", **info)
        for q in route:
            _, term, trunc = self._step(np.r_[q, 0.0], "return")
            name, moved = disturbed()
            if moved > self.return_cfg.max_object_motion_m and name != target:
                info.update({"disturbed_object": name, "disturbance_m": round(moved, 4)})
                return Step.fail(TaskStop.RECOVERY_FAILED, f"{name} moved {moved:.3f} m on the way back", **info)
            if term or trunc:
                return Step.fail(TaskStop.EPISODE_TIME_LIMIT, "episode ended on the way back", **info)
        err = float(np.max(np.abs(np.asarray(self.world.joint_pos(self.obs)) - self.start_joints)))
        info["final_joint_error_max_rad"] = round(err, 5)
        if err > self.return_cfg.max_final_joint_error_rad:
            return Step.fail(TaskStop.RECOVERY_FAILED, f"observation pose not reached (error {err:.3f} rad)", **info)
        self._event("recovered_to_observation_pose", unit=unit, **info)
        return Step.success(**info)

    def run_to_end(self, attempt: int) -> Step:
        self._unit = len(self.units)

        def step_env(action, tag):
            return self._step(action, "open" if tag.kind == "gripper" else "continue", tag.plan_step)

        res = run_actions(self.client, self.obs, self.instruction, step_env, max(1, self._steps_left()), None,
                          first=self._pending)
        self._pending = None
        if res.ok:
            return Step.success()
        if res.truncated:
            return Step.fail(TaskStop.EPISODE_TIME_LIMIT, res.reason)
        return Step.fail(TaskStop.SIM_TIME_LIMIT if self._steps_left() <= 0 else TaskStop.INVALID_MOTION, res.reason)

    # ---------------------------------------------------------------- scoring
    def evaluate(self) -> tuple[dict, dict]:
        """The baseline's goal check on the final state, and on the state at the baseline's episode end."""
        base = {"goal": self.goal, "target_object": self.target_object, "object_local_bounds": self.bounds}
        final = dict(base, final_object_positions=self.world.object_positions(),
                     final_object_quats=self.world.object_quats())
        open_end = bool(self.last_cmd[-1] <= 0.5)
        at_end = check_goal(self.goal, final, open_end)
        snap = self._snapshot
        if snap is None:  # the trial ended before the baseline's episode length: same state
            at_limit = at_end
        else:
            at_limit = check_goal(self.goal, dict(base, final_object_positions=snap["positions"],
                                                  final_object_quats=snap["quats"]), snap["gripper_open"])
        return at_end, at_limit


def fill_task_record(rec: TaskTrialRecord, result: TaskResult, backend: TaskSimBackend) -> None:
    rec.mode = rec.mode or backend.limits.mode
    rec.lift_mode = rec.lift_mode or backend.lift_mode
    rec.stop_reason, rec.stop_detail = result.stop.value, result.detail
    rec.checkpoints = [{k: (v.value if isinstance(v, Verdict) else v) for k, v in c.items()}
                       for c in result.checkpoints]
    rec.plans = backend.plans
    rec.num_attempts, rec.num_retries, rec.num_planning_calls = result.attempts, result.retries, result.planning_calls
    rec.first_checkpoint_verdict = rec.checkpoints[0]["verdict"] if rec.checkpoints else None
    rec.first_plan_source = backend.plans[0]["source"] if backend.plans else None
    rec.sim_duration_s, rec.wall_duration_s = round(result.sim_s, 3), round(result.wall_s, 3)
    rec.events = result.events + backend.events
    rec.harness_exception = result.harness_exception
    at_end, at_limit = backend.evaluate()
    rec.task_check, rec.task_check_at_baseline_limit = at_end, at_limit
    rec.task_success, rec.task_success_at_baseline_limit = at_end.get("task_success"), at_limit.get("task_success")
    rec.recovered = bool(result.retries > 0 and rec.task_success)
    rec.object_names = backend.names
    rec.object_local_bounds = backend.bounds
    rec.initial_object_positions, rec.initial_object_quats = backend.initial_positions, backend.initial_quats
    rec.final_object_positions = backend.world.object_positions()
    rec.final_object_quats = backend.world.object_quats()
    rec.final_object_centers = {n: v.tolist() for n, v in backend.centers().items()}


# --------------------------------------------------------------------------- CLI / simulator glue


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="One task-level checkpoint-recovery trial")
    p.add_argument("--sim-root", type=Path, default=Path.cwd())
    p.add_argument("--tiptop-dir", type=Path, default=None, help="tiptop checkout, for recording its SHA")
    p.add_argument("--scene", type=int, required=True)
    p.add_argument("--variant", type=int, required=True)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--instruction", required=True)
    p.add_argument("--goal-json", default=None, help="the scene's goal from configs/scenes.json, as JSON")
    p.add_argument("--target-object", default=None, help="resolves $target in the goal (scenes 1-3)")
    p.add_argument("--object-aliases", type=Path, default=PROJECT_ROOT / "configs" / "object_aliases.json")
    p.add_argument("--trial-id", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--headless", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--ws-host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--provenance-file", type=Path, default=None)
    p.add_argument("--gripper-bodies", nargs="+", default=["left_inner_finger", "right_inner_finger"])
    p.add_argument("--ee-body", default="base_link", help="body moved by the vertical IK lift")
    p.add_argument("--probe-label", default=None)

    g = p.add_argument_group("checkpoint")
    g.add_argument("--lift-mode", choices=LIFT_MODES, default="reverse_approach")
    g.add_argument("--reverse-stride", type=float, default=DEFAULT_REVERSE_STRIDE)
    g.add_argument("--ik-speed", type=float, default=DEFAULT_IK_SPEED)
    d = TaskLimits()
    g.add_argument("--max-retries", type=int, default=d.max_retries, help="0 = checkpoint-only control")
    g.add_argument("--max-retries-per-grasp", type=int, default=d.max_retries_per_grasp)
    g.add_argument("--max-planning-calls", type=int, default=d.max_planning_calls)
    g.add_argument("--max-sim-s", type=float, default=d.max_sim_s)
    g.add_argument("--max-wall-s", type=float, default=d.max_wall_s)
    g.add_argument("--baseline-episode-s", type=float, default=d.baseline_episode_s)
    cc = TaskCheckCriteria()
    for f in cc.to_dict():
        g.add_argument("--" + f.replace("_", "-"), type=float, default=getattr(cc, f))

    r = p.add_argument_group("return to the observation pose")
    rc = ReturnConfig()
    r.add_argument("--return-max-joint-speed", type=float, default=rc.max_joint_speed_rad_s)
    r.add_argument("--return-max-object-motion-m", type=float, default=rc.max_object_motion_m)

    pr = p.add_argument_group("pairing / probes (attempt 0)")
    pr.add_argument("--plan-json", type=Path, default=None, help="replay this baseline tiptop_plan.json for attempt 0")
    pr.add_argument("--paired-baseline-run", default=None)
    pr.add_argument("--paired-baseline-dir", default=None)
    pr.add_argument("--displace-dx-m", type=float, default=0.0)
    pr.add_argument("--displace-dy-m", type=float, default=0.0)
    pr.add_argument("--displace-yaw-deg", type=float, default=0.0)
    args, _ = p.parse_known_args(argv)
    return args


def build_config(args) -> tuple[TaskLimits, TaskCheckCriteria, ReturnConfig]:
    limits = TaskLimits(max_retries=args.max_retries, max_retries_per_grasp=args.max_retries_per_grasp,
                        max_planning_calls=args.max_planning_calls, max_sim_s=args.max_sim_s,
                        max_wall_s=args.max_wall_s, baseline_episode_s=args.baseline_episode_s)
    crit = TaskCheckCriteria(**{f: getattr(args, f) for f in TaskCheckCriteria().to_dict()})
    ret = ReturnConfig(max_joint_speed_rad_s=args.return_max_joint_speed,
                       max_object_motion_m=args.return_max_object_motion_m)
    return limits, crit, ret


CARRY_LEFT_HAND_M = 0.05


def left_hand(offset_at_confirm, offset_now, tol_m: float = CARRY_LEFT_HAND_M) -> bool:
    """True if a carried object has left the hand: its distance to the finger reference changed by > tol_m.

    The distance, not the world-frame offset vector: the wrist rotates during transport, which swings the
    world-frame offset of a firmly held object by many centimetres (first GPU smoke: ~14 cm on an angled
    grasp, distance 0.167 -> 0.177 m) while its distance to a point fixed on the hand stays put.
    """
    return abs(float(np.linalg.norm(offset_now)) - float(np.linalg.norm(offset_at_confirm))) > tol_m


def damped_vertical_step(jacobian_6x7: np.ndarray, q_cmd: np.ndarray, dz: float, lower=None, upper=None,
                         damping: float = 0.05, max_step: float = 0.05) -> np.ndarray:
    """One damped least-squares IK step: move the end effector +dz along world z, orientation held."""
    J = np.asarray(jacobian_6x7, dtype=float)
    dx = np.array([0.0, 0.0, float(dz), 0.0, 0.0, 0.0])
    dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), dx)
    dq = np.clip(dq, -max_step, max_step)
    q = np.asarray(q_cmd, dtype=float) + dq
    if lower is not None and upper is not None:
        q = np.clip(q, lower, upper)
    return q


def main(argv=None) -> int:  # pragma: no cover - needs Isaac Sim
    args = parse_args(argv)
    sim_root = args.sim_root.resolve()
    sys.path.insert(0, str(sim_root))
    os.chdir(sim_root)
    from grasp_eval.observed_eval import (_file_sha256, _git_diff_sha256, _git_dirty, _git_sha, _utc_now,
                                          make_observed_client_class, usd_local_bounds)
    from grasp_eval.recovery_eval import _StreamingVideo, make_recovery_client_class

    trial_id = args.trial_id or f"s{args.scene}v{args.variant}t{args.trial}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out = args.out_dir.resolve()
    trial_dir = out / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(trial_dir / "task_recovery_eval.log")
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

    limits, crit, ret_cfg = build_config(args)
    goal = json.loads(args.goal_json) if args.goal_json else None
    aliases = json.loads(args.object_aliases.read_text()) if args.object_aliases and args.object_aliases.exists() else {}
    displacement = ({"dx_m": args.displace_dx_m, "dy_m": args.displace_dy_m, "yaw_deg": args.displace_yaw_deg}
                    if any(v != 0 for v in (args.displace_dx_m, args.displace_dy_m, args.displace_yaw_deg)) else None)
    probe = {"type": "displacement_fixture", "label": args.probe_label, "displacement": displacement} if displacement \
        else {"type": "baseline", "label": args.probe_label}
    episode_length_s = limits.max_sim_s + 10.0
    rec = TaskTrialRecord(
        trial_id=trial_id, scene=args.scene, variant=args.variant, trial=args.trial, instruction=args.instruction,
        goal=goal, target_object=args.target_object, mode=limits.mode, lift_mode=args.lift_mode,
        limits=limits.to_dict(), check_criteria=crit.to_dict(), return_config=ret_cfg.to_dict(), probe=probe,
        seed=args.seed, paired_baseline_run=args.paired_baseline_run, paired_baseline_dir=args.paired_baseline_dir,
        started_utc=_utc_now(), episode_length_s=episode_length_s,
        versions={"project_sha": _git_sha(PROJECT_ROOT), "project_dirty": _git_dirty(PROJECT_ROOT),
                  "sim_sha": _git_sha(sim_root), "sim_dirty": _git_dirty(sim_root),
                  "tiptop_sha": _git_sha(args.tiptop_dir) if args.tiptop_dir else None,
                  "tiptop_diff_sha256": _git_diff_sha256(args.tiptop_dir) if args.tiptop_dir else None,
                  "provenance_file": str(args.provenance_file) if args.provenance_file else None,
                  "provenance_sha256": _file_sha256(args.provenance_file) if args.provenance_file else None})
    env = client = backend = None
    video = _StreamingVideo(trial_dir / "video.mp4", 15)
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
        try:
            bounds, _notes = usd_local_bounds(scene, names)
        except Exception:
            bounds = {}
        body_ids, _ = robot.find_bodies(args.gripper_bodies, preserve_order=True)
        ee_ids, _ = robot.find_bodies([args.ee_body])
        ee_jac_index = ee_ids[0] - 1 if robot.is_fixed_base else ee_ids[0]
        limits_t = robot.data.soft_joint_pos_limits[0, :7].cpu().numpy()

        class IsaacTaskWorld:
            step_dt = float(env.unwrapped.step_dt)
            object_names = names
            object_local_bounds = bounds

            def step(self, action):
                o, _, term, trunc, _ = env.step(torch.tensor(action, dtype=torch.float32)[None])
                return o, bool(term), bool(trunc)

            def object_positions(self):
                return {n: scene.rigid_objects[n].data.root_pos_w[0].tolist() for n in names}

            def object_quats(self):
                return {n: scene.rigid_objects[n].data.root_quat_w[0].tolist() for n in names}

            def gripper_ref(self):
                return robot.data.body_pos_w[0, body_ids].mean(dim=0).tolist()

            def joint_pos(self, o):
                return o["policy"]["arm_joint_pos"][0].cpu().numpy()

            def gripper_obs(self, o):
                return float(o["policy"]["gripper_pos"][0, 0])

            def vertical_lift_step(self, q_cmd, dz):
                jac = robot.root_physx_view.get_jacobians()[0, ee_jac_index, :, :7].cpu().numpy()
                return damped_vertical_step(jac, q_cmd, dz, limits_t[:, 0], limits_t[:, 1])

            def set_object_pose(self, name, pos, quat):
                obj = scene[name]
                obj.write_root_pose_to_sim(torch.tensor([[*pos, *quat]], dtype=torch.float32, device=device))
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))

        world = IsaacTaskWorld()
        caption_ok = [True]

        def on_step(o, action, phase, attempt, plan_step):
            try:
                pol = o["policy"]
                viz = np.concatenate([pol["external_cam"][0].cpu().numpy(), pol["wrist_cam"][0].cpu().numpy()], axis=1)
                viz = add_top_padding(viz, pad_px=40)
                overlay_timer_ms(viz, int(backend.steps * 1000 / 15))
                if caption_ok[0]:
                    step = "" if plan_step is None or plan_step < 0 else f" step {plan_step + 1}"
                    viz = add_caption(viz, f"attempt {attempt + 1} | grasp {backend._unit + 1} | {phase}{step} | "
                                           f"gripper {'CLOSE' if action[-1] > 0.5 else 'open'} | {args.lift_mode}")
                video.add(viz)
            except Exception:
                caption_ok[0] = False
                logger.warning("video frame skipped:\n%s", traceback.format_exc())

        ClientCls = make_recovery_client_class(make_observed_client_class(TiptopWebsocketClient))
        if args.plan_json:
            ClientCls.replay_plan_path = args.plan_json.resolve()
        with torch.no_grad():
            obs, _ = env.reset()
            obs, _ = env.reset()
            obs, _ = env.reset()
            obs = settle_sim(env, obs, reset_episode_buf=True)
            client = ClientCls(host=args.ws_host, port=args.ws_port)
            backend = TaskSimBackend(world, client, obs, instruction=args.instruction, lift_mode=args.lift_mode,
                                     limits=limits, criteria=crit, return_cfg=ret_cfg, goal=goal,
                                     target_object=args.target_object, aliases=aliases,
                                     replay_plan=str(args.plan_json.resolve()) if args.plan_json else None,
                                     displacement=displacement, reverse_stride=args.reverse_stride,
                                     ik_speed=args.ik_speed, on_step=on_step)
            rec.process_startup_s = round(time.monotonic() - PROCESS_T0, 2)
            result = TaskSupervisor(backend, limits).run()
        fill_task_record(rec, result, backend)
    except Exception:
        rec.harness_exception = traceback.format_exc()
        rec.stop_reason = rec.stop_reason or TaskStop.INFRASTRUCTURE.value
        logger.error("Harness exception:\n%s", rec.harness_exception)
    finally:
        try:
            video.close()
            if video.frames:
                rec.artifacts["video"] = str(video.path)
            if backend is not None and backend.trace["t"]:
                np.savez_compressed(trial_dir / "trace.npz", object_names=np.asarray(backend.names),
                                    phase_names=np.asarray(PHASES),
                                    **{k: np.asarray(v) for k, v in backend.trace.items()})
                rec.artifacts["trace"] = str(trial_dir / "trace.npz")
        except Exception:
            rec.artifacts["artifact_error"] = traceback.format_exc()
        rec.artifacts["log"] = str(trial_dir / "task_recovery_eval.log")
        rec.finished_utc = _utc_now()
        (trial_dir / "record.json").write_text(json.dumps(rec.to_dict(), indent=2, sort_keys=True, default=str))
        append_jsonl(out / TASK_TRIALS_FILE, json.loads(json.dumps(rec.to_dict(), default=str)))
        logger.info("Trial %s: stop=%s task_success=%s retries=%s", trial_id, rec.stop_reason, rec.task_success,
                    rec.num_retries)
        try:
            if client is not None:
                client.close()
            if env is not None:
                env.close()
        except Exception:
            pass
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
