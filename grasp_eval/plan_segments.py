"""Find and execute the grasp segment of a TiPToP plan: approach, close, post-close retraction.

Supported structure (cuTAMP 0.0.6 motion_solver.py, TiPToP d8f5afd):

    Pick(X, ...)  trajectory  x2 (from q0: approach, approach->grasp) or x3 (retract first)
    Pick(X, ...)  gripper close
    Place(X, ...) trajectory  <- retraction: plan_single to world_from_ee @ approach_offset
    Place(X, ...) trajectory     (transfer to approach)          -- never executed here
    Place(X, ...) trajectory     (final approach)                -- never executed here
    Place(X, ...) gripper open
    GoToInitial   trajectory x2

The post-grasp lift is the *first Place-labelled trajectory*, so filtering on Pick labels
misses it. A pick-only plan ends Pick -> close -> GoToInitial, whose first trajectory is the
same approach_offset retraction; that is supported too. `approach_offset` is -5 cm along the
end-effector z axis, i.e. straight up for a top-down grasp. Verify on video / FK that it is.

Anything else (an earlier Pick of another object, a Place before the close, two closes
before the retraction, an unknown operator) is refused rather than guessed: executing the
wrong boundary would carry the object into transport.

The step indices here are into the client's `_plan` list (the server steps with metadata
removed), which is how the upstream client indexes them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

_LABEL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")
RETRACT_OPERATORS = {"Place": "place_retract", "PlaceNear": "place_retract", "GoToInitial": "go_to_initial_retract"}


class UnsupportedPlanError(ValueError):
    """The plan does not have the single-target structure this milestone supports."""


def parse_label(label: str | None) -> tuple[str | None, list[str]]:
    """'Pick(rubiks_cube, grasp1, q1)' -> ('Pick', ['rubiks_cube', 'grasp1', 'q1'])."""
    m = _LABEL_RE.match(label or "")
    if not m:
        return None, []
    args = [a.strip() for a in m.group(2).split(",")] if m.group(2).strip() else []
    return m.group(1), args


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True)
class GraspSegment:
    target_label: str  # object argument of the Pick, as TiPToP perceived it
    pick_label: str
    pick_traj_indices: tuple[int, ...]
    close_index: int
    retraction_index: int
    retraction_label: str
    retraction_source: str  # place_retract | go_to_initial_retract
    retraction_num_waypoints: int
    retraction_dt: float | None
    num_plan_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_grasp_segment(steps: list[dict], target_label: str | None = None) -> GraspSegment:
    """Locate the target grasp and its retraction in `steps` (metadata already removed).

    `target_label` (optional) is the perceived name we expect TiPToP to pick. If given, the
    plan's Pick object must match it (case/punctuation-insensitive).
    """
    if not steps:
        raise UnsupportedPlanError("empty plan")
    closes = [i for i, s in enumerate(steps) if s.get("type") == "gripper" and s.get("action") == "close"]
    if not closes:
        raise UnsupportedPlanError("plan has no gripper close")
    if len(closes) > 1:
        raise UnsupportedPlanError(f"plan has {len(closes)} gripper closes; only single-grasp plans are supported")
    picked = {args[0] for op, args in (parse_label(s.get("label")) for s in steps) if op == "Pick" and args}
    if len(picked) > 1:
        raise UnsupportedPlanError(f"plan picks several objects {sorted(picked)}; multi-object plans are not supported")
    ci = closes[0]
    close_op, close_args = parse_label(steps[ci].get("label"))
    if close_op != "Pick" or not close_args:
        raise UnsupportedPlanError(f"first close is not labelled Pick(...): {steps[ci].get('label')!r}")
    obj = close_args[0]

    pre = steps[:ci]
    if not pre:
        raise UnsupportedPlanError("no approach trajectory before the close")
    for i, s in enumerate(pre):
        op, args = parse_label(s.get("label"))
        if s.get("type") != "trajectory":
            raise UnsupportedPlanError(f"step {i} before the close is a {s.get('type')!r}, not a trajectory")
        if op != "Pick" or not args or args[0] != obj:
            raise UnsupportedPlanError(
                f"step {i} before the target close is {s.get('label')!r}; only Pick({obj}, ...) trajectories "
                "are supported there (an earlier pick/place of another object is a multi-object plan)")
        if s.get("label") != steps[ci].get("label"):
            raise UnsupportedPlanError(f"step {i} label {s.get('label')!r} differs from the close's label")
    if len(pre) > 3:
        raise UnsupportedPlanError(f"{len(pre)} approach trajectories before the close; cuTAMP 0.0.6 emits 2 or 3")

    if target_label is not None and _norm(obj) != _norm(target_label):
        raise UnsupportedPlanError(
            f"plan picks {obj!r} first, expected target {target_label!r} (auxiliary move or grounding error)")

    ri = ci + 1
    if ri >= len(steps):
        raise UnsupportedPlanError("plan ends at the close; no retraction to execute")
    r = steps[ri]
    r_op, r_args = parse_label(r.get("label"))
    if r.get("type") != "trajectory" or r_op not in RETRACT_OPERATORS:
        raise UnsupportedPlanError(
            f"step after the close is {r.get('type')!r} {r.get('label')!r}; expected a Place/PlaceNear/GoToInitial "
            "trajectory (the cuTAMP retraction)")
    if r_op in ("Place", "PlaceNear") and (not r_args or r_args[0] != obj):
        raise UnsupportedPlanError(f"retraction {r.get('label')!r} does not move the picked object {obj!r}")
    positions = r.get("positions")
    n_wp = int(len(positions)) if positions is not None else 0
    if n_wp == 0:
        raise UnsupportedPlanError("retraction trajectory has no waypoints")
    return GraspSegment(
        target_label=obj, pick_label=steps[ci].get("label"), pick_traj_indices=tuple(range(ci)),
        close_index=ci, retraction_index=ri, retraction_label=r.get("label"),
        retraction_source=RETRACT_OPERATORS[r_op], retraction_num_waypoints=n_wp,
        retraction_dt=float(r["dt"]) if r.get("dt") is not None else None, num_plan_steps=len(steps),
    )


# --------------------------------------------------------------------------- execution boundary


@dataclass(frozen=True)
class ClientCursor:
    """Snapshot of the upstream client's execution bookkeeping (TiptopWebsocketClient)."""

    plan_step: int
    waypoint_idx: int
    traj_id: int | None
    traj_len: int | None
    gripper_pending: str | None

    @classmethod
    def of(cls, client) -> "ClientCursor":
        traj = getattr(client, "_current_trajectory", None)
        return cls(
            plan_step=int(client._current_plan_step),
            waypoint_idx=int(client._current_waypoint_idx),
            traj_id=id(traj) if traj is not None else None,
            traj_len=len(traj) if traj is not None else None,
            gripper_pending=client._gripper_action_pending,
        )


@dataclass(frozen=True)
class ActionTag:
    kind: str  # "waypoint" | "gripper" | "hold" (plan finished, holding measured state)
    plan_step: int | None  # index into client._plan of the step this action belongs to
    last_of_step: bool  # waypoint actions only: the final waypoint of its trajectory


def tag_action(before: ClientCursor, after: ClientCursor) -> ActionTag:
    """Which plan step produced the action that `infer` just returned.

    The upstream client advances its bookkeeping when it *returns* a command, before the
    command reaches env.step: `_current_plan_step` already points past a trajectory once its
    first waypoint is returned, and `_current_waypoint_idx` is incremented before the step.
    So `plan_done` can be True while the final waypoint is still unapplied (the week-one loop
    and upstream tiptop_eval.py check plan_done before env.step and drop that waypoint).
    """
    if after.gripper_pending is not None:
        return ActionTag("gripper", after.plan_step, False)
    advanced = (after.traj_id != before.traj_id or after.waypoint_idx != before.waypoint_idx)
    if after.traj_id is not None and advanced:
        return ActionTag("waypoint", after.plan_step - 1, after.waypoint_idx >= (after.traj_len or 0))
    return ActionTag("hold", None, False)


@dataclass
class SegmentResult:
    ok: bool
    reason: str
    actions_applied: int
    retraction_applied: bool  # the final retraction waypoint went through env.step
    final_joint_cmd: list[float] | None
    final_gripper_cmd: float | None
    tags: list[dict[str, Any]] = field(default_factory=list)  # per applied action, compact
    obs: Any = None  # observation after the last applied step
    terminated: bool = False
    truncated: bool = False


def run_segment(
    client,
    obs: Any,
    instruction: str,
    segment: GraspSegment,
    step_env: Callable[[np.ndarray, ActionTag], tuple[Any, bool, bool]],
    max_actions: int,
    first: tuple[dict, ActionTag] | None = None,
) -> SegmentResult:
    """Apply plan actions through env.step until the final retraction waypoint has been applied.

    `step_env(action, tag) -> (obs, terminated, truncated)` must call env.step exactly once.
    `first` is an action already returned by `infer` (the call that fetched the plan) and not
    yet applied. Never consults `plan_done`. Refuses to apply any action past the retraction.
    """
    applied = 0
    tags: list[dict[str, Any]] = []
    last_cmd = None
    pending = first
    while applied < max_actions:
        if pending is not None:
            ret, tag = pending
            pending = None
        else:
            before = ClientCursor.of(client)
            ret = client.infer(obs, instruction)
            tag = tag_action(before, ClientCursor.of(client))
        if tag.kind == "hold" or (tag.plan_step is not None and tag.plan_step > segment.retraction_index):
            return SegmentResult(False, f"client moved past the retraction (tag {tag})", applied, False,
                                 last_cmd, None, tags, obs)
        if tag.kind == "gripper" and tag.plan_step != segment.close_index:
            return SegmentResult(False, f"unexpected gripper step {tag.plan_step} inside the grasp segment",
                                 applied, False, last_cmd, None, tags, obs)
        action = np.asarray(ret["action"], dtype=float)
        obs, term, trunc = step_env(action, tag)
        applied += 1
        last_cmd = action
        tags.append({"k": tag.kind[0], "s": tag.plan_step, "l": tag.last_of_step})
        done = tag.kind == "waypoint" and tag.plan_step == segment.retraction_index and tag.last_of_step
        if done:
            return SegmentResult(True, "retraction_applied", applied, True, [float(v) for v in action[:7]],
                                 float(action[-1]), tags, obs, bool(term), bool(trunc))
        if term or trunc:
            return SegmentResult(False, "episode ended before the retraction finished", applied, False,
                                 [float(v) for v in action[:7]], float(action[-1]), tags, obs, bool(term), bool(trunc))
    return SegmentResult(False, f"retraction not reached within {max_actions} actions", applied, False,
                         None if last_cmd is None else [float(v) for v in last_cmd[:7]],
                         None if last_cmd is None else float(last_cmd[-1]), tags, obs)


def clear_client_plan(client) -> None:
    """Drop the remaining plan and buffered waypoints without reconnecting (state as after __init__)."""
    client._plan = None
    client._current_plan_step = 0
    client._current_trajectory = None
    client._current_waypoint_idx = 0
    client._gripper_action_pending = None
    client._gripper_action_steps_remaining = 0
    client._action_chunk_done = False
    client._last_planning_time = None


# --------------------------------------------------------------------------- task level: every grasp in a plan
#
# Full checkpoint recovery checks the hold after *each* grasp of a (possibly multi-object) plan and
# otherwise executes the plan exactly as upstream does. A cuTAMP 0.0.6 plan is a sequence of
#
#     Pick(X, ...) trajectory x2 or x3      approach (the 3rd is the retreat from the previous place)
#     Pick(X, ...) gripper close
#     Place(X, ...) trajectory              <- retraction (approach_offset, 5 cm): the checkpoint follows it
#     Place(X, ...) trajectory x2           transfer, final approach
#     Place(X, ...) gripper open
#
# blocks, ending with GoToInitial trajectories (a pick-only block retracts with GoToInitial).
# Verified on real scene-1/3/5 plans (tests/data/plans).


@dataclass(frozen=True)
class GraspUnit:
    """One grasp of a plan: approach steps, the close, and its retraction (the checkpoint boundary)."""

    index: int  # 0-based order of this grasp in the plan
    target_label: str  # object argument of the Pick, as TiPToP perceived it
    pick_label: str
    first_step: int  # first approach trajectory of this grasp
    close_index: int
    retraction_index: int
    retraction_label: str
    retraction_source: str
    retraction_num_waypoints: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_grasp_units(steps: list[dict]) -> list[GraspUnit]:
    """Every grasp in `steps` (metadata removed), in execution order.

    Raises UnsupportedPlanError for a structure it does not recognise; task-level recovery then
    executes the plan unchanged, without checkpoints, rather than guessing a boundary.
    """
    if not steps:
        raise UnsupportedPlanError("empty plan")
    closes = [i for i, s in enumerate(steps) if s.get("type") == "gripper" and s.get("action") == "close"]
    if not closes:
        raise UnsupportedPlanError("plan has no gripper close")
    units: list[GraspUnit] = []
    prev_end = 0
    for n, ci in enumerate(closes):
        label = steps[ci].get("label")
        op, args = parse_label(label)
        if op != "Pick" or not args:
            raise UnsupportedPlanError(f"close {n} at step {ci} is not labelled Pick(...): {label!r}")
        obj = args[0]
        j = ci - 1
        while (j >= prev_end and steps[j].get("type") == "trajectory" and steps[j].get("label") == label):
            j -= 1
        first = j + 1
        if first == ci:
            raise UnsupportedPlanError(f"close {n} ({label!r}) has no Pick({obj}, ...) approach trajectory before it")
        if ci - first > 3:
            raise UnsupportedPlanError(f"close {n}: {ci - first} approach trajectories; cuTAMP 0.0.6 emits 2 or 3")
        ri = ci + 1
        if ri >= len(steps):
            raise UnsupportedPlanError(f"close {n} ends the plan; no retraction")
        r = steps[ri]
        r_op, r_args = parse_label(r.get("label"))
        if r.get("type") != "trajectory" or r_op not in RETRACT_OPERATORS:
            raise UnsupportedPlanError(f"step after close {n} is {r.get('type')!r} {r.get('label')!r}; expected the "
                                       "Place/PlaceNear/GoToInitial retraction")
        if r_op in ("Place", "PlaceNear") and (not r_args or r_args[0] != obj):
            raise UnsupportedPlanError(f"retraction {r.get('label')!r} does not move the picked object {obj!r}")
        positions = r.get("positions")
        n_wp = int(len(positions)) if positions is not None else 0
        if n_wp == 0:
            raise UnsupportedPlanError(f"retraction after close {n} has no waypoints")
        units.append(GraspUnit(index=n, target_label=obj, pick_label=label, first_step=first, close_index=ci,
                               retraction_index=ri, retraction_label=r.get("label"),
                               retraction_source=RETRACT_OPERATORS[r_op], retraction_num_waypoints=n_wp))
        prev_end = ri + 1
    return units


def run_actions(
    client,
    obs: Any,
    instruction: str,
    step_env: Callable[[np.ndarray, ActionTag], tuple[Any, bool, bool]],
    max_actions: int,
    stop_after_step: int | None,
    first: tuple[dict, ActionTag] | None = None,
) -> SegmentResult:
    """Execute the client's plan from its current position.

    stop_after_step=k: stop once step k is complete. For a trajectory step that is its final
    waypoint (a unit's retraction: the vertical_ik checkpoint boundary); for a gripper step the last
    tick of the client's gripper dwell (a unit's close: the reverse_approach boundary). Refuses to
    apply anything past step k.
    stop_after_step=None: run to the end exactly like the week-one/upstream loop, which checks
    `plan_done` after `infer` and before `env.step` (so the final waypoint is not applied there either).
    """
    applied = 0
    tags: list[dict[str, Any]] = []
    last_cmd = None
    pending = first

    def result(ok, reason, term=False, trunc=False, done=False):
        cmd = None if last_cmd is None else [float(v) for v in last_cmd[:7]]
        grip = None if last_cmd is None else float(last_cmd[-1])
        return SegmentResult(ok, reason, applied, done, cmd, grip, tags, obs, bool(term), bool(trunc))

    while applied < max_actions:
        if pending is not None:
            ret, tag = pending
            pending = None
        else:
            before = ClientCursor.of(client)
            ret = client.infer(obs, instruction)
            tag = tag_action(before, ClientCursor.of(client))
        if stop_after_step is None and client.plan_done:
            return result(True, "plan_done", done=True)
        if stop_after_step is not None:
            if tag.kind == "hold":
                return result(False, f"plan ended before step {stop_after_step} finished")
            if tag.plan_step is not None and tag.plan_step > stop_after_step:
                return result(False, f"client moved past step {stop_after_step} (tag {tag})")
        action = np.asarray(ret["action"], dtype=float)
        obs, term, trunc = step_env(action, tag)
        applied += 1
        last_cmd = action
        tags.append({"k": tag.kind[0], "s": tag.plan_step, "l": tag.last_of_step})
        if stop_after_step is not None and tag.plan_step == stop_after_step and (
                (tag.kind == "waypoint" and tag.last_of_step)
                or (tag.kind == "gripper" and int(getattr(client, "_gripper_action_steps_remaining", 0)) == 0)):
            return result(True, "boundary_applied", term, trunc, done=True)
        if term or trunc:
            return result(False, "episode ended", term, trunc)
    return result(False, f"limit of {max_actions} actions reached")


def skip_client_to(client, plan_step: int, gripper_state: float) -> None:
    """Point the upstream client at the start of `plan_step`, as if every earlier step had run.

    Used after a reverse_approach check: the arm skipped the plan's own retraction (it lifted along
    the reversed approach instead and came back down to the retraction end), so execution resumes
    at the step after the retraction with the gripper still closed.
    """
    client._current_plan_step = int(plan_step)
    client._current_trajectory = None
    client._current_waypoint_idx = 0
    client._gripper_action_pending = None
    client._gripper_action_steps_remaining = 0
    client._last_gripper_state = float(gripper_state)
    client._action_chunk_done = False


def reverse_approach_path(steps: list[dict], unit: GraspUnit, stride: float) -> np.ndarray:
    """Joint waypoints retracing `unit`'s approach backwards, from the grasp toward its start pose.

    `stride` is plan waypoints per simulation step: the upstream client uses 3 (dt 0.02 s at 15 Hz),
    so stride 1.5 retraces at half speed. Linear interpolation between plan waypoints; the first
    returned waypoint is one stride above the grasp, the last is the approach's first waypoint.
    """
    traj = [np.asarray(steps[i]["positions"], dtype=float) for i in range(unit.first_step, unit.close_index)]
    path = np.concatenate(traj, axis=0)[::-1]
    if stride <= 0:
        raise ValueError("stride must be positive")
    pos = np.arange(stride, len(path) - 1 + 1e-9, stride)
    if len(pos) == 0 or pos[-1] < len(path) - 1:
        pos = np.append(pos, len(path) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(path) - 1)
    frac = (pos - lo)[:, None]
    return path[lo] * (1 - frac) + path[hi] * frac
