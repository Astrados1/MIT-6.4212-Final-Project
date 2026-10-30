"""Stable-lift scoring from a logged trace. Reads states only; never touches the sim.

The scored segment is the first grasp: from the first commanded close until the
next commanded open (the plan's intentional release), or the end of the trace.
A sample "holds" when the target is at least `min_rise_m` above its settled
starting height, within `max_obj_to_gripper_m` of the gripper reference point,
and the gripper is commanded closed. The grasp succeeds when holding samples
span at least `min_hold_s` without a break.

The default thresholds are pilot values. Validate them against native motion
and object geometry in pilot runs, then freeze them before any comparison
(see docs/scoring.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LiftCriteria:
    min_rise_m: float = 0.05
    min_hold_s: float = 0.5
    max_obj_to_gripper_m: float = 0.10
    # Commanded gripper values above this are treated as "close" (matches the
    # simulator's BinaryJointPositionZeroToOneAction threshold).
    close_threshold: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraspScore:
    lifted: bool
    num_close_commands: int
    first_close_t: float | None
    release_t: float | None
    max_rise_m: float | None
    max_rise_near_gripper_m: float | None
    longest_hold_s: float
    # Suggested failure label from the trace alone. Confirm against the video.
    failure_hint: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def count_close_commands(gripper_cmd: np.ndarray, close_threshold: float = 0.5) -> int:
    """Number of open->close transitions in the commanded gripper signal."""
    closed = np.asarray(gripper_cmd) > close_threshold
    if closed.size == 0:
        return 0
    return int(closed[0]) + int(np.count_nonzero(closed[1:] & ~closed[:-1]))


def _longest_true_run_duration(mask: np.ndarray, t: np.ndarray) -> float:
    """Longest time span t[j] - t[i] over runs of consecutive True samples."""
    best = 0.0
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            end = i if m else i - 1
            best = max(best, float(t[end] - t[start]))
            start = None
    return best


def score_first_grasp(
    t: np.ndarray,
    obj_pos: np.ndarray,
    gripper_pos: np.ndarray,
    gripper_cmd: np.ndarray,
    baseline_z: float,
    criteria: LiftCriteria = LiftCriteria(),
) -> GraspScore:
    """Score the first grasp/lift segment of one episode.

    Args:
        t: (N,) sample times in seconds, increasing.
        obj_pos: (N, 3) target object position in the world frame.
        gripper_pos: (N, 3) gripper reference point (finger midpoint) in the world frame.
        gripper_cmd: (N,) commanded gripper value that produced each sample.
        baseline_z: target height after settling, before any plan action.
    """
    t = np.asarray(t, dtype=float)
    obj_pos = np.asarray(obj_pos, dtype=float).reshape(-1, 3)
    gripper_pos = np.asarray(gripper_pos, dtype=float).reshape(-1, 3)
    closed = np.asarray(gripper_cmd, dtype=float) > criteria.close_threshold
    n = len(t)
    if not (len(obj_pos) == len(gripper_pos) == len(closed) == n):
        raise ValueError("t, obj_pos, gripper_pos and gripper_cmd must have the same length")

    num_closes = count_close_commands(gripper_cmd, criteria.close_threshold)
    max_rise = float(np.max(obj_pos[:, 2] - baseline_z)) if n else None

    close_idx = np.flatnonzero(closed)
    if close_idx.size == 0:
        return GraspScore(False, num_closes, None, None, max_rise, None, 0.0, None)

    c0 = int(close_idx[0])
    reopen = np.flatnonzero(~closed[c0:])
    end = c0 + int(reopen[0]) if reopen.size else n
    release_t = float(t[end]) if end < n else None

    seg = slice(c0, end)
    rise = obj_pos[seg, 2] - baseline_z
    dist = np.linalg.norm(obj_pos[seg] - gripper_pos[seg], axis=1)
    near = dist <= criteria.max_obj_to_gripper_m
    holding = near & (rise >= criteria.min_rise_m)
    longest = _longest_true_run_duration(holding, t[seg])
    lifted = longest >= criteria.min_hold_s
    max_rise_near = float(np.max(rise[near])) if near.any() else None

    if lifted:
        hint = None
    elif not holding.any():
        hint = "missed_grasp"
    else:
        hint = "drop_after_closure_or_lift"

    return GraspScore(
        lifted=lifted,
        num_close_commands=num_closes,
        first_close_t=float(t[c0]),
        release_t=release_t,
        max_rise_m=max_rise,
        max_rise_near_gripper_m=max_rise_near,
        longest_hold_s=longest,
        failure_hint=hint,
    )
