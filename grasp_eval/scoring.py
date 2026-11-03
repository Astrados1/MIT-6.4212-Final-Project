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


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Express world vector v in the frame whose orientation is q (w, x, y, z)."""
    qv = np.concatenate([[0.0], v])
    return _quat_mul(_quat_mul(_quat_conj(q), qv), q)[1:]


def in_hand_motion(
    t: np.ndarray,
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    gripper_pos: np.ndarray,
    gripper_quat: np.ndarray,
    gripper_cmd: np.ndarray,
    baseline_z: float,
    criteria: LiftCriteria = LiftCriteria(),
) -> dict:
    """How much the target moved relative to the hand while carried (rigid-grasp check).

    Covers the first grasp from the first holding sample (lifted and near the closed
    gripper, as in score_first_grasp) until the sample before the next open command.
    The object pose is expressed in the gripper frame; shift and rotation are measured
    against the pose at the first holding sample. A drop is a later sample, still
    commanded closed, where the object is farther than max_obj_to_gripper_m from the hand.
    """
    t = np.asarray(t, dtype=float)
    obj_pos = np.asarray(obj_pos, dtype=float).reshape(-1, 3)
    obj_quat = np.asarray(obj_quat, dtype=float).reshape(-1, 4)
    gripper_pos = np.asarray(gripper_pos, dtype=float).reshape(-1, 3)
    gripper_quat = np.asarray(gripper_quat, dtype=float).reshape(-1, 4)
    closed = np.asarray(gripper_cmd, dtype=float) > criteria.close_threshold
    close_idx = np.flatnonzero(closed)
    if close_idx.size == 0:
        return {"carried": False}
    c0 = int(close_idx[0])
    reopen = np.flatnonzero(~closed[c0:])
    end = c0 + int(reopen[0]) if reopen.size else len(t)
    dist = np.linalg.norm(obj_pos - gripper_pos, axis=1)
    holding = (dist <= criteria.max_obj_to_gripper_m) & (obj_pos[:, 2] - baseline_z >= criteria.min_rise_m)
    held = np.flatnonzero(holding[c0:end])
    if held.size == 0:
        return {"carried": False}
    h0 = c0 + int(held[0])

    def rel(i: int) -> tuple[np.ndarray, np.ndarray]:
        q_rel = _quat_mul(_quat_conj(gripper_quat[i]), obj_quat[i])
        return _quat_rotate_inv(gripper_quat[i], obj_pos[i] - gripper_pos[i]), q_rel / np.linalg.norm(q_rel)

    p0, q0 = rel(h0)
    shifts, rots = [], []
    for i in range(h0, end):
        p, q = rel(i)
        shifts.append(float(np.linalg.norm(p - p0)))
        rots.append(float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q, q0))))))))
    far = np.flatnonzero(dist[h0:end] > criteria.max_obj_to_gripper_m)
    return {
        "carried": True,
        "carry_start_t": float(t[h0]),
        "carry_end_t": float(t[end - 1]),
        "released_by_plan": end < len(t),
        "max_in_hand_shift_m": round(max(shifts), 4),
        "max_in_hand_rot_deg": round(max(rots), 2),
        "in_hand_shift_at_release_m": round(shifts[-1], 4),
        "in_hand_rot_at_release_deg": round(rots[-1], 2),
        "dropped_before_release": bool(far.size),
        "drop_t": float(t[h0 + int(far[0])]) if far.size else None,
    }
