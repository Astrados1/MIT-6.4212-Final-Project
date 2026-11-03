"""Stage attribution: where along perception -> grounding -> grasp -> execution a run went wrong.

Combines, after the fact and without touching the simulator:
  * the run record and trace written by observed_eval.py (simulator ground truth),
  * the scene TiPToP perceived (extract_perceived_scene.py output),
  * the executed plan (tiptop_plan.json).

Every metric is a measurement. `stage_hint` is a heuristic from fixed thresholds
(ATTRIBUTION_THRESHOLDS) to guide video review, not a label.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Simulator client lowers the camera pose by this much before sending it to TiPToP
# (droid-sim-evals tiptop_websocket.py), so TiPToP's world z is shifted down by it.
CLIENT_CAM_Z_OFFSET_M = 0.015
CLIENT_ASSUMED_PLAN_HZ = 50.0
SIM_CONTROL_HZ = 15.0

ATTRIBUTION_THRESHOLDS = {
    "perception_xy_error_m": 0.02,
    "gripper_xy_offset_at_close_m": 0.02,
    "joint_tracking_error_rad": 0.05,
    "pushed_before_close_m": 0.01,
    "missed_detection_radius_m": 0.05,
}

_PICK_RE = re.compile(r"^\s*Pick\(\s*([^,()]+?)\s*,")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def pick_objects(plan: dict | None) -> list[str]:
    """Object names of Pick actions, in plan order (deduplicated, as labels repeat per step)."""
    names: list[str] = []
    for step in (plan or {}).get("steps", []):
        m = _PICK_RE.match(step.get("label") or "")
        if m and (not names or names[-1] != m.group(1)):
            names.append(m.group(1))
    return names


def plan_timing(plan: dict | None) -> dict[str, Any]:
    dts = sorted({round(float(s["dt"]), 6) for s in (plan or {}).get("steps", [])
                  if s.get("type") == "trajectory" and s.get("dt")})
    if not dts:
        return {"trajectory_dts_s": []}
    stride = max(1, int(round(CLIENT_ASSUMED_PLAN_HZ / SIM_CONTROL_HZ)))
    return {
        "trajectory_dts_s": dts,
        "implied_plan_hz": [round(1.0 / d, 3) for d in dts],
        "client_assumed_plan_hz": CLIENT_ASSUMED_PLAN_HZ,
        "client_waypoint_stride": stride,
        # executed duration / planned duration for each dt (1.0 = executed at planned speed)
        "execution_time_scale": [round((1.0 / SIM_CONTROL_HZ) / (stride * d), 3) for d in dts],
    }


def _perceived_center(obj: dict) -> np.ndarray | None:
    c = obj.get("aabb_center")
    if c is None and obj.get("pose"):
        c = obj["pose"][:3]
    if c is None:
        return None
    c = np.asarray(c, dtype=float).copy()
    c[2] += CLIENT_CAM_Z_OFFSET_M
    return c


def _nearest(point: np.ndarray, gt: dict[str, list[float]]) -> tuple[str | None, float | None]:
    best, best_d = None, None
    for name, pos in gt.items():
        d = float(np.linalg.norm(np.asarray(pos[:2]) - point[:2]))
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best, best_d


def perception_and_grounding(perceived: dict | None, plan: dict | None, gt_initial: dict, target: str | None):
    out: dict[str, Any] = {}
    if not perceived or "objects" not in perceived:
        out["perceived_scene_available"] = False
        if perceived and perceived.get("error"):
            out["perceived_scene_error"] = perceived["error"]
        return out
    out["perceived_scene_available"] = True
    movables = [o for o in perceived["objects"] if o.get("role") == "movable"]
    out["perceived_movables"] = [o["name"] for o in movables]

    per_object = []
    matched_gt = set()
    for o in movables:
        c = _perceived_center(o)
        if c is None:
            continue
        gt_name, d = _nearest(c, gt_initial)
        if gt_name is None:
            continue
        gt = np.asarray(gt_initial[gt_name], dtype=float)
        per_object.append({"perceived": o["name"], "nearest_sim_object": gt_name,
                           "xy_error_m": round(d, 4), "z_error_m": round(float(c[2] - gt[2]), 4)})
        if d <= ATTRIBUTION_THRESHOLDS["missed_detection_radius_m"]:
            matched_gt.add(gt_name)
    out["perceived_objects"] = per_object
    out["sim_objects_without_nearby_detection"] = sorted(set(gt_initial) - matched_gt)

    picks = pick_objects(plan)
    out["plan_pick_objects"] = picks
    if picks:
        first = picks[0]
        match = next((p for p in per_object if p["perceived"] == first), None) or next(
            (p for p in per_object if normalize_name(p["perceived"]) == normalize_name(first)), None)
        out["first_pick_found_in_perceived_scene"] = match is not None
        if match:
            out["first_pick_maps_to_sim_object"] = match["nearest_sim_object"]
            out["perception_xy_error_m"] = match["xy_error_m"]
            out["perception_z_error_m"] = match["z_error_m"]
            if target:
                out["first_pick_matches_target"] = match["nearest_sim_object"] == target
    return out


def execution(record: dict, trace: dict | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not trace or len(trace.get("t", [])) == 0:
        return out
    names = [str(n) for n in trace["object_names"]]
    target = record.get("target_object")
    if target not in names:
        return out
    k = names.index(target)
    obj = np.asarray(trace["objects_pos"])[:, k, :]
    grip = np.asarray(trace["gripper_pos"])
    cmd = np.asarray(trace["gripper_cmd"]) > 0.5
    q_meas = np.asarray(trace["joint_pos"])
    q_cmd = np.asarray(trace["joint_cmd"])
    err = np.abs(q_cmd - q_meas).max(axis=1) if q_cmd.shape == q_meas.shape else None
    if err is not None:
        out["max_joint_tracking_error_rad"] = round(float(err.max()), 4)

    closes = np.flatnonzero(cmd)
    if closes.size == 0:
        return out
    c = int(closes[0])
    # Settled pose that the close was aiming at: the sample just before the close command.
    i = max(c - 1, 0)
    offset = grip[i] - obj[i]
    out["first_close_t_s"] = round(float(np.asarray(trace["t"])[c]), 3)
    out["gripper_minus_target_at_close_m"] = [round(float(v), 4) for v in offset]
    out["gripper_xy_offset_at_close_m"] = round(float(np.linalg.norm(offset[:2])), 4)
    out["target_pushed_before_close_m"] = round(float(np.linalg.norm(obj[i, :2] - obj[0, :2])), 4)
    if err is not None:
        out["joint_tracking_error_at_close_rad"] = round(float(err[i]), 4)
    return out


def stage_hint(record: dict, attr: dict) -> str | None:
    """Earliest stage whose measurement crosses its threshold, for failed executed runs only."""
    if record.get("stable_lift") or not record.get("server_success") or not record.get("executed_steps"):
        return None
    th = ATTRIBUTION_THRESHOLDS
    if attr.get("first_pick_matches_target") is False:
        return "grounding"
    if (attr.get("perception_xy_error_m") or 0) > th["perception_xy_error_m"]:
        return "perception_pose"
    if (attr.get("joint_tracking_error_at_close_rad") or 0) > th["joint_tracking_error_rad"]:
        return "execution_tracking"
    if (attr.get("target_pushed_before_close_m") or 0) > th["pushed_before_close_m"]:
        return "approach_contact"
    if (attr.get("gripper_xy_offset_at_close_m") or 0) > th["gripper_xy_offset_at_close_m"]:
        return "grasp_pose_or_frame"
    if not attr.get("trace_available"):
        return "insufficient_data"
    if "gripper_xy_offset_at_close_m" not in attr:
        return "no_close"
    return "closure_or_lift_dynamics"


def attribute(record: dict, trace: dict | None, perceived: dict | None, plan: dict | None) -> dict[str, Any]:
    attr: dict[str, Any] = {"run_id": record["run_id"], "thresholds": ATTRIBUTION_THRESHOLDS}
    attr["trace_available"] = bool(trace) and len(trace.get("t", [])) > 0
    attr["plan_timing"] = plan_timing(plan)
    attr.update(perception_and_grounding(
        perceived, plan, record.get("initial_object_positions") or {}, record.get("target_object")))
    attr.update(execution(record, trace))
    attr["stage_hint"] = stage_hint(record, attr)
    return attr
