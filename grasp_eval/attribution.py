"""Stage attribution: where along perception -> grounding -> grasp -> execution a run went wrong.

Combines, after the fact and without touching the simulator:
  * the run record and trace written by observed_eval.py (simulator ground truth),
  * the scene TiPToP perceived (extract_perceived_scene.py output),
  * the executed plan (tiptop_plan.json).

Every metric is a measurement. `stage_hint` is a heuristic from fixed thresholds
(ATTRIBUTION_THRESHOLDS) to guide video review, not a label. The same pass also checks
task success (task_goal.py), measures in-hand motion while carried, and assigns the
TiPToP paper's failure category (paper_taxonomy.py) as a first pass for review.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from grasp_eval.paper_taxonomy import classify, cutamp_failure_kind
from grasp_eval.scoring import LiftCriteria, in_hand_motion
from grasp_eval.task_goal import TARGET_TOKEN, check_goal, goal_objects, object_world_aabb, shape_comparison

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


def _is_support_surface(obj: dict) -> bool:
    """The table/workspace cuboid TiPToP adds to every scene: not an object to match."""
    if normalize_name(obj.get("name", "")) in ("table", "workspace", "tablesurface"):
        return True
    lo, hi = obj.get("aabb_min"), obj.get("aabb_max")
    return lo is not None and hi is not None and max(hi[0] - lo[0], hi[1] - lo[1]) > 0.5


def _perceived_box(obj: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if obj.get("aabb_min") is None or obj.get("aabb_max") is None:
        return None
    shift = np.array([0.0, 0.0, CLIENT_CAM_Z_OFFSET_M])
    return np.asarray(obj["aabb_min"], dtype=float) + shift, np.asarray(obj["aabb_max"], dtype=float) + shift


def _lookup_label(label: str, mapping: dict):
    if label in mapping:
        return mapping[label]
    return next((v for k, v in mapping.items() if normalize_name(k) == normalize_name(label)), None)


def goal_translation(atoms: list[dict] | None, label_to_sim: dict[str, str], goal: list[dict] | None,
                     target: str | None) -> dict[str, Any]:
    """Compare Gemini's grounded atoms with the expected goal, both in sim object names.

    Gemini writes "in the bowl" as on(object, bowl), so every goal clause expects
    on(x, container) for `count` of its objects and nothing else.
    """
    out: dict[str, Any] = {}
    if atoms is None or not goal:
        return out
    objs, conts = goal_objects(goal, target)
    if not conts or any(o is None for o in objs):
        return out
    mapped, unmapped = [], []
    for a in atoms:
        args = [_lookup_label(x, label_to_sim) for x in a.get("args", [])]
        if any(x is None for x in args):
            unmapped.append(a)
        else:
            mapped.append((a.get("predicate", "").lower(), *args))
    expected = sorted({("on", o, spec["container"]) for spec in goal
                       for o in [target if x == TARGET_TOKEN else x for x in spec.get("objects", [])]})
    out["grounded_atoms_mapped"] = [list(m) for m in mapped]
    out["expected_atoms"] = [list(e) for e in expected]
    if unmapped:
        out["grounded_atoms_unmapped"] = unmapped
    correct = True
    used = set()
    for spec in goal:
        members = {target if x == TARGET_TOKEN else x for x in spec.get("objects", [])}
        need = int(spec.get("count", len(members)))
        hits = {m for m in mapped if m[0] == "on" and len(m) == 3 and m[1] in members and m[2] == spec["container"]}
        used |= hits
        if len(hits) != need:
            correct = False
    if set(mapped) - used:
        correct = False  # extra predicates (wrong object, wrong container, holding, ...)
    if unmapped and not correct:
        out["goal_translation_correct"] = None  # can't tell a bad translation from a bad detection
    else:
        out["goal_translation_correct"] = correct and not unmapped
    return out


def perception_and_grounding(perceived: dict | None, plan: dict | None, gt_initial: dict, target: str | None,
                             record: dict | None = None):
    out: dict[str, Any] = {}
    record = record or {}
    goal = record.get("goal")
    if perceived:
        if perceived.get("grounded_atoms") is not None:
            out["grounded_atoms"] = perceived["grounded_atoms"]
        reason = perceived.get("planning_failure_reason")
        if reason:
            out["cutamp_failure_reason"] = reason
            out["cutamp_failure_kind"] = cutamp_failure_kind(reason)
    if not perceived or "objects" not in perceived:
        out["perceived_scene_available"] = False
        if perceived and perceived.get("error"):
            out["perceived_scene_error"] = perceived["error"]
        return out
    out["perceived_scene_available"] = True
    movables = [o for o in perceived["objects"] if o.get("role") == "movable"]
    out["perceived_movables"] = [o["name"] for o in movables]
    candidates = movables + [o for o in perceived["objects"]
                             if o.get("role") != "movable" and not _is_support_surface(o)]

    per_object = []
    matched_gt = set()
    label_to_sim: dict[str, str] = {}
    boxes: dict[str, tuple] = {}
    for o in candidates:
        c = _perceived_center(o)
        if c is None:
            continue
        gt_name, d = _nearest(c, gt_initial)
        if gt_name is None:
            continue
        gt = np.asarray(gt_initial[gt_name], dtype=float)
        per_object.append({"perceived": o["name"], "role": o.get("role"), "nearest_sim_object": gt_name,
                           "xy_error_m": round(d, 4), "z_error_m": round(float(c[2] - gt[2]), 4)})
        if d <= ATTRIBUTION_THRESHOLDS["missed_detection_radius_m"]:
            matched_gt.add(gt_name)
            label_to_sim[o["name"]] = gt_name
            box = _perceived_box(o)
            if box is not None and (gt_name not in boxes or d < boxes[gt_name][2]):
                boxes[gt_name] = (box[0], box[1], d, o["name"])
    out["perceived_objects"] = per_object
    out["sim_objects_without_nearby_detection"] = sorted(set(gt_initial) - matched_gt)

    picks = pick_objects(plan)
    out["plan_pick_objects"] = picks
    first_sim = None
    if picks:
        first = picks[0]
        match = next((p for p in per_object if p["perceived"] == first), None) or next(
            (p for p in per_object if normalize_name(p["perceived"]) == normalize_name(first)), None)
        out["first_pick_found_in_perceived_scene"] = match is not None
        if match:
            first_sim = match["nearest_sim_object"]
            out["first_pick_maps_to_sim_object"] = first_sim
            out["perception_xy_error_m"] = match["xy_error_m"]
            out["perception_z_error_m"] = match["z_error_m"]
            if target:
                out["first_pick_matches_target"] = first_sim == target
        counts = perceived.get("m2t2_grasp_counts")
        if counts is not None:
            n = _lookup_label(first, counts)
            out["first_pick_m2t2_grasp_count"] = n
            # cuTAMP 0.0.6 uses M2T2 grasps for an object iff TiPToP associated at least one with it;
            # otherwise it falls back to its heuristic 4-DoF sampler.
            out["grasp_source"] = "m2t2" if n else "heuristic"

    objs, conts = goal_objects(goal, target)
    relevant = {n for n in objs | conts | {first_sim} if n in gt_initial}
    if goal and not any(o is None for o in objs):
        named = objs | conts
        out["goal_objects_without_detection"] = sorted((named & set(gt_initial)) - matched_gt)
        if named - set(gt_initial):
            out["goal_names_not_in_sim"] = sorted(named - set(gt_initial))
    shapes = {}
    for name in sorted(relevant):
        true_box = object_world_aabb(name, record, "initial")
        if name in boxes and true_box is not None:
            p_min, p_max, _, label = boxes[name]
            shapes[name] = {"perceived": label, **shape_comparison(p_min, p_max, *true_box)}
    if shapes:
        out["shape"] = shapes
    out.update(goal_translation(perceived.get("grounded_atoms"), label_to_sim, goal, target))
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
    if "objects_quat" in trace and "gripper_quat" in trace:
        baseline = (record.get("initial_object_positions") or {}).get(target)
        if baseline is not None:
            out["in_hand"] = in_hand_motion(
                trace["t"], obj, np.asarray(trace["objects_quat"])[:, k, :], grip, trace["gripper_quat"],
                trace["gripper_cmd"], baseline[2], LiftCriteria(**(record.get("criteria") or {})))
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
        perceived, plan, record.get("initial_object_positions") or {}, record.get("target_object"), record))
    attr.update(execution(record, trace))
    attr["stage_hint"] = stage_hint(record, attr)
    if record.get("server_success") and record.get("executed_steps"):
        gripper_open = None
        if attr["trace_available"]:
            gripper_open = bool(np.asarray(trace["gripper_cmd"])[-1] <= 0.5)
        attr["task_check"] = check_goal(record.get("goal"), record, gripper_open)
    attr.update(classify(record, attr))
    return attr
