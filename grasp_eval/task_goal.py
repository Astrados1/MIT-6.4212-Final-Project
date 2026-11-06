"""Automatic task-success check and box-shape comparisons from simulator ground truth.

Pure numpy; runs after the episode from the run record. Quaternions are (w, x, y, z).

A goal spec is a list of clauses, one per goal predicate in the instruction:

    [{"objects": ["$target"], "relation": "in", "container": "bowl"}]
    [{"objects": ["block_red", "block_blue", "block_green", "block_yellow"], "count": 3,
      "relation": "in", "container": "bowl"}]

"$target" stands for the run's scored target object. `count` (default: all objects)
is how many of `objects` must satisfy the relation. A clause with a null container or
any unknown object name cannot be checked, and the whole task is then "unchecked".

Relations, tested on the final resting state with the gripper open:
  in:  object center inside the container's xy box, and object bottom no lower than the
       container bottom and no higher than the container top (resting inside it).
  on:  object center inside the container's xy box, and object bottom within
       ON_Z_TOL_M of the container top.
TiPToP/Gemini express both as on(object, container); the physical check differs.

A clause may set "frame": "container" (relation "in" only). The test is then made in the container's
own frame: the object's geometry center must lie inside the container's walls and between its floor and
its rim. Use it for a rectangular container that can be turned on the table and for long objects that
rest tilted (scene 3, banana in bin). The default world-axis boxes misjudge both: a tilted banana's box
dips below the bin floor, and a turned bin's box takes in the table beside it (baseline review, Nov 6
2026: 8 false failures and 3 false successes out of 29). Round containers (bowl, mug) keep the default.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TARGET_TOKEN = "$target"
XY_MARGIN_M = 0.0      # shrink (+) or grow (-) the container's xy box
IN_Z_TOL_M = 0.01      # object bottom may sit this far below the container bottom (penetration/noise)
IN_Z_ABOVE_TOP_M = 0.02
ON_Z_TOL_M = 0.02


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def geometry_center_positions(positions, quats, local_bounds) -> np.ndarray:
    """World geometry-box centers from rigid-body origins and wxyz orientations.

    Supports one pose or batches. Several upstream assets have mesh centers more
    than 10 cm from their body origins; scoring those origins gives false misses.
    """
    p, q = np.asarray(positions, dtype=float), np.asarray(quats, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    center = np.asarray(local_bounds, dtype=float).mean(axis=0)
    cross = np.cross(q[..., 1:], center)
    return p + center + 2 * (q[..., :1] * cross + np.cross(q[..., 1:], cross))


def world_aabb(local_min, local_max, pos, quat) -> tuple[np.ndarray, np.ndarray]:
    """World-frame axis-aligned box of a local box placed at (pos, quat)."""
    lo, hi = np.asarray(local_min, dtype=float), np.asarray(local_max, dtype=float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    world = corners @ quat_to_matrix(quat).T + np.asarray(pos, dtype=float)
    return world.min(axis=0), world.max(axis=0)


def aabb_iou(a_min, a_max, b_min, b_max) -> float:
    a_min, a_max, b_min, b_max = (np.asarray(v, dtype=float) for v in (a_min, a_max, b_min, b_max))
    inter = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None).prod()
    union = (a_max - a_min).prod() + (b_max - b_min).prod() - inter
    return float(inter / union) if union > 0 else 0.0


def object_world_aabb(name: str, record: dict, when: str = "initial") -> tuple[np.ndarray, np.ndarray] | None:
    """True world box of a sim object from the record's local bounds and its initial/final pose."""
    bounds = (record.get("object_local_bounds") or {}).get(name)
    pos = (record.get(f"{when}_object_positions") or {}).get(name)
    quat = (record.get(f"{when}_object_quats") or {}).get(name)
    if bounds is None or pos is None or quat is None:
        return None
    return world_aabb(bounds[0], bounds[1], pos, quat)


def _relation_holds(relation: str, obj_box, cont_box) -> tuple[bool, dict]:
    (o_min, o_max), (c_min, c_max) = obj_box, cont_box
    center = (o_min + o_max) / 2
    in_xy = bool(np.all(center[:2] >= c_min[:2] + XY_MARGIN_M) and np.all(center[:2] <= c_max[:2] - XY_MARGIN_M))
    bottom = float(o_min[2])
    detail = {"center_xy_inside_container": in_xy, "object_bottom_z": round(bottom, 4),
              "container_z_range": [round(float(c_min[2]), 4), round(float(c_max[2]), 4)]}
    if relation == "in":
        z_ok = c_min[2] - IN_Z_TOL_M <= bottom <= c_max[2] + IN_Z_ABOVE_TOP_M
    elif relation == "on":
        z_ok = abs(bottom - c_max[2]) <= ON_Z_TOL_M
    else:
        raise ValueError(f"unknown relation {relation!r}")
    return in_xy and bool(z_ok), detail


def _in_container_frame(obj: str, container: str, record: dict) -> tuple[bool, dict] | None:
    """'in' tested in the container's frame from the final poses; None if a pose or bound is missing."""
    bounds = record.get("object_local_bounds") or {}
    pos, quat = record.get("final_object_positions") or {}, record.get("final_object_quats") or {}
    if any(k not in d for k in (obj, container) for d in (bounds, pos, quat)):
        return None
    center = geometry_center_positions(pos[obj], quat[obj], bounds[obj])
    rot = quat_to_matrix(quat[container])
    local = rot.T @ (np.asarray(center, dtype=float) - np.asarray(pos[container], dtype=float))
    lo, hi = np.asarray(bounds[container][0], dtype=float), np.asarray(bounds[container][1], dtype=float)
    up = int(np.argmax(np.abs(rot[2, :])))  # the container axis closest to world vertical
    sign = 1.0 if rot[2, up] > 0 else -1.0
    floor, rim = (lo[up], hi[up]) if sign > 0 else (-hi[up], -lo[up])
    height = float(local[up] * sign)
    walls = [i for i in range(3) if i != up]
    inside = bool(all(lo[i] + XY_MARGIN_M < local[i] < hi[i] - XY_MARGIN_M for i in walls))
    ok = inside and floor - IN_Z_TOL_M <= height <= rim
    return ok, {"frame": "container", "center_inside_walls": inside,
                "center_height_in_container_m": round(height - float(floor), 4),
                "container_depth_m": round(float(rim - floor), 4)}


def check_goal(goal: list[dict] | None, record: dict, gripper_open_at_end: bool | None) -> dict[str, Any]:
    """Return {"task_success": True/False/None, "clauses": [...], "reason": str|None}."""
    if not goal:
        return {"task_success": None, "reason": "no goal spec for this scene"}
    target = record.get("target_object")
    clauses, unchecked = [], []
    for spec in goal:
        objs = [target if o == TARGET_TOKEN else o for o in spec.get("objects", [])]
        container = spec.get("container")
        count = int(spec.get("count", len(objs)))
        relation = spec.get("relation", "in")
        if not container or not objs or any(o in (None, "") for o in objs):
            unchecked.append(f"clause {spec} has an unset object or container name")
            continue
        cont_box = object_world_aabb(container, record, "final")
        if cont_box is None:
            unchecked.append(f"no final pose/bounds for container {container!r}")
            continue
        per_obj = {}
        if spec.get("frame") == "container":
            if relation != "in":
                unchecked.append(f"frame 'container' supports relation 'in' only, not {relation!r}")
                continue
            for o in objs:
                res = _in_container_frame(o, container, record)
                if res is None:
                    unchecked.append(f"no final pose/bounds for object {o!r}")
                    break
                per_obj[o] = {"satisfied": res[0], **res[1]}
            else:
                n_ok = sum(v["satisfied"] for v in per_obj.values())
                clauses.append({"relation": relation, "container": container, "frame": "container",
                                "required": count, "satisfied_count": n_ok, "ok": n_ok >= count,
                                "objects": per_obj})
            continue
        for o in objs:
            box = object_world_aabb(o, record, "final")
            if box is None:
                unchecked.append(f"no final pose/bounds for object {o!r}")
                break
            ok, detail = _relation_holds(relation, box, cont_box)
            per_obj[o] = {"satisfied": ok, **detail}
        else:
            n_ok = sum(v["satisfied"] for v in per_obj.values())
            clauses.append({"relation": relation, "container": container, "required": count,
                            "satisfied_count": n_ok, "ok": n_ok >= count, "objects": per_obj})
    if unchecked:
        return {"task_success": None, "clauses": clauses, "reason": "; ".join(unchecked)}
    success = all(c["ok"] for c in clauses)
    reason = None
    if success and gripper_open_at_end is False:
        success, reason = False, "gripper still closed at the end of the episode"
    return {"task_success": success, "clauses": clauses, "reason": reason}


def goal_objects(goal: list[dict] | None, target: str | None) -> tuple[set[str], set[str]]:
    """(objects to move, containers) named by a goal spec, with $target resolved."""
    objs, conts = set(), set()
    for spec in goal or []:
        objs |= {target if o == TARGET_TOKEN else o for o in spec.get("objects", []) if o}
        if spec.get("container"):
            conts.add(spec["container"])
    objs.discard(None)
    return objs, conts


def shape_comparison(perceived_min, perceived_max, true_min, true_max) -> dict[str, Any]:
    """Perceived (scene-completed) box vs the true box: per-axis extent ratio and 3D IoU."""
    pe = np.asarray(perceived_max, dtype=float) - np.asarray(perceived_min, dtype=float)
    te = np.asarray(true_max, dtype=float) - np.asarray(true_min, dtype=float)
    ratio = np.divide(pe, te, out=np.full(3, np.nan), where=te > 1e-6)
    return {
        "perceived_extent_m": [round(float(v), 4) for v in pe],
        "true_extent_m": [round(float(v), 4) for v in te],
        "extent_ratio": [None if np.isnan(v) else round(float(v), 3) for v in ratio],
        "aabb_iou": round(aabb_iou(perceived_min, perceived_max, true_min, true_max), 3),
    }
