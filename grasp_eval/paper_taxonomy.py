"""Classify runs into the TiPToP paper's failure taxonomy (arXiv 2603.09971, failure-analysis figure).

The paper traces each of 173 real-robot trials to one root-cause module:

    Success
    Grasp Failure            -> M2T2 Unstable Grasp | Heuristic Unstable Grasp
    Scene Completion Failure -> Partial Observability
    Gemini VLM Failure       -> Object Detection Failure | Goal Translation Failure
    cuTAMP Failure           -> cuRobo Motion Planning Failure | Rigid Grasp Assumption

`classify` assigns the earliest module (Gemini -> scene completion -> cuTAMP planning ->
grasp -> in-hand motion) whose measurement in the attribution crosses a fixed threshold
(PAPER_THRESHOLDS). Labels outside the paper's figure are kept separate so they never
inflate a paper category: infrastructure, needs_review, execution_failure, and extra
subcategories marked with "(not in paper)" in PAPER_CATEGORIES. Every automatic label
is a first pass to confirm on video and in the server log; reviewers override it with
the paper_category column of manual_labels.csv.

Pure standard library.
"""

from __future__ import annotations

from typing import Any

SUCCESS = "success"

# label -> (top-level node, sub node, in paper figure?)
PAPER_CATEGORIES: dict[str, tuple[str, str, bool]] = {
    SUCCESS: ("Success", "Success", True),
    "grasp_failure/m2t2_unstable_grasp": ("Grasp Failure", "M2T2 Unstable Grasp", True),
    "grasp_failure/heuristic_unstable_grasp": ("Grasp Failure", "Heuristic Unstable Grasp", True),
    "grasp_failure/unknown_grasp_source": ("Grasp Failure", "Unknown Grasp Source (not in paper)", False),
    "scene_completion_failure/partial_observability": ("Scene Completion Failure", "Partial Observability", True),
    "scene_completion_failure/shape_over_approximation": (
        "Scene Completion Failure", "Shape Over-approximation (not in paper figure)", False),
    "gemini_vlm_failure/object_detection_failure": ("Gemini VLM Failure", "Object Detection Failure", True),
    "gemini_vlm_failure/goal_translation_failure": ("Gemini VLM Failure", "Goal Translation Failure", True),
    "gemini_vlm_failure/invalid_response": ("Gemini VLM Failure", "Invalid Response (not in paper)", False),
    "cutamp_failure/curobo_motion_planning_failure": ("cuTAMP Failure", "cuRobo Motion Planning Failure", True),
    "cutamp_failure/rigid_grasp_assumption": ("cuTAMP Failure", "Rigid Grasp Assumption", True),
    "cutamp_failure/no_feasible_plan": ("cuTAMP Failure", "No Feasible Plan in Budget (not in paper figure)", False),
    "execution_failure/tracking_or_collision": ("Execution Failure (not in paper)", "Tracking or Collision", False),
    "unattributed_task_failure/cause_not_established": (
        "Unattributed Task Failure (not in paper)", "Cause Not Established (not in paper)", False),
    "infrastructure": ("Infrastructure (excluded)", "Infrastructure", False),
    "needs_review": ("Needs Review (excluded)", "Needs Review", False),
}

# Labels left out of the paper-comparable denominator ("All Trials" in the figure).
# A manually reviewed, confirmed task failure with an unresolved cause stays in
# that denominator via unattributed_task_failure/cause_not_established. Otherwise
# uncertainty about the cause would artificially increase the success rate.
EXCLUDED_FROM_PAPER_DENOMINATOR = {"infrastructure", "needs_review"}

PAPER_THRESHOLDS = {
    "shape_aabb_iou_min": 0.5,       # perceived vs true box overlap below this: bad completion
    "shape_extent_ratio_min": 0.7,   # any perceived/true extent below this: under-approximated
    "shape_extent_ratio_max": 1.3,   # any above this: over-approximated
    "in_hand_shift_m": 0.015,        # object moved this far relative to the hand while carried
    "in_hand_rot_deg": 10.0,
    "joint_tracking_error_rad": 0.05,
}

_INFRA_ERROR_MARKERS = (
    "could not connect", "connection refused", "api key", "api_key", "permission_denied", "resource_exhausted",
    "quota", "unavailable", "deadline_exceeded", "429", "503", "500 internal", "clienterror", "servererror",
    "timeout", "out of memory", "cuda error", "non-json server reply",
)


def cutamp_failure_kind(reason: str | None) -> str | None:
    """Bucket cuTAMP v0.0.6 failure_reason strings (cutamp/algorithm.py)."""
    if not reason:
        return None
    r = reason.lower()
    if r.startswith("motion planning failed"):
        return "motion_planning"
    if "time budget" in r:
        return "time_budget"
    if "particle initialization" in r:
        return "particle_initialization"
    if "no valid plan skeletons" in r:
        return "no_plan_skeleton"
    if "no satisfying particles" in r:
        return "no_satisfying_particles"
    return "other"


def _shape_flags(shapes: dict[str, dict]) -> tuple[list[str], list[str]]:
    th = PAPER_THRESHOLDS
    under, over = [], []
    for name, s in (shapes or {}).items():
        ratios = [r for r in s.get("extent_ratio", []) if r is not None]
        if (s.get("aabb_iou") is not None and s["aabb_iou"] < th["shape_aabb_iou_min"]) or \
                any(r < th["shape_extent_ratio_min"] for r in ratios):
            under.append(name)
        elif any(r > th["shape_extent_ratio_max"] for r in ratios):
            over.append(name)
    return under, over


def classify(record: dict | None, attr: dict | None, task_success: bool | None = None) -> dict[str, Any]:
    """Return {"paper_category": label, "evidence": [...]} for one run.

    `task_success` overrides the automatic check in attr["task_check"] (e.g. a manual label).
    """
    attr = attr or {}
    ev: list[str] = []

    def out(label: str) -> dict[str, Any]:
        return {"paper_category": label, "evidence": ev}

    if record is None:
        ev.append("no record written (crash or kill)")
        return out("infrastructure")
    if record.get("harness_exception"):
        ev.append("harness/simulator exception")
        return out("infrastructure")
    if not record.get("server_response_received"):
        ev.append(record.get("server_error") or "no response from TiPToP server")
        return out("infrastructure")

    err = (record.get("server_error") or "")
    err_l = err.lower()
    plan_ok = bool(record.get("server_success"))
    if not plan_ok and "gemini returned a non-json response" in err_l:
        ev.append("Gemini output could not be parsed")
        return out("gemini_vlm_failure/invalid_response")
    if not plan_ok and any(m in err_l for m in _INFRA_ERROR_MARKERS) and "cutamp failed" not in err_l:
        ev.append(f"server error looks like infrastructure: {err[:200]}")
        return out("infrastructure")

    if task_success is None:
        task_success = (attr.get("task_check") or {}).get("task_success")
    in_hand = attr.get("in_hand") or {}
    executed = plan_ok and (record.get("executed_steps") or 0) > 0
    if executed and task_success is True:
        return out(SUCCESS)

    # 1. Gemini: goal translation, then detection/grounding.
    if attr.get("goal_translation_correct") is False:
        ev.append(f"grounded atoms {attr.get('grounded_atoms_mapped')} != expected {attr.get('expected_atoms')}")
        return out("gemini_vlm_failure/goal_translation_failure")
    missed = attr.get("goal_objects_without_detection") or []
    if missed:
        ev.append(f"goal objects with no perceived object within 5 cm: {missed}")
        return out("gemini_vlm_failure/object_detection_failure")
    if attr.get("first_pick_matches_target") is False:
        ev.append(f"first Pick maps to {attr.get('first_pick_maps_to_sim_object')}, not {record.get('target_object')}")
        return out("gemini_vlm_failure/object_detection_failure")

    # 2. Scene completion (shape of picked object / goal objects / containers).
    under, over = _shape_flags(attr.get("shape") or {})
    if under:
        ev.append(f"under-approximated shape: {under}")
        return out("scene_completion_failure/partial_observability")
    if over:
        ev.append(f"over-approximated shape: {over}")
        return out("scene_completion_failure/shape_over_approximation")

    # 3. cuTAMP found no plan.
    if not plan_ok:
        kind = attr.get("cutamp_failure_kind") or cutamp_failure_kind(err.split("cuTAMP failed to find a plan:")[-1].strip()
                                                                       if "cutamp failed" in err_l else None)
        if kind == "motion_planning":
            ev.append(f"cuTAMP: {attr.get('cutamp_failure_reason') or err}")
            return out("cutamp_failure/curobo_motion_planning_failure")
        if kind is not None:
            ev.append(f"cuTAMP ({kind}): {attr.get('cutamp_failure_reason') or err}")
            return out("cutamp_failure/no_feasible_plan")
        ev.append(f"server returned no plan for an unrecognized reason: {err[:200]}")
        return out("needs_review")
    if not executed:
        ev.append("plan returned but no step executed")
        return out("needs_review")

    # 4. Grasp (failed to lift, or dropped while carrying).
    lifted = bool(record.get("stable_lift"))
    dropped = bool(in_hand.get("dropped_before_release"))
    if not lifted or dropped:
        if not (record.get("score") or {}).get("num_close_commands"):
            ev.append("plan executed without a close command" + (" (time limit)" if record.get("truncated") else ""))
            return out("needs_review")
        tracking = attr.get("joint_tracking_error_at_close_rad")
        if tracking is not None and tracking > PAPER_THRESHOLDS["joint_tracking_error_rad"]:
            ev.append(f"joint tracking error at close {tracking} rad")
            return out("execution_failure/tracking_or_collision")
        ev.append("dropped while carried" if lifted else f"no stable lift ({(record.get('score') or {}).get('failure_hint')})")
        source = attr.get("grasp_source")
        ev.append(f"grasp source: {source or 'unknown'}")
        if source == "m2t2":
            return out("grasp_failure/m2t2_unstable_grasp")
        if source == "heuristic":
            return out("grasp_failure/heuristic_unstable_grasp")
        return out("grasp_failure/unknown_grasp_source")

    # 5. Lifted and carried, but the task did not end in the goal state.
    shift, rot = in_hand.get("max_in_hand_shift_m"), in_hand.get("max_in_hand_rot_deg")
    if (shift is not None and shift > PAPER_THRESHOLDS["in_hand_shift_m"]) or \
            (rot is not None and rot > PAPER_THRESHOLDS["in_hand_rot_deg"]):
        ev.append(f"object moved in hand: {shift} m, {rot} deg")
        if task_success is False:
            return out("cutamp_failure/rigid_grasp_assumption")
    if task_success is None:
        ev.append("task success not checked (no goal spec or unknown names) and not manually labeled")
    else:
        ev.append("lifted and carried without slip, but the goal state was not reached")
    return out("needs_review")
