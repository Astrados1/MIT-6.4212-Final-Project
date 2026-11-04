"""Captions for run videos: what the plan is doing in each frame, and the outcome at the end.

Drawing only touches the recorded frames, never the simulator or the actions. OpenCV is
imported lazily (it is in the droid-sim-evals environment); the label helpers are pure Python.
"""

from __future__ import annotations

import numpy as np

CAPTION_PX = 64
OUTCOME_SECONDS = 3.0
_FONT_SCALE = 1.1
_WHITE, _GREEN, _RED, _YELLOW = (255, 255, 255), (80, 220, 100), (240, 80, 80), (250, 210, 60)


def plan_step_text(plan: list | None, current_step: int, gripper_pending: str | None) -> str:
    """Human-readable plan step for the action the client just returned.

    Mirrors the upstream client's bookkeeping: `current_step` already points past a
    trajectory that has started, while a pending gripper action has not advanced it yet.
    """
    if plan is None:
        return "waiting for plan"
    if not plan:
        return "no plan"
    i = current_step if gripper_pending is not None else current_step - 1
    if not 0 <= i < len(plan):
        return f"plan finished ({len(plan)} steps)"
    step = plan[i]
    what = step.get("label") or step.get("action") or step.get("type")
    if step.get("type") == "gripper":
        what = f"gripper {step.get('action')} ({step.get('label') or ''})".replace(" ()", "")
    return f"step {i + 1}/{len(plan)}: {what}"


def caption_line(step_text: str, gripper_cmd: float, rise_m: float | None, probe: str | None) -> str:
    parts = [step_text, f"gripper cmd: {'CLOSE' if gripper_cmd > 0.5 else 'open'}"]
    if rise_m is not None:
        parts.append(f"target {100 * rise_m:+.1f} cm vs start")
    if probe:
        parts.append(f"probe: {probe}")
    return "   |   ".join(parts)


def add_caption(frame: np.ndarray, text: str) -> np.ndarray:
    """Return the frame with a black caption bar underneath."""
    import cv2

    bar = np.zeros((CAPTION_PX, frame.shape[1], 3), dtype=frame.dtype)
    cv2.putText(bar, text[:180], (16, CAPTION_PX - 22), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _WHITE, 2,
                cv2.LINE_AA)
    return np.concatenate([frame, bar], axis=0)


def outcome_lines(record: dict) -> list[tuple[str, tuple[int, int, int]]]:
    """Text for the end card, with a colour per line."""
    lines = [(f"run {record.get('run_id')}  scene {record.get('scene')} variant {record.get('variant')}"
              f"  trial {record.get('trial')}", _WHITE)]
    if record.get("harness_exception"):
        return lines + [("HARNESS ERROR (see observed_eval.log)", _RED)]
    if not record.get("server_success"):
        return lines + [(f"NO PLAN: {(record.get('server_error') or 'unknown')[:120]}", _RED)]
    score = record.get("score") or {}
    lifted = record.get("stable_lift")
    lines.append((f"stable lift: {'YES' if lifted else 'NO'}", _GREEN if lifted else _RED))
    if score:
        rise = score.get("max_rise_near_gripper_m")
        lines.append((f"closes: {score.get('num_close_commands')}   longest hold: {score.get('longest_hold_s', 0):.2f} s"
                      f"   max lift near gripper: {'n/a' if rise is None else f'{100 * rise:.1f} cm'}", _WHITE))
        if score.get("failure_hint"):
            lines.append((f"trace hint: {score['failure_hint']}", _YELLOW))
    lines.append(("task success and root cause: see review.html", _WHITE))
    return lines


def outcome_card(last_frame: np.ndarray, record: dict, fps: int) -> list[np.ndarray]:
    """A few seconds of the final frame, dimmed, with the outcome written over it."""
    import cv2

    card = (last_frame.astype(np.float32) * 0.35).astype(last_frame.dtype)
    y = 80
    for text, colour in outcome_lines(record):
        cv2.putText(card, text[:150], (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3, cv2.LINE_AA)
        y += 64
    return [card] * int(OUTCOME_SECONDS * fps)
