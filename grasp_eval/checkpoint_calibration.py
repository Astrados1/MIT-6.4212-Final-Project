"""Calibrate the task-level checkpoint on baseline traces (docs/task_recovery.md, section "Calibration").

    python3 -m grasp_eval.checkpoint_calibration results/baseline_panda_20261106 [more dirs] \
        [--out results/checkpoint_calibration] [--<criterion> value ...]

Every baseline trial passes through close -> retraction -> transfer, so its trace contains the motion the
checkpoint watches. For every grasp (rising edge of the gripper command) this:
  * labels it from the rest of the trial: truly held = the object rose >= 5 cm while within 0.20 m of
    the finger reference (the frozen week-one criterion), before the gripper opened;
  * measures the object's rise when the hand had risen 3/5/8/10 cm, using only the near-vertical part
    of the motion (hand within --max-lateral-m of its x/y at the close), which is what both lift modes do;
  * replays LiftMonitor with the given criteria over that same near-vertical part and records its verdict.
The report is a confusion table (false "held" is the dangerous error: it carries nothing) and, per
hand-rise level, the overlap between held and missed grasps. Writes calibration.csv and calibration.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from grasp_eval.task_checkpoint import LiftMonitor, TaskCheckCriteria, Verdict

HAND_LEVELS = (0.03, 0.05, 0.08, 0.10)
TRUE_RISE_M = 0.05
TRUE_NEAR_M = 0.20


def grasp_windows(trace: dict) -> list[tuple[int, int]]:
    """(close sample, open sample) for every close: rising edges of the gripper command."""
    cmd = np.asarray(trace["gripper_cmd"], dtype=float) > 0.5
    out = []
    i = 0
    while i < len(cmd):
        if cmd[i] and (i == 0 or not cmd[i - 1]):
            j = i
            while j < len(cmd) and cmd[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def analyse_grasp(trace: dict, names: list[str], c0: int, c1: int, target: str | None,
                  criteria: TaskCheckCriteria, dt: float, max_lateral_m: float) -> dict[str, Any]:
    grip = np.asarray(trace["gripper_pos"], dtype=float)
    cen = np.asarray(trace["objects_center"] if "objects_center" in trace else trace["objects_pos"], dtype=float)
    r0 = max(0, c0 - 1)
    if target is None or target not in names:
        target = names[int(np.argmin(np.linalg.norm(cen[r0] - grip[c0], axis=1)))]
    k = names.index(target)
    rise = cen[c0:c1, k, 2] - cen[r0, k, 2]
    dist = np.linalg.norm(cen[c0:c1, k] - grip[c0:c1], axis=1)
    truly_held = bool(np.any((rise >= TRUE_RISE_M) & (dist <= TRUE_NEAR_M)))
    hand_rise = grip[c0:c1, 2] - grip[c0, 2]
    lateral = np.linalg.norm(grip[c0:c1, :2] - grip[c0, :2], axis=1)
    vertical_end = int(np.argmax(lateral > max_lateral_m)) if np.any(lateral > max_lateral_m) else len(lateral)
    levels = {}
    for h in HAND_LEVELS:
        hit = np.flatnonzero(hand_rise[:vertical_end] >= h)
        levels[f"obj_rise_at_hand_{int(h * 100)}cm"] = round(float(rise[hit[0]]), 4) if hit.size else None
    others0 = {n: cen[r0, j] for j, n in enumerate(names) if n != target}
    mon = LiftMonitor(criteria, dt, grip[c0], cen[r0, k], others0)
    verdict = None
    for i in range(c0, c0 + vertical_end):
        verdict = mon.update(grip[i], cen[i, k], {n: cen[i, j] for j, n in enumerate(names) if n != target})
        if verdict is not None:
            break
        if grip[i, 2] - grip[c0, 2] >= criteria.hand_rise_cap_m and mon.phase == "lift":
            verdict = mon.at_lift_end()
            break
    if verdict is None:
        verdict = mon.at_lift_end()
    return {"target": target, "truly_held": truly_held, "max_rise_m": round(float(rise.max()), 4),
            "vertical_hand_rise_m": round(float(hand_rise[:vertical_end].max()) if vertical_end else 0.0, 4),
            "monitor_verdict": verdict.value, "monitor_reason": mon.reason, **levels}


def analyse_dirs(dirs: list[Path], criteria: TaskCheckCriteria, max_lateral_m: float) -> list[dict]:
    rows = []
    for d in dirs:
        for run in sorted((d / "runs").glob("*")):
            tp, rp = run / "trace.npz", run / "record.json"
            if not (tp.exists() and rp.exists()):
                continue
            rec = json.loads(rp.read_text())
            z = np.load(tp, allow_pickle=False)
            trace = {k: z[k] for k in z.files}
            names = [str(n) for n in trace["object_names"]]
            t = np.asarray(trace["t"], dtype=float)
            dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 15
            for g, (c0, c1) in enumerate(grasp_windows(trace)):
                target = rec.get("target_object") if g == 0 else None
                row = {"suite": d.name, "run_id": rec["run_id"], "scene": rec.get("scene"), "grasp": g,
                       "stable_lift": rec.get("stable_lift") if g == 0 else None,
                       **analyse_grasp(trace, names, c0, c1, target, criteria, dt, max_lateral_m)}
                rows.append(row)
    return rows


def report(rows: list[dict], criteria: TaskCheckCriteria) -> str:
    lines = ["# Checkpoint calibration on baseline traces", "",
             f"{len(rows)} grasps. Truth: object rose >= {TRUE_RISE_M * 100:.0f} cm within {TRUE_NEAR_M} m of the "
             "finger reference before the gripper opened. Criteria:", "", "```", json.dumps(criteria.to_dict()), "```",
             "", "## Monitor verdict vs truth", "", "| truth \\ verdict | held | empty | slipped | uncertain | object_lost |",
             "|---|---|---|---|---|---|"]
    for truth in (True, False):
        c = Counter(r["monitor_verdict"] for r in rows if r["truly_held"] == truth)
        lines.append(f"| {'held' if truth else 'not held'} | " + " | ".join(
            str(c.get(v.value, 0)) for v in (Verdict.HELD, Verdict.EMPTY, Verdict.SLIPPED, Verdict.UNCERTAIN, Verdict.LOST))
            + " |")
    false_held = sum(1 for r in rows if not r["truly_held"] and r["monitor_verdict"] == "held")
    false_empty = sum(1 for r in rows if r["truly_held"] and r["monitor_verdict"] == "empty")
    lines += ["", f"- **False held (dangerous, carries nothing): {false_held}**",
              f"- False empty (wastes a retry on a good grasp): {false_empty}", "",
              "## Object rise at each hand rise (near-vertical part only)", "",
              "| hand rise | held grasps: min object rise | not-held grasps: max object rise | separated? | n held / n not |",
              "|---|---|---|---|---|"]
    for h in HAND_LEVELS:
        key = f"obj_rise_at_hand_{int(h * 100)}cm"
        held = [r[key] for r in rows if r["truly_held"] and r[key] is not None]
        miss = [r[key] for r in rows if not r["truly_held"] and r[key] is not None]
        sep = "yes" if held and miss and min(held) > max(miss) else ("n/a" if not held or not miss else "no")
        lines.append(f"| {int(h * 100)} cm | {min(held) * 100:.1f} cm |" if held else f"| {int(h * 100)} cm | — |")
        lines[-1] += (f" {max(miss) * 100:.1f} cm |" if miss else " — |") + f" {sep} | {len(held)} / {len(miss)} |"
    agree = [r for r in rows if r["stable_lift"] is not None]
    if agree:
        same = sum(1 for r in agree if bool(r["stable_lift"]) == r["truly_held"])
        lines += ["", f"- Truth label agrees with the week-one `stable_lift` on {same}/{len(agree)} first grasps."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dirs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-lateral-m", type=float, default=0.03, help="near-vertical: hand x/y within this of the close")
    cc = TaskCheckCriteria()
    for f in cc.to_dict():
        p.add_argument("--" + f.replace("_", "-"), type=float, default=getattr(cc, f))
    a = p.parse_args(argv)
    crit = TaskCheckCriteria(**{f: getattr(a, f) for f in cc.to_dict()})
    rows = analyse_dirs(a.dirs, crit, a.max_lateral_m)
    out = a.out or a.dirs[0] / "checkpoint_calibration"
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / "calibration.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    text = report(rows, crit)
    (out / "calibration.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
