"""Paired summary of a task-level recovery suite against its baseline (docs/task_recovery.md).

    python3 -m grasp_eval.task_summary results/<task suite> --baseline results/<baseline suite>

Writes summary_task.md and summary_task.json in the suite directory. Task success comes from the same
goal check in both arms: the baseline's attribution.jsonl `task_check` (overridden by its
manual_labels.csv `task_success` where a reviewer set one), and the recovery record's `task_success`.
The recovery arm is reported twice: at its own end, and at the baseline's episode end (150 s)
(`task_success_at_baseline_limit`), which removes any advantage from the extra time a retry takes.
Baseline trials without a plan (skipped.jsonl) count as failures in both arms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from grasp_eval.records import read_jsonl
from grasp_eval.recovery_records import TASK_TRIALS_FILE


def sign_test_p(rescued: int, harmed: int) -> float:
    """Two-sided exact binomial (sign) test on the discordant pairs."""
    n = rescued + harmed
    if n == 0:
        return 1.0
    k = max(rescued, harmed)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def baseline_success(baseline: Path) -> dict[str, bool | None]:
    """run_id -> task success (attribution task_check, reviewer override wins). None = unchecked."""
    out: dict[str, bool | None] = {}
    for a in read_jsonl(baseline / "attribution.jsonl"):
        out[a["run_id"]] = (a.get("task_check") or {}).get("task_success")
    for r in read_jsonl(baseline / "runs.jsonl"):
        if r["run_id"] not in out:
            out[r["run_id"]] = False if not r.get("server_success") else None
    labels = baseline / "manual_labels.csv"
    if labels.exists():
        with open(labels, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                v = (row.get("task_success") or "").strip().lower()
                if v in ("true", "1", "yes"):
                    out[row["run_id"]] = True
                elif v in ("false", "0", "no"):
                    out[row["run_id"]] = False
    return out


def paired_rows(suite: Path, baseline: Path) -> list[dict[str, Any]]:
    base = baseline_success(baseline)
    scene_of = {r["run_id"]: int(r["scene"]) for r in read_jsonl(baseline / "runs.jsonl")}
    rows = []
    for rec in read_jsonl(suite / TASK_TRIALS_FILE):
        rid = rec.get("paired_baseline_run")
        if rid is None:
            continue
        rows.append({"run_id": rid, "scene": rec.get("scene", scene_of.get(rid)), "baseline": base.get(rid),
                     "recovery": rec.get("task_success"), "recovery_at_limit": rec.get("task_success_at_baseline_limit"),
                     "retries": rec.get("num_retries", 0), "stop": rec.get("stop_reason"),
                     "first_verdict": rec.get("first_checkpoint_verdict"), "checkpoints": rec.get("checkpoints") or [],
                     "events": rec.get("events") or [], "skipped": False})
    for s in read_jsonl(suite / "skipped.jsonl"):
        rid = s["run_id"]
        rows.append({"run_id": rid, "scene": scene_of.get(rid), "baseline": False, "recovery": False,
                     "recovery_at_limit": False, "retries": 0, "stop": "baseline_no_plan", "first_verdict": None,
                     "checkpoints": [], "events": [], "skipped": True})
    return rows


def compare(rows: list[dict], key: str) -> dict[str, Any]:
    pairs = [(r["baseline"], r[key]) for r in rows if r["baseline"] is not None and r[key] is not None]
    n = len(pairs)
    b = sum(1 for x, _ in pairs if x)
    r = sum(1 for _, y in pairs if y)
    rescued = sum(1 for x, y in pairs if not x and y)
    harmed = sum(1 for x, y in pairs if x and not y)
    return {"pairs": n, "baseline_success": b, "recovery_success": r, "rescued": rescued, "harmed": harmed,
            "sign_test_p": round(sign_test_p(rescued, harmed), 4),
            "baseline_ci": [round(v, 3) for v in wilson(b, n)], "recovery_ci": [round(v, 3) for v in wilson(r, n)],
            "unchecked": sum(1 for x in rows if x["baseline"] is None or x[key] is None)}


def summarize(rows: list[dict]) -> dict[str, Any]:
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
    verdicts = Counter(c.get("verdict") for r in rows for c in r["checkpoints"])
    first = Counter(r["first_verdict"] for r in rows if not r["skipped"])
    retried = [r for r in rows if r["retries"]]
    no_retry = [r for r in rows if not r["retries"] and not r["skipped"] and r["baseline"] is not None
                and r["recovery"] is not None]
    agree = sum(1 for r in no_retry if bool(r["baseline"]) == bool(r["recovery"]))
    return {
        "overall": compare(rows, "recovery"), "overall_at_baseline_limit": compare(rows, "recovery_at_limit"),
        "by_scene": {str(s): compare(v, "recovery") for s, v in sorted(by_scene.items(), key=lambda x: str(x[0]))},
        "checkpoint_verdicts": dict(verdicts), "first_checkpoint_verdicts": dict(first),
        "trials_with_retry": len(retried), "recovered": sum(1 for r in retried if r["recovery"]),
        "stops": dict(Counter(r["stop"] for r in rows)),
        "dropped_during_carry": sum(1 for r in rows for e in r["events"] if e.get("what") == "dropped_during_carry"),
        "replay_fidelity": {"trials_without_retry": len(no_retry), "same_outcome_as_baseline": agree},
        "skipped_baseline_no_plan": sum(1 for r in rows if r["skipped"]),
    }


def render(s: dict, suite: str, baseline: str) -> str:
    def row(name, c):
        return (f"| {name} | {c['pairs']} | {c['baseline_success']} ({100 * c['baseline_success'] / max(1, c['pairs']):.0f}%) "
                f"| {c['recovery_success']} ({100 * c['recovery_success'] / max(1, c['pairs']):.0f}%) | {c['rescued']} "
                f"| {c['harmed']} | {c['sign_test_p']} |")
    lines = [f"# Paired task-level recovery: {suite}", "", f"Baseline: `{baseline}`. Oracle checkpoint (simulator "
             "truth), never observation-only. Task success uses the baseline's goal check in both arms.", "",
             "| scope | pairs | baseline success | recovery success | rescued (fail→success) | harmed (success→fail) "
             "| sign test p |", "|---|---|---|---|---|---|---|",
             row("overall, recovery at its end", s["overall"]),
             row("overall, recovery at the baseline's episode end", s["overall_at_baseline_limit"])]
    lines += [row(f"scene {k}", v) for k, v in s["by_scene"].items()]
    rf = s["replay_fidelity"]
    lines += ["", f"- Checkpoint verdicts (all grasps): {s['checkpoint_verdicts']}",
              f"- First-grasp verdicts: {s['first_checkpoint_verdicts']}",
              f"- Trials with a retry: {s['trials_with_retry']}; task succeeded after a retry: {s['recovered']}",
              f"- Drops during carry (recorded, not acted on): {s['dropped_during_carry']}",
              f"- Replay fidelity: {rf['same_outcome_as_baseline']}/{rf['trials_without_retry']} trials without a "
              "retry ended like their baseline trial (differences come from the check motion or simulator "
              "non-determinism)",
              f"- Baseline trials without a plan (failure in both arms): {s['skipped_baseline_no_plan']}",
              f"- Stop reasons: {s['stops']}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("suite", type=Path)
    p.add_argument("--baseline", type=Path, required=True)
    a = p.parse_args(argv)
    rows = paired_rows(a.suite, a.baseline)
    s = summarize(rows)
    (a.suite / "summary_task.json").write_text(json.dumps(s, indent=2))
    text = render(s, a.suite.name, str(a.baseline))
    (a.suite / "summary_task.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
