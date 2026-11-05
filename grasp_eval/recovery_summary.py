"""Attempt-aware summary for week-two checkpoint recovery results.

    python -m grasp_eval.recovery_summary results/week2_checkpoint_oracle

Reads recovery_trials.jsonl and attempts.jsonl (and schedule.jsonl, if a suite wrote one, so
trials that crashed before writing a record still count in their condition's denominator) and
writes summary_checkpoint.md next to them. Conditions are kept apart by mode, detector, probe
and instruction; every rate states its denominator.

This is a grasp-only experiment: the final state is deliberately unplaced, so none of these
trials feed the paper's whole-task taxonomy.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from grasp_eval.records import read_jsonl
from grasp_eval.recovery_records import ATTEMPTS_FILE, TRIALS_FILE
from grasp_eval.summarize import _frac, _frac_ci

NO_RECORD = "no_record"


def condition_key(t: dict) -> tuple:
    probe = t.get("probe") or {}
    return (t.get("mode", "?"), t.get("detector", "?"), probe.get("label") or probe.get("type", "baseline"),
            t.get("instruction", ""))


def load(results_dir: Path) -> tuple[list[dict], dict[str, list[dict]], list[str]]:
    trials = read_jsonl(results_dir / TRIALS_FILE)
    attempts: dict[str, list[dict]] = defaultdict(list)
    for a in read_jsonl(results_dir / ATTEMPTS_FILE):
        attempts[a["trial_id"]].append(a)
    for v in attempts.values():
        v.sort(key=lambda a: a["attempt_index"])
    warnings = []
    seen = Counter(t["trial_id"] for t in trials)
    dup = [k for k, n in seen.items() if n > 1]
    if dup:
        warnings.append(f"duplicate trial records (kept the last): {dup[:5]}")
        trials = list({t["trial_id"]: t for t in trials}.values())
    recorded = {t["trial_id"] for t in trials}
    for s in read_jsonl(results_dir / "schedule.jsonl"):
        if s["trial_id"] not in recorded:
            trials.append({**s, "final_success": False, "stop_reason": NO_RECORD, "num_attempts": 0,
                           "num_closes": 0, "num_planning_calls": 0, "first_attempt_success": None})
            warnings.append(f"{s['trial_id']}: scheduled but no record (crash or timeout); counted as a failure")
    return trials, attempts, warnings


def _med(xs: list[float]) -> str:
    return f"{statistics.median(xs):.1f}" if xs else "n/a"


def condition_metrics(trials: list[dict], attempts: dict[str, list[dict]]) -> dict:
    n = len(trials)
    a0 = {t["trial_id"]: (attempts.get(t["trial_id"]) or [None])[0] for t in trials}
    first_hist = sum(1 for t in trials if t.get("first_attempt_historical_lift"))
    first_ckpt = sum(1 for t in trials if t.get("first_attempt_success"))
    final = sum(1 for t in trials if t.get("final_success"))
    # First grasp failed: attempt 0 executed and was not confirmed held.
    first_failed = [t for t in trials if a0[t["trial_id"]] and a0[t["trial_id"]].get("outcome") is not None
                    and a0[t["trial_id"]]["outcome"] != "holding_target"]
    eligible = [t for t in first_failed if len(attempts.get(t["trial_id"], [])) > 1]
    recovered_ff = sum(1 for t in first_failed if t.get("final_success"))
    recovered_el = sum(1 for t in eligible if t.get("final_success"))
    lifted_not_held = 0
    for t in trials:
        for a in attempts.get(t["trial_id"], []):
            if (a.get("historical_lift") or {}).get("lifted") and a.get("outcome") != "holding_target":
                lifted_not_held += 1
    disagreements = [a for t in trials for a in attempts.get(t["trial_id"], [])
                     if a.get("reference_outcome") and a.get("outcome") and a["reference_outcome"] != a["outcome"]]
    compared = [a for t in trials for a in attempts.get(t["trial_id"], []) if a.get("reference_outcome")]
    succ = [t for t in trials if t.get("final_success")]
    return {
        "n": n, "first_hist": first_hist, "first_ckpt": first_ckpt, "final": final,
        "first_failed": len(first_failed), "recovered_ff": recovered_ff,
        "eligible": len(eligible), "recovered_el": recovered_el,
        "attempts": [t.get("num_attempts", 0) for t in trials],
        "planning": [t.get("num_planning_calls", 0) for t in trials],
        "closes": [t.get("num_closes", 0) for t in trials],
        "tts_sim": [t["time_to_success_sim_s"] for t in succ if t.get("time_to_success_sim_s") is not None],
        "tts_wall": [t["time_to_success_wall_s"] for t in succ if t.get("time_to_success_wall_s") is not None],
        "stops": Counter(t.get("stop_reason", "?") for t in trials),
        "lifted_not_held": lifted_not_held,
        "disagree": len(disagreements), "compared": len(compared),
        "privileged": sorted({str(t.get("detector_privileged")) for t in trials}),
    }


def render(trials: list[dict], attempts: dict[str, list[dict]], warnings: list[str], title: str) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        groups[condition_key(t)].append(t)
    out = [f"# {title}", "",
           "Grasp-checkpoint recovery (week two). Trials stop once the target is confirmed held after the "
           "post-grasp retraction; nothing is placed, so task success is out of scope.", ""]
    if any(str(k[1]).startswith("oracle") for k in groups):
        out += ["**Detector: `oracle-state` uses privileged simulator state to decide whether to retry.** "
                "These are oracle-checkpoint results, not observation-only or image-based recovery.", ""]
    out += ["Denominators: *per scheduled trial* includes trials that stopped on planning, infrastructure or "
            "uncertain outcomes. *First grasp failed* = attempt 1 executed and was not confirmed held. "
            "*Retry-eligible* = those that actually got a second attempt.", ""]
    for key in sorted(groups):
        mode, det, probe, instr = key
        m = condition_metrics(groups[key], attempts)
        out += [f"## {mode} · detector `{det}` · {probe}", "", f"Instruction: {instr!r}", "",
                "| metric | value |", "|---|---|",
                f"| scheduled trials | {m['n']} |",
                f"| first-attempt stable lift (week-one criterion, per trial) | {_frac_ci(m['first_hist'], m['n'])} |",
                f"| first-attempt confirmed held at checkpoint | {_frac_ci(m['first_ckpt'], m['n'])} |",
                f"| final confirmed success (per scheduled trial) | {_frac_ci(m['final'], m['n'])} |",
                f"| recovery success, among first grasp failed | {_frac(m['recovered_ff'], m['first_failed'])} |",
                f"| recovery success, among retry-eligible | {_frac(m['recovered_el'], m['eligible'])} |",
                f"| attempts per trial (median / max) | {_med(m['attempts'])} / {max(m['attempts'], default=0)} |",
                f"| planning calls per trial (median / max) | {_med(m['planning'])} / {max(m['planning'], default=0)} |",
                f"| closes per trial (median / max) | {_med(m['closes'])} / {max(m['closes'], default=0)} |",
                f"| time to success, simulated s (median, successes) | {_med(m['tts_sim'])} |",
                f"| time to success, wall s (median, successes) | {_med(m['tts_wall'])} |",
                f"| attempts lifted earlier but not held at checkpoint | {m['lifted_not_held']} |"]
        if m["compared"]:
            out.append(f"| detector vs oracle disagreement | {_frac(m['disagree'], m['compared'])} |")
        out += ["", "Stop reasons: " + ", ".join(f"{k} {v}" for k, v in m["stops"].most_common()), ""]
        if m["first_hist"] != m["first_ckpt"]:
            out += ["Note: the week-one stable-lift criterion (5 cm rise) and the checkpoint decision differ here. "
                    "The cuTAMP retraction lifts only ~5 cm, so the week-one criterion can be borderline at this "
                    "checkpoint; check the threshold against video before comparing.", ""]
    pairs = paired_rows(trials)
    if pairs:
        out += ["## Paired trials (same scene, variant, trial, probe)", "",
                "| scene/variant/trial | probe | one attempt | recovery |", "|---|---|---|---|"]
        out += [f"| {k[0]}/{k[1]}/{k[2]} | {k[3]} | {v.get('one', '-')} | {v.get('rec', '-')} |" for k, v in pairs]
        out.append("")
    if warnings:
        out += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    return "\n".join(out)


def paired_rows(trials: list[dict]) -> list[tuple[tuple, dict]]:
    rows: dict[tuple, dict] = defaultdict(dict)
    for t in trials:
        probe = (t.get("probe") or {}).get("label") or (t.get("probe") or {}).get("type", "baseline")
        k = (t.get("scene"), t.get("variant"), t.get("trial"), probe)
        slot = "one" if t.get("mode") == "checkpoint_one_attempt" else "rec"
        rows[k][slot] = "success" if t.get("final_success") else t.get("stop_reason", "?")
    return sorted(((k, v) for k, v in rows.items() if len(v) == 2), key=lambda kv: str(kv[0]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results_dir", type=Path)
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)
    trials, attempts, warnings = load(a.results_dir)
    if not trials:
        print(f"no trials in {a.results_dir / TRIALS_FILE}", file=sys.stderr)
        return 1
    text = render(trials, attempts, warnings, a.title or f"Checkpoint recovery: {a.results_dir.name}")
    (a.results_dir / "summary_checkpoint.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
