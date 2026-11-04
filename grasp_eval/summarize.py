"""Summarize a results directory into counts with explicit denominators.

Usage:  python -m grasp_eval.summarize results/pilot_scene1 [--out summary.md]

Reads:
  schedule.jsonl       every run the pilot scheduled (written before launch)
  runs.jsonl           one record per run that got far enough to write one
  manual_labels.csv    optional reviewer overrides, columns:
                       run_id,stable_lift,failure_category,task_success,note[,paper_category]
  attribution.jsonl    optional per-run stage attribution (attribute_run.py), including the
                       automatic task check and the paper-taxonomy label
  skipped.jsonl        optional replays never scheduled because their source had no plan
A scheduled run with no record counts as an infrastructure failure. Probe suites add
per-condition and per-group tables (e.g. success rate against displacement distance).
The paper-taxonomy section mirrors the TiPToP paper's failure-analysis figure and also
writes paper_taxonomy.json (Sankey nodes and links) for plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from grasp_eval.paper_taxonomy import EXCLUDED_FROM_PAPER_DENOMINATOR, PAPER_CATEGORIES, SUCCESS, classify
from grasp_eval.records import FAILURE_LABELS, read_jsonl
from grasp_eval.review_page import write_review

_BOOL = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}


def _parse_bool(value: str, field: str, run_id: str) -> bool | None:
    v = (value or "").strip().lower()
    if not v:
        return None
    if v not in _BOOL:
        raise ValueError(f"manual_labels.csv: bad {field}={value!r} for {run_id}")
    return _BOOL[v]


def load_manual_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_id = row["run_id"].strip()
            cat = (row.get("failure_category") or "").strip()
            if cat and cat not in FAILURE_LABELS:
                raise ValueError(f"manual_labels.csv: unknown failure_category {cat!r} for {run_id}")
            paper = (row.get("paper_category") or "").strip()
            if paper and paper not in PAPER_CATEGORIES:
                raise ValueError(f"manual_labels.csv: unknown paper_category {paper!r} for {run_id} "
                                 f"(choose from {sorted(PAPER_CATEGORIES)})")
            labels[run_id] = {
                "stable_lift": _parse_bool(row.get("stable_lift", ""), "stable_lift", run_id),
                "failure_category": cat or None,
                "task_success": _parse_bool(row.get("task_success", ""), "task_success", run_id),
                "paper_category": paper or None,
                "note": (row.get("note") or "").strip(),
            }
    return labels


def build_rows(results_dir: Path) -> tuple[list[dict], list[str]]:
    schedule = read_jsonl(results_dir / "schedule.jsonl")
    records = {}
    warnings = []
    for rec in read_jsonl(results_dir / "runs.jsonl"):
        if rec["run_id"] in records:
            warnings.append(f"duplicate record for {rec['run_id']}; using the last one")
        records[rec["run_id"]] = rec
    labels = load_manual_labels(results_dir / "manual_labels.csv")
    attrs = load_attributions(results_dir)

    scheduled_ids = {s["run_id"] for s in schedule}
    for run_id in records.keys() - scheduled_ids:
        warnings.append(f"record {run_id} is not in schedule.jsonl; excluded from denominators")
    for run_id in labels.keys() - scheduled_ids:
        warnings.append(f"manual label for unknown run {run_id} ignored")

    rows = []
    for s in schedule:
        rec = records.get(s["run_id"])
        attr = attrs.get(s["run_id"])
        auto_task = ((attr or {}).get("task_check") or {}).get("task_success")
        row = {
            "run_id": s["run_id"],
            "variant": s["variant"],
            "condition": s.get("condition", "baseline"),
            "group": s.get("group", s.get("condition", "baseline")),
            "target_source": rec.get("target_source") if rec else None,
            "has_record": rec is not None,
            "plan_returned": bool(rec and rec.get("server_success")),
            "valid_executed": bool(
                rec and rec.get("server_success") and rec.get("executed_steps", 0) > 0
                and rec.get("harness_exception") is None
            ),
            "stable_lift": bool(rec and rec.get("stable_lift")),
            "failure_category": rec.get("failure_category") if rec else "infrastructure",
            "task_success": auto_task,
            "task_success_source": "auto" if auto_task is not None else None,
            "planning_time_s": rec.get("planning_time_s") if rec else None,
            "overridden": False,
        }
        if rec is None:
            row["failure_category"] = "infrastructure"
        lab = labels.get(s["run_id"])
        if lab:
            if lab["stable_lift"] is not None:
                row["stable_lift"] = lab["stable_lift"] and row["valid_executed"]
                if lab["stable_lift"] and not row["valid_executed"]:
                    warnings.append(f"{s['run_id']}: stable_lift override ignored, no valid executed plan")
            if lab["failure_category"] is not None:
                row["failure_category"] = lab["failure_category"]
            if lab["task_success"] is not None:
                row["task_success"] = lab["task_success"]
                row["task_success_source"] = "manual"
            row["overridden"] = True
        if row["stable_lift"]:
            row["failure_category"] = None
        elif row["failure_category"] is None:
            row["failure_category"] = "unknown"
        manual_outcome = bool(lab and (lab["task_success"] is not None or lab["stable_lift"] is not None))
        if attr is not None and "paper_category" in attr and not manual_outcome:
            paper = {"paper_category": attr["paper_category"], "evidence": attr.get("evidence", [])}
        else:  # no attribution, or a manual outcome label that can change the classification
            paper = classify(dict(rec, stable_lift=row["stable_lift"]) if rec else None, attr,
                             task_success=row["task_success"])
        row["paper_category"], row["paper_evidence"] = paper["paper_category"], paper["evidence"]
        row["paper_category_source"] = "auto"
        if lab and lab["paper_category"]:
            row["paper_category"], row["paper_category_source"] = lab["paper_category"], "manual"
        rows.append(row)
    return rows, warnings


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion k/n (well behaved at small n and at 0 or n)."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _ci_text(k: int, n: int) -> str:
    ci = wilson_ci(k, n)
    return f"[{100 * ci[0]:.0f}-{100 * ci[1]:.0f}%]" if ci else ""


def _frac(k: int, n: int) -> str:
    return f"{k}/{n} ({100.0 * k / n:.0f}%)" if n else f"{k}/0 (n/a)"


def _frac_ci(k: int, n: int) -> str:
    return f"{_frac(k, n)}, 95% CI {_ci_text(k, n)}" if n else _frac(k, n)


def render(rows: list[dict], warnings: list[str], title: str) -> str:
    n = len(rows)
    lifts = sum(r["stable_lift"] for r in rows)
    plans = sum(r["plan_returned"] for r in rows)
    valid = sum(r["valid_executed"] for r in rows)
    records = sum(r["has_record"] for r in rows)
    labeled_task = [r for r in rows if r["task_success"] is not None]
    times = [r["planning_time_s"] for r in rows if r["planning_time_s"] is not None]

    out = [f"# {title}", ""]
    out.append(f"- Scheduled trials: {n} (records written: {records})")
    out.append(f"- Plan returned by server: {_frac(plans, n)}")
    out.append(f"- Valid executed plans: {_frac(valid, n)}")
    out.append(f"- **Stable lift, overall (per scheduled trial): {_frac_ci(lifts, n)}**")
    out.append(f"- **Stable lift, conditional (per valid executed plan): {_frac_ci(lifts, valid)}**")
    out.append(
        f"- Final task success (runs with an automatic check or manual label): "
        f"{_frac_ci(sum(bool(r['task_success']) for r in labeled_task), len(labeled_task))}; "
        f"unchecked: {n - len(labeled_task)}"
    )
    if times:
        out.append(f"- Planning time (s): median {sorted(times)[len(times) // 2]:.1f}, "
                   f"min {min(times):.1f}, max {max(times):.1f}, n={len(times)}")
    out.append(f"- Runs with manual label overrides: {sum(r['overridden'] for r in rows)}")

    out += ["", "## By variant", "", "| variant | scheduled | plan returned | valid executed | stable lift |",
            "|---|---|---|---|---|"]
    by_var = defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)
    for v in sorted(by_var):
        rs = by_var[v]
        out.append(f"| {v} | {len(rs)} | {sum(r['plan_returned'] for r in rs)} | "
                   f"{sum(r['valid_executed'] for r in rs)} | {sum(r['stable_lift'] for r in rs)} |")

    for key, heading in (("condition", "By condition"), ("group", "By group (pooled conditions)")):
        values = list(dict.fromkeys(r[key] for r in rows))  # first-seen order
        if len(values) <= 1 or (key == "group" and values == list(dict.fromkeys(r["condition"] for r in rows))):
            continue
        out += ["", f"## {heading}", "",
                f"| {key} | scheduled | valid executed | stable lift (overall) | stable lift (conditional) |",
                "|---|---|---|---|---|"]
        for val in values:
            rs = [r for r in rows if r[key] == val]
            k, v = sum(r["stable_lift"] for r in rs), sum(r["valid_executed"] for r in rs)
            out.append(f"| {val} | {len(rs)} | {v} | {_frac(k, len(rs))} | {_frac(k, v)} |")

    out += ["", f"## Failure categories (denominator: {n} scheduled trials)", "",
            "| category | count |", "|---|---|"]
    counts = Counter(r["failure_category"] for r in rows if r["failure_category"])
    for cat in FAILURE_LABELS:
        if counts.get(cat):
            out.append(f"| {cat} | {counts[cat]} |")
    out.append(f"| (stable lift) | {lifts} |")

    fallback = [r["run_id"] for r in rows if r["target_source"] in ("nearest_at_close", "unresolved")]
    if fallback:
        warnings = warnings + [f"{len(fallback)} run(s) scored a target not named by the plan "
                               f"(nearest object or unresolved); review: {', '.join(fallback)}"]
    if warnings:
        out += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
    return "\n".join(out) + "\n"


_ATTR_METRICS = (
    ("perception_xy_error_m", "perceived vs true object xy (m)"),
    ("gripper_xy_offset_at_close_m", "gripper vs object xy at close (m)"),
    ("target_pushed_before_close_m", "object pushed before close (m)"),
    ("joint_tracking_error_at_close_rad", "joint tracking error at close (rad)"),
)


def _median(xs: list[float]) -> str:
    return f"{statistics.median(xs):.3f} (n={len(xs)})" if xs else "n/a"


def render_attribution(rows: list[dict], attrs: dict[str, dict]) -> str:
    """Stage-attribution section: metric medians by outcome, stage hints, grounding and timing."""
    executed = [r for r in rows if r["valid_executed"] and r["run_id"] in attrs]
    if not executed:
        return ""
    out = ["", "## Stage attribution (valid executed runs with attribution)", "",
           "| metric | median, stable lift | median, failed |", "|---|---|---|"]
    for key, label in _ATTR_METRICS:
        ok = [attrs[r["run_id"]][key] for r in executed if r["stable_lift"] and attrs[r["run_id"]].get(key) is not None]
        bad = [attrs[r["run_id"]][key] for r in executed if not r["stable_lift"] and attrs[r["run_id"]].get(key) is not None]
        out.append(f"| {label} | {_median(ok)} | {_median(bad)} |")

    hints = Counter(attrs[r["run_id"]].get("stage_hint") for r in executed if not r["stable_lift"])
    if hints:
        out += ["", "Heuristic stage hints for failed runs (confirm on video):", ""]
        out += [f"- {h or 'none'}: {c}" for h, c in hints.most_common()]

    mism = [r["run_id"] for r in executed if attrs[r["run_id"]].get("first_pick_matches_target") is False]
    out += ["", f"- Plan picked a different object than the scored target: {len(mism)}"
            + (f" ({', '.join(mism)})" if mism else "")]
    unmatched = [r["run_id"] for r in executed
                 if attrs[r["run_id"]].get("first_pick_found_in_perceived_scene") is False]
    if unmatched:
        out.append(f"- Plan's first Pick object not found among perceived objects (no perception error "
                   f"computed): {', '.join(unmatched)}")
    missed = Counter(o for r in executed for o in attrs[r["run_id"]].get("sim_objects_without_nearby_detection", []))
    if missed:
        out.append("- Sim objects with no perceived object within 5 cm: "
                   + ", ".join(f"{k} ({v} runs)" for k, v in missed.most_common()))
    timings = {(tuple(t.get("trajectory_dts_s", [])), tuple(t.get("execution_time_scale", [])))
               for t in (attrs[r["run_id"]].get("plan_timing", {}) for r in executed)}
    for dts, scale in sorted(timings):
        if dts:
            out.append(f"- Plan trajectory dt {list(dts)} s; executed/planned duration {list(scale)} "
                       "(1.0 means the client's 50 Hz assumption holds)")
    return "\n".join(out) + "\n"


# Counts read off the TiPToP paper's failure-analysis figure (173 real-robot trials).
PAPER_REFERENCE_COUNTS = {
    "Success": 118, "Failure": 55,
    "Grasp Failure": 31, "M2T2 Unstable Grasp": 20, "Heuristic Unstable Grasp": 11,
    "Scene Completion Failure": 13, "Partial Observability": 13,
    "Gemini VLM Failure": 6, "Object Detection Failure": 5, "Goal Translation Failure": 1,
    "cuTAMP Failure": 5, "cuRobo Motion Planning Failure": 3, "Rigid Grasp Assumption": 2,
}


def _paper_comparable(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["paper_category"] not in EXCLUDED_FROM_PAPER_DENOMINATOR]


def paper_sankey(rows: list[dict]) -> dict:
    """Nodes and links in the shape of the paper's figure, over paper-comparable trials."""
    comparable = _paper_comparable(rows)
    links: Counter = Counter()
    for r in comparable:
        top, sub, _ = PAPER_CATEGORIES[r["paper_category"]]
        if r["paper_category"] == SUCCESS:
            links[("All Trials", "Success")] += 1
            continue
        links[("All Trials", "Failure")] += 1
        links[("Failure", top)] += 1
        links[(top, sub)] += 1
    return {
        "comparable_trials": len(comparable),
        "excluded": dict(Counter(r["paper_category"] for r in rows if r not in comparable)),
        "nodes": list(dict.fromkeys(n for pair in links for n in pair)),
        "links": [{"source": a, "target": b, "value": v} for (a, b), v in links.items()],
        "paper_reference_counts": PAPER_REFERENCE_COUNTS,
    }


def render_paper_taxonomy(rows: list[dict]) -> str:
    """Counts per node of the paper's failure-analysis figure, ours next to the paper's."""
    if not rows:
        return ""
    comparable = _paper_comparable(rows)
    n = len(comparable)
    fails = [r for r in comparable if r["paper_category"] != SUCCESS]
    out = ["", "## Paper failure taxonomy (TiPToP failure-analysis figure)", "",
           f"Denominator: {n} paper-comparable trials = {len(rows)} scheduled minus {len(rows) - n} excluded "
           "(infrastructure or needs_review). Automatic labels are first passes: confirm on video and in the "
           "server log, and override with `paper_category` in manual_labels.csv. The paper's 173 trials ran on "
           "a real robot, so proportions in simulation are not expected to match. With a few dozen trials the "
           "95% intervals are wide: \"paper % inside our CI\" means only that this run can't tell the two apart.", "",
           "| node | ours | % of our trials | % of our failures | paper (of 173) | in paper figure "
           "| our 95% CI | paper % inside our CI |",
           "|---|---|---|---|---|---|---|---|"]

    def line(name: str, k: int, in_paper: bool, indent: int = 0, share_of_failures: bool = True) -> None:
        pct = f"{100 * k / n:.0f}%" if n else "n/a"
        fpct = f"{100 * k / len(fails):.0f}%" if share_of_failures and fails else ""
        ref = PAPER_REFERENCE_COUNTS.get(name, "")
        label = "&nbsp;&nbsp;" * indent + name
        ci = wilson_ci(k, n)
        inside = "" if ref == "" or ci is None else ("yes" if ci[0] <= ref / 173 <= ci[1] else "no")
        out.append(f"| {label} | {k} | {pct} | {fpct} | {ref} | {'yes' if in_paper else 'no'} "
                   f"| {_ci_text(k, n)} | {inside} |")

    line("Success", n - len(fails), True, share_of_failures=False)
    line("Failure", len(fails), True, share_of_failures=False)
    by_top: dict[str, Counter] = defaultdict(Counter)
    for r in fails:
        by_top[PAPER_CATEGORIES[r["paper_category"]][0]][r["paper_category"]] += 1
    for top in dict.fromkeys(v[0] for v in PAPER_CATEGORIES.values()):
        if top not in by_top:
            continue
        subs = by_top[top]
        line(top, sum(subs.values()), "(not in paper)" not in top, indent=1)
        for label in (k for k in PAPER_CATEGORIES if k in subs):
            _, sub, in_paper = PAPER_CATEGORIES[label]
            line(sub, subs[label], in_paper, indent=2)

    excluded = Counter(r["paper_category"] for r in rows if r not in comparable)
    if excluded:
        out += ["", "Excluded from the denominator: " + ", ".join(f"{k} {v}" for k, v in excluded.items())]
    out.append("")
    out.append(f"- Paper labels set manually: {sum(r['paper_category_source'] == 'manual' for r in rows)}")
    out.append(f"- Task success source: automatic {sum(r['task_success_source'] == 'auto' for r in rows)}, "
               f"manual {sum(r['task_success_source'] == 'manual' for r in rows)}, "
               f"unchecked {sum(r['task_success'] is None for r in rows)}")
    out += ["", "| run | paper category | source | evidence |", "|---|---|---|---|"]
    for r in rows:
        if r["paper_category"] != SUCCESS:
            ev = "; ".join(r["paper_evidence"]).replace("|", "/")
            out.append(f"| {r['run_id']} | {r['paper_category']} | {r['paper_category_source']} | {ev} |")
    return "\n".join(out) + "\n"


def load_attributions(results_dir: Path) -> dict[str, dict]:
    return {a["run_id"]: a for a in read_jsonl(results_dir / "attribution.jsonl")}


MERGED_FILES = ("schedule.jsonl", "runs.jsonl", "attribution.jsonl", "skipped.jsonl", "exit_codes.tsv")


def merge_shards(shard_dirs: list[Path], into: Path) -> None:
    """Combine shard results (run_suite --shard) into one directory for a single summary.

    The merged files are regenerated on every call. Put manual labels in <into>/manual_labels.csv;
    shard label files are ignored, so each run is labeled in one place.
    """
    into.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}
    for d in shard_dirs:
        for s in read_jsonl(d / "schedule.jsonl"):
            if s["run_id"] in seen:
                raise ValueError(f"run {s['run_id']} is scheduled in both {seen[s['run_id']]} and {d}")
            seen[s["run_id"]] = d
    for name in MERGED_FILES:
        parts = [(d / name).read_text() for d in shard_dirs if (d / name).exists()]
        (into / name).write_text("".join(p if p.endswith("\n") or not p else p + "\n" for p in parts))
    (into / "merged_from.json").write_text(json.dumps([str(d) for d in shard_dirs], indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results_dir", type=Path, nargs="+", help="one results directory, or several shards with --into")
    p.add_argument("--into", type=Path, default=None, help="merge the given shard directories into this one")
    p.add_argument("--out", type=Path, default=None, help="default: <results_dir>/summary.md")
    args = p.parse_args(argv)
    if len(args.results_dir) > 1 or args.into:
        if not args.into:
            p.error("several results directories need --into <merged dir>")
        merge_shards(args.results_dir, args.into)
        args.results_dir = args.into
    else:
        args.results_dir = args.results_dir[0]
    rows, warnings = build_rows(args.results_dir)
    skipped = read_jsonl(args.results_dir / "skipped.jsonl")
    if skipped:
        warnings.append(f"{len(skipped)} replay run(s) not scheduled because their source run produced "
                        "no plan (see skipped.jsonl); they are outside all denominators")
    text = render(rows, warnings, f"Summary: {args.results_dir.name}")
    text += render_attribution(rows, load_attributions(args.results_dir))
    text += render_paper_taxonomy(rows)
    (args.results_dir / "paper_taxonomy.json").write_text(json.dumps(paper_sankey(rows), indent=2))
    records = {r["run_id"]: r for r in read_jsonl(args.results_dir / "runs.jsonl")}
    review = write_review(args.results_dir, rows, records, load_attributions(args.results_dir))
    text += f"\nPer-run videos with their causes: {review.name} (open in a browser).\n"
    out = args.out or args.results_dir / "summary.md"
    out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
