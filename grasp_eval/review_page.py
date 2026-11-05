"""review.html: every run's video next to its root cause, evidence and key measurements.

Written by summarize.py into each results directory. Open it in a browser on any machine
that has the results folder: videos and run files are linked by relative path, so copy the
whole results directory (videos are gitignored). If server outputs are copied under
server_outputs/<timestamp>/, their links also work without the original GPU filesystem.
Standard library only.
"""

from __future__ import annotations

import html
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from grasp_eval.paper_taxonomy import PAPER_CATEGORIES, SUCCESS

_CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#666; --card:#f5f5f7; --line:#ddd; --ok:#1a7f37; --bad:#c62828; --warn:#9a6700; }
@media (prefers-color-scheme: dark) { :root { --bg:#111; --fg:#eee; --muted:#aaa; --card:#1c1c1e; --line:#333;
  --ok:#4ade80; --bad:#f87171; --warn:#fbbf24; } }
body { background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1200px; padding:16px; }
h1 { margin:0 0 4px } .muted { color:var(--muted) } a { color:inherit }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin:14px 0 }
.grid { display:grid; grid-template-columns:minmax(0,3fr) minmax(0,2fr); gap:14px }
@media (max-width:800px) { .grid { grid-template-columns:1fr } }
video { width:100%; border-radius:6px; background:#000 }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:13px; border:1px solid currentColor }
.ok { color:var(--ok) } .bad { color:var(--bad) } .warn { color:var(--warn) }
table { border-collapse:collapse; width:100%; table-layout:fixed }
td { padding:3px 6px; border-top:1px solid var(--line); vertical-align:top; overflow-wrap:anywhere }
td:first-child { color:var(--muted); width:50% } ul { margin:4px 0 4px 18px; padding:0 }
"""


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _rel(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    original = Path(path)
    # Records retain their immutable GPU-machine paths. When a whole results
    # folder is copied to a laptop, resolve its local run artifacts first.
    if original.parent.parent.name == "runs":
        local = base / "runs" / original.parent.name / original.name
        if local.exists():
            return local.relative_to(base).as_posix()
    if "tiptop_server_outputs" in original.parts:
        index = original.parts.index("tiptop_server_outputs")
        local = base / "server_outputs" / Path(*original.parts[index + 1:])
        if local.exists():
            return local.relative_to(base).as_posix()
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # different drive on Windows
        return Path(path).as_posix()


def _fmt(v, unit: str = "", scale: float = 1.0, nd: int = 1) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int) and scale == 1.0:
        return f"{v}{unit}"
    if isinstance(v, (int, float)):
        return f"{v * scale:.{nd}f}{unit}"
    return str(v)


def measurements(attr: dict | None, rec: dict | None) -> list[tuple[str, str]]:
    """The few numbers that explain a run, in pipeline order (perception -> plan -> grasp -> carry)."""
    a, r = attr or {}, rec or {}
    rows: list[tuple[str, str | None]] = [
        ("Gemini goal (mapped to sim)", ", ".join("(" + ", ".join(x) + ")" for x in a.get("grounded_atoms_mapped", []))
         or None),
        ("goal translation correct", _fmt(a.get("goal_translation_correct"))),
        ("goal objects not detected", ", ".join(a.get("goal_objects_without_detection", [])) or None),
        ("plan picks first", a.get("first_pick_maps_to_sim_object")),
        ("perceived vs true position (xy)", _fmt(a.get("perception_xy_error_m"), " cm", 100)),
        ("grasp source", a.get("grasp_source")),
        ("cuTAMP failure", a.get("cutamp_failure_reason")),
        ("planning time", _fmt(r.get("planning_time_s"), " s")),
        ("object pushed before close", _fmt(a.get("target_pushed_before_close_m"), " cm", 100)),
        ("gripper vs object at close (xy)", _fmt(a.get("gripper_xy_offset_at_close_m"), " cm", 100)),
        ("joint tracking error at close", _fmt(a.get("joint_tracking_error_at_close_rad"), " rad", 1, 3)),
        ("closes", _fmt((r.get("score") or {}).get("num_close_commands"))),
        ("max lift near gripper", _fmt((r.get("score") or {}).get("max_rise_near_gripper_m"), " cm", 100)),
        ("longest hold", _fmt((r.get("score") or {}).get("longest_hold_s"), " s", 1, 2)),
        ("in-hand shift / rotation while carried",
         None if not (a.get("in_hand") or {}).get("carried") else
         f"{100 * a['in_hand']['max_in_hand_shift_m']:.1f} cm / {a['in_hand']['max_in_hand_rot_deg']:.0f} deg"
         + (" (dropped)" if a["in_hand"].get("dropped_before_release") else "")),
        ("task check", (a.get("task_check") or {}).get("reason")),
    ]
    for name, s in (a.get("shape") or {}).items():
        rows.append((f"shape of {name} (perceived/true per axis, IoU)",
                     f"{s.get('extent_ratio')} , {s.get('aabb_iou')}"))
    return [(k, v) for k, v in rows if v not in (None, "")]


def _run_card(row: dict, rec: dict | None, attr: dict | None, base: Path) -> str:
    ok = row["paper_category"] == SUCCESS
    top, sub, _ = PAPER_CATEGORIES.get(row["paper_category"], ("?", row["paper_category"], False))
    cls = "ok" if ok else ("warn" if row["paper_category"] in ("needs_review", "infrastructure") else "bad")
    r = rec or {}
    arts = r.get("artifacts") or {}
    video = _rel(arts.get("video"), base)
    run_dir = base / "runs" / row["run_id"]
    links = [("record", run_dir / "record.json"), ("attribution", run_dir / "attribution.json"),
             ("trace log", run_dir / "observed_eval.log"), ("console", run_dir / "console.log")]
    link_html = " · ".join(f'<a href="{_e(_rel(str(p), base))}">{_e(n)}</a>' for n, p in links)
    for name, key in (("server log", "server_log"), ("plan", "server_plan")):
        if key == "server_plan" and not r.get("server_success"):
            continue  # failed requests record an intended path, but create no plan
        if arts.get(key):
            link_html += f' · <a href="{_e(_rel(arts[key], base))}">{_e(name)}</a>'
    lift = row["stable_lift"]
    task = row["task_success"]
    evidence = "".join(f"<li>{_e(x)}</li>" for x in row.get("paper_evidence") or [])
    meas = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in measurements(attr, rec))
    media = (f'<video controls preload="none" src="{_e(video)}"></video>' if video
             else '<p class="muted">No video: nothing was executed (see the cause and the server log).</p>')
    return f"""
<div class="card" id="{_e(row['run_id'])}">
  <div><strong>{_e(row['run_id'])}</strong> <span class="badge {cls}">{_e(top)} → {_e(sub)}</span>
   <span class="muted">({_e(row['paper_category_source'])} label)</span></div>
  <div class="muted">scene {_e(r.get('scene'))}, variant {_e(row['variant'])}, condition {_e(row['condition'])},
   trial {_e(r.get('trial'))} · “{_e(r.get('instruction'))}” · target {_e(r.get('target_object'))}</div>
  <div>stable lift: <b class="{'ok' if lift else 'bad'}">{'yes' if lift else 'no'}</b> ·
   task success: <b class="{'ok' if task else ('muted' if task is None else 'bad')}">{'unchecked' if task is None else ('yes' if task else 'no')}</b>
   {f"({_e(row['task_success_source'])})" if row.get('task_success_source') else ''}</div>
  <div class="grid">
    <div>{media}</div>
    <div><b>Why</b><ul>{evidence or '<li>success</li>'}</ul><table>{meas}</table></div>
  </div>
  <div class="muted">{link_html}</div>
</div>"""


def render_review(results_dir: Path, rows: list[dict], records: dict[str, dict], attrs: dict[str, dict]) -> str:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["paper_category"]].append(r)
    order = [c for c in PAPER_CATEGORIES if c in by_cat and c != SUCCESS] + ([SUCCESS] if SUCCESS in by_cat else [])
    toc = "".join(
        f"<li><b>{_e(PAPER_CATEGORIES[c][0])} → {_e(PAPER_CATEGORIES[c][1])}</b>: {len(by_cat[c])} — "
        + ", ".join(f'<a href="#{_e(r["run_id"])}">{_e(r["run_id"])}</a>' for r in by_cat[c]) + "</li>"
        for c in order)
    cards = "".join(_run_card(r, records.get(r["run_id"]), attrs.get(r["run_id"]), results_dir)
                    for c in order for r in by_cat[c])
    n, lifts = len(rows), sum(r["stable_lift"] for r in rows)
    checked = [r for r in rows if r["task_success"] is not None]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Run review: {_e(results_dir.name)}</title>
<style>{_CSS}</style></head><body>
<h1>Run review: {_e(results_dir.name)}</h1>
<div class="muted">Generated {stamp} by grasp_eval.summarize. Causes are automatic first passes: confirm on the video,
then correct them in manual_labels.csv and re-run the summary. Details and denominators: summary.md.</div>
<div class="card">{n} scheduled runs · stable lift {lifts}/{n} ·
 task success {sum(bool(r['task_success']) for r in checked)}/{len(checked)} checked ({n - len(checked)} unchecked)
 <ul>{toc}</ul></div>
{cards}
</body></html>
"""


def write_review(results_dir: Path, rows: list[dict], records: dict[str, dict], attrs: dict[str, dict]) -> Path:
    out = results_dir / "review.html"
    out.write_text(render_review(results_dir, rows, records, attrs), encoding="utf-8")
    return out
