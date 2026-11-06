"""Paired task-level recovery suite: one recovery trial per baseline trial (docs/task_recovery.md).

    python3 -m grasp_eval.task_suite --baseline results/baseline_panda_20261106 --tiptop-dir $TIPTOP_DIR \
        --lift-mode reverse_approach --out-name task_reverse_20261106 [--scenes 1 2] [--limit 5] [--resume] \
        [--eval-args ...]      # --eval-args must be last: it passes everything after it to the trial

Each trial reuses its baseline trial's scene, variant, seed, instruction and goal, and attempt 0 replays
the baseline's saved plan (`<server_save_dir>/tiptop_plan.json`). Only a retry calls TiPToP (and Gemini),
so the comparison is paired: the two trials differ by the checkpoint and its retries. A baseline trial
without a plan (a planning failure) has nothing to replay; it is listed in skipped.jsonl and counted as
a failure in both arms by the summary. `--max-retries 0` runs the checkpoint-only control.
Stops before starting another simulator after an API quota error, like run_suite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_eval.checkpoint import is_api_quota_error  # noqa: E402
from grasp_eval.records import read_jsonl  # noqa: E402
from grasp_eval.recovery_records import TASK_TRIALS_FILE  # noqa: E402
from grasp_eval.run_suite import run_process, wait_for_servers  # noqa: E402
from grasp_eval.task_checkpoint import LIFT_MODES  # noqa: E402

TASK_PY = PROJECT_ROOT / "grasp_eval" / "task_recovery_eval.py"


def baseline_trials(baseline: Path, scenes_cfg: dict, scenes: list[int] | None = None,
                    runs: list[str] | None = None) -> list[dict]:
    """Baseline records in schedule order, with the goal and plan path the recovery trial needs."""
    out = []
    for r in read_jsonl(baseline / "runs.jsonl"):
        if scenes and int(r["scene"]) not in scenes:
            continue
        if runs and r["run_id"] not in runs:
            continue
        sc = scenes_cfg.get(str(r["scene"]), {})
        plan = None
        if r.get("server_success") and r.get("server_save_dir"):
            plan = str(Path(r["server_save_dir"]) / "tiptop_plan.json")
        target = r.get("target_object")
        if target in (None, "", "from-plan"):
            target = sc.get("target_object")
        out.append({"run_id": r["run_id"], "scene": int(r["scene"]), "variant": int(r["variant"]),
                    "trial": int(r.get("trial", 0)), "seed": r.get("seed"), "instruction": r.get("instruction")
                    or sc.get("instruction"), "goal": r.get("goal") or sc.get("goal"), "target_object": target,
                    "plan": plan})
    return out


def trial_id(t: dict, a) -> str:
    return f"{t['run_id']}_{'ckonly' if a.max_retries == 0 else 'rec'}_{a.lift_mode}"


def command(t: dict, a, out: Path, sim: Path, tiptop: Path, baseline: Path) -> list[str]:
    cmd = ["uv", "run", "python", str(TASK_PY), "--sim-root", str(sim), "--tiptop-dir", str(tiptop),
           "--scene", str(t["scene"]), "--variant", str(t["variant"]), "--trial", str(t["trial"]),
           "--instruction", t["instruction"], "--trial-id", trial_id(t, a), "--out-dir", str(out),
           "--lift-mode", a.lift_mode, "--max-retries", str(a.max_retries), "--plan-json", t["plan"],
           "--paired-baseline-run", t["run_id"], "--paired-baseline-dir", str(baseline),
           "--ws-host", a.ws_host, "--ws-port", str(a.ws_port)]
    if t.get("seed") is not None:
        cmd += ["--seed", str(t["seed"])]
    if t.get("goal"):
        cmd += ["--goal-json", json.dumps(t["goal"])]
    if t.get("target_object"):
        cmd += ["--target-object", t["target_object"]]
    if a.provenance_file:
        cmd += ["--provenance-file", str(Path(a.provenance_file).resolve())]
    return cmd + a.eval_args


def quota_error(out: Path, tid: str) -> str | None:
    path = out / "trials" / tid / "record.json"
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    for p in rec.get("plans") or []:
        if not p.get("server_success", True) and is_api_quota_error(p.get("server_error")):
            return str(p.get("server_error"))
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", type=Path, required=True, help="baseline results dir (runs.jsonl)")
    p.add_argument("--tiptop-dir", type=Path, help="parent of tiptop/ and droid-sim-evals/ (required unless --dry-run)")
    p.add_argument("--out-name", default=None)
    p.add_argument("--lift-mode", choices=LIFT_MODES, default="reverse_approach")
    p.add_argument("--max-retries", type=int, default=2, help="0 = checkpoint-only control")
    p.add_argument("--scenes", type=int, nargs="+", default=None)
    p.add_argument("--runs", nargs="+", default=None, help="only these baseline run ids")
    p.add_argument("--limit", type=int, default=None, help="at most this many trials (smoke tests)")
    p.add_argument("--scenes-config", default=str(PROJECT_ROOT / "configs" / "scenes.json"))
    p.add_argument("--provenance-file", default=None)
    p.add_argument("--ws-host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--m2t2-url", default="http://localhost:8123")
    p.add_argument("--no-health-check", action="store_true")
    p.add_argument("--server-wait-s", type=float, default=600)
    p.add_argument("--run-timeout-s", type=float, default=2400)
    p.add_argument("--max-hours", type=float, default=None, help="schedule no new trial after this long")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--eval-args", nargs=argparse.REMAINDER, default=[], help="passed to task_recovery_eval.py")
    a = p.parse_args(argv)
    if not a.dry_run and not a.tiptop_dir:
        p.error("--tiptop-dir is required unless --dry-run")
    a.out_name = a.out_name or f"task_{'ckonly' if a.max_retries == 0 else 'rec'}_{a.lift_mode}"
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    baseline = a.baseline.resolve()
    scenes_cfg = json.loads(Path(a.scenes_config).read_text())
    trials = baseline_trials(baseline, scenes_cfg, a.scenes, a.runs)
    if a.limit is not None:
        trials = trials[:a.limit]
    out = PROJECT_ROOT / "results" / a.out_name
    runnable = [t for t in trials if t["plan"]]
    if a.dry_run:
        print(f"{len(runnable)} paired trials ({len(trials) - len(runnable)} baseline trials without a plan) -> {out}")
        for t in trials[:10]:
            print(f"  {trial_id(t, a)}  seed={t['seed']}  plan={'yes' if t['plan'] else 'NONE (skipped)'}")
        return 0
    sim = (a.tiptop_dir / "droid-sim-evals").resolve()
    tiptop = (a.tiptop_dir / "tiptop").resolve()
    scheduled = {s["trial_id"] for s in read_jsonl(out / "schedule.jsonl")}
    if scheduled and not a.resume:
        print(f"ERROR: {out} already has a schedule; use --resume or a new --out-name.", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                               indent=2))
    done = {r["trial_id"] for r in read_jsonl(out / TASK_TRIALS_FILE)}
    exit_file = out / "exit_codes.tsv"
    if exit_file.exists():
        done |= {line.split("\t")[0] for line in exit_file.read_text().splitlines() if line.strip()}
    skipped = {s["run_id"] for s in read_jsonl(out / "skipped.jsonl")}
    t_start = time.monotonic()
    stopped = None
    for i, t in enumerate(trials, 1):
        tid = trial_id(t, a)
        if tid in done or t["run_id"] in skipped:
            continue
        if not t["plan"]:
            _append(out / "skipped.jsonl", {"run_id": t["run_id"], "trial_id": tid,
                                           "reason": "baseline trial has no plan to replay (planning failure)"})
            continue
        if a.max_hours is not None and time.monotonic() - t_start > 3600 * a.max_hours:
            stopped = f"--max-hours {a.max_hours} reached"
            break
        if not a.no_health_check and not wait_for_servers(a):
            stopped = "a server is down"
            break
        if tid not in scheduled:
            _append(out / "schedule.jsonl", {"trial_id": tid, **{k: t[k] for k in ("run_id", "scene", "variant",
                                                                                   "trial", "seed")}})
        print(f"==> [{i}/{len(trials)}] {tid}", flush=True)
        t0 = time.monotonic()
        (out / "trials" / tid).mkdir(parents=True, exist_ok=True)
        code = run_process(command(t, a, out, sim, tiptop, baseline), sim, out / "trials" / tid / "console.log",
                           a.run_timeout_s)
        with open(exit_file, "a") as f:
            f.write(f"{tid}\t{code}\n")
        print(f"    exit code {code} ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
        if quota_error(out, tid):
            stopped = "upstream API quota/rate limit; resolve it before --resume"
            break
    from grasp_eval.task_summary import main as summarize

    rc = summarize([str(out), "--baseline", str(baseline)]) if (out / TASK_TRIALS_FILE).exists() else 1
    if stopped:
        print(f"Stopped early: {stopped}. Continue with --resume.", file=sys.stderr)
        return 3
    return rc


def _append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
