"""Schedule paired week-two trials: one-attempt checkpoint control vs two-attempt recovery.

    python3 -m grasp_eval.recovery_suite --tiptop-dir $TIPTOP_DIR --target-object <sim name> \
        --variants 0 1 2 --trials 2 --paired --out-name week2_oracle_scene1

For every (variant, trial, condition) it runs the one-attempt control first (fresh plan),
then the recovery mode. With --paired, the recovery trial replays the control's first plan for
its attempt 0 (same scene and disturbance), so only the retry gets a new plan. Client seeds do
not seed Gemini, M2T2 or cuTAMP, so without --paired the two first plans can differ.

Each trial is its own process (grasp_eval/recovery_eval.py) with an outer timeout, as in
week one; week one's run_suite.py is reused, not modified. Results go to results/<out-name>/,
never into a week-one results directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_eval.records import read_jsonl  # noqa: E402
from grasp_eval.recovery_records import TRIALS_FILE  # noqa: E402
from grasp_eval.run_suite import DEFAULT_INSTRUCTION, run_process, wait_for_servers  # noqa: E402

RECOVERY_PY = PROJECT_ROOT / "grasp_eval" / "recovery_eval.py"


@dataclass
class Trial:
    scene: int
    variant: int
    trial: int
    condition: str
    max_attempts: int
    seed: int
    flags: dict = field(default_factory=dict)
    replay_from: "Trial | None" = None
    trial_id: str = ""

    @property
    def mode(self) -> str:
        return "checkpoint_one_attempt" if self.max_attempts == 1 else f"checkpoint_recovery_{self.max_attempts}"


def parse_condition(text: str) -> tuple[str, dict]:
    """'baseline' or 'disp:DX_M,DY_M,YAW_DEG' (displacement applied once, before attempt 1)."""
    if text == "baseline":
        return text, {}
    if text.startswith("disp:"):
        dx, dy, yaw = (float(v) for v in text[5:].split(","))
        return (f"disp_{100 * dx:+g}cm_{100 * dy:+g}cm_{yaw:+g}deg",
                {"displace-dx-m": dx, "displace-dy-m": dy, "displace-yaw-deg": yaw})
    raise ValueError(f"bad condition {text!r}; use 'baseline' or 'disp:dx,dy,yaw'")


def build_trials(a) -> list[Trial]:
    out = []
    for v in a.variants:
        for t in range(a.trials):
            for i, cond in enumerate(a.conditions):
                name, flags = parse_condition(cond)
                seed = a.seed_base + 1000 * v + 10 * t + i
                one = Trial(a.scene, v, t, name, 1, seed, flags)
                rec = Trial(a.scene, v, t, name, a.max_attempts, seed, flags, replay_from=one if a.paired else None)
                out += [one, rec]
    for r in out:
        r.trial_id = f"s{r.scene}v{r.variant}t{r.trial}_{r.condition}_a{r.max_attempts}"
    return out


def first_plan(out: Path, source: Trial) -> str | None:
    path = out / "trials" / source.trial_id / "attempts.json"
    if not path.exists():
        return None
    attempts = json.loads(path.read_text())
    if not attempts or not attempts[0].get("server_success"):
        return None
    plan = (attempts[0].get("artifacts") or {}).get("server_plan")
    return plan if plan and Path(plan).exists() else None


def schedule_entry(r: Trial, a) -> dict:
    probe_type = "displacement" if r.flags else "baseline"
    if r.replay_from is not None:
        probe_type = "replay" + ("+displacement" if r.flags else "")
    return {"trial_id": r.trial_id, "scene": r.scene, "variant": r.variant, "trial": r.trial, "seed": r.seed,
            "mode": r.mode, "detector": "oracle-state", "instruction": a.instruction,
            "probe": {"type": probe_type, "label": r.condition},
            "replay_from": r.replay_from.trial_id if r.replay_from else None}


def command(r: Trial, a, out: Path, sim: Path, tiptop: Path, plan: str | None) -> list[str]:
    cmd = ["uv", "run", "python", str(RECOVERY_PY), "--sim-root", str(sim), "--tiptop-dir", str(tiptop),
           "--scene", str(r.scene), "--variant", str(r.variant), "--trial", str(r.trial), "--seed", str(r.seed),
           "--instruction", a.instruction, "--target-object", a.target_object, "--trial-id", r.trial_id,
           "--out-dir", str(out), "--max-attempts", str(r.max_attempts), "--detector", "oracle-state",
           "--max-sim-s", str(a.max_sim_s), "--max-wall-s", str(a.max_wall_s),
           "--ws-host", a.ws_host, "--ws-port", str(a.ws_port), "--probe-label", r.condition]
    if a.target_label:
        cmd += ["--target-label", a.target_label]
    if a.provenance_file:
        cmd += ["--provenance-file", str(Path(a.provenance_file).resolve())]
    for k, v in r.flags.items():
        cmd += [f"--{k}", str(v)]
    if plan:
        cmd += ["--plan-json", plan]
    return cmd + a.eval_args


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tiptop-dir", type=Path, help="parent of tiptop/ and droid-sim-evals/ (required unless --dry-run)")
    p.add_argument("--out-name", default=None)
    p.add_argument("--scene", type=int, default=1)
    p.add_argument("--variants", type=int, nargs="+", default=[0])
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--conditions", nargs="+", default=["baseline"],
                   help="'baseline' and/or 'disp:DX_M,DY_M,YAW_DEG', e.g. disp:0.02,0,0")
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--target-object", required=True)
    p.add_argument("--target-label", default=None)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--max-sim-s", type=float, default=180.0)
    p.add_argument("--max-wall-s", type=float, default=600.0)
    p.add_argument("--paired", action="store_true", help="recovery attempt 0 replays the control's first plan")
    p.add_argument("--seed-base", type=int, default=5000)
    p.add_argument("--provenance-file", default=None)
    p.add_argument("--ws-host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--m2t2-url", default="http://localhost:8123")
    p.add_argument("--no-health-check", action="store_true")
    p.add_argument("--server-wait-s", type=float, default=600)
    p.add_argument("--run-timeout-s", type=float, default=1500, help="outer guard per trial process")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--eval-args", nargs=argparse.REMAINDER, default=[], help="passed through to recovery_eval.py")
    a = p.parse_args(argv)
    if a.max_attempts < 2:
        p.error("--max-attempts is the recovery mode's budget; the one-attempt control is always included")
    for c in a.conditions:
        try:
            parse_condition(c)
        except ValueError as e:
            p.error(str(e))
    if not a.dry_run and not a.tiptop_dir:
        p.error("--tiptop-dir is required unless --dry-run")
    a.out_name = a.out_name or f"week2_checkpoint_scene{a.scene}"
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    trials = build_trials(a)
    out = PROJECT_ROOT / "results" / a.out_name
    if a.dry_run:
        print(f"{len(trials)} trials -> {out}")
        for r in trials:
            src = f"  (attempt 0 replays {r.replay_from.trial_id})" if r.replay_from else ""
            print(f"  {r.trial_id}  seed={r.seed}  {r.flags}{src}")
        return 0
    sim = (a.tiptop_dir / "droid-sim-evals").resolve()
    tiptop = (a.tiptop_dir / "tiptop").resolve()
    scheduled = {s["trial_id"] for s in read_jsonl(out / "schedule.jsonl")}
    if scheduled and not a.resume:
        print(f"ERROR: {out} already has a schedule; use --resume or a new --out-name.", file=sys.stderr)
        return 1
    done = {t["trial_id"] for t in read_jsonl(out / TRIALS_FILE)}
    exit_file = out / "exit_codes.tsv"
    if exit_file.exists():
        done |= {line.split("\t")[0] for line in exit_file.read_text().splitlines() if line.strip()}
    skipped = {s["trial_id"] for s in read_jsonl(out / "skipped.jsonl")}
    (out / "trials").mkdir(parents=True, exist_ok=True)
    stopped = None
    for i, r in enumerate(trials, 1):
        if r.trial_id in done or r.trial_id in skipped:
            continue
        plan = None
        if r.replay_from is not None:
            plan = first_plan(out, r.replay_from)
            if plan is None:  # not this condition's failure: kept out of its denominator, logged
                _append(out / "skipped.jsonl", {**schedule_entry(r, a),
                                                "reason": f"{r.replay_from.trial_id} produced no first plan"})
                print(f"==> [{i}/{len(trials)}] {r.trial_id} skipped: no plan to replay", flush=True)
                continue
        if not a.no_health_check and not wait_for_servers(a):
            stopped = "a server is down"
            break
        if r.trial_id not in scheduled:
            _append(out / "schedule.jsonl", schedule_entry(r, a))
        print(f"==> [{i}/{len(trials)}] {r.trial_id}", flush=True)
        t0 = time.monotonic()
        (out / "trials" / r.trial_id).mkdir(parents=True, exist_ok=True)
        code = run_process(command(r, a, out, sim, tiptop, plan), sim,
                           out / "trials" / r.trial_id / "console.log", a.run_timeout_s)
        with open(exit_file, "a") as f:
            f.write(f"{r.trial_id}\t{code}\n")
        print(f"    exit code {code} ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    from grasp_eval.recovery_summary import main as summarize

    rc = summarize([str(out)]) if (out / TRIALS_FILE).exists() else 1
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
