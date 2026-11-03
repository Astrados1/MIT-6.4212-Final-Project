"""Schedule and run a suite of observed episodes, then attribute and summarize them.

Usage (on the GPU machine, with the M2T2 and TiPToP servers running unless noted):

    python3 -m grasp_eval.run_suite pilot        --tiptop-dir $TIPTOP_DIR --target-object <name>
    python3 -m grasp_eval.run_suite displacement --tiptop-dir $TIPTOP_DIR --target-object <name>
    python3 -m grasp_eval.run_suite sensor       --tiptop-dir $TIPTOP_DIR --target-object <name>
    python3 -m grasp_eval.run_suite scenes       --tiptop-dir $TIPTOP_DIR
    ... add --dry-run to print the schedule without running anything.

Every run gets its own process (fresh simulator, fresh server connection). Each run is
written to schedule.jsonl before it starts, so crashes still count in the denominators.
After each run, the perceived scene is extracted in the tiptop pixi environment and
stage attribution is computed in the simulator's uv environment (see docs/probes.md).
This module itself needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PY = PROJECT_ROOT / "grasp_eval" / "observed_eval.py"
EXTRACT_PY = PROJECT_ROOT / "grasp_eval" / "extract_perceived_scene.py"
ATTRIBUTE_PY = PROJECT_ROOT / "grasp_eval" / "attribute_run.py"
DEFAULT_INSTRUCTION = "Put the Rubik's cube in the bowl."

# Sensor-corruption conditions: name -> observed_eval flags. Magnitudes are first guesses
# at realistic RGB-D / hand-eye calibration error; adjust after the first sweep.
SENSOR_CONDITIONS = {
    "nominal": {},
    "depth_noise_2mm": {"depth-noise-std-m": 0.002},
    "depth_noise_5mm": {"depth-noise-std-m": 0.005},
    "depth_dropout_10pct": {"depth-dropout-frac": 0.10},
    "depth_dropout_30pct": {"depth-dropout-frac": 0.30},
    "cam_trans_5mm": {"cam-trans-mm": 5.0},
    "cam_trans_10mm": {"cam-trans-mm": 10.0},
    "cam_rot_1deg": {"cam-rot-deg": 1.0},
    "cam_rot_2deg": {"cam-rot-deg": 2.0},
}

DIRECTIONS = {"x+": (1.0, 0.0), "x-": (-1.0, 0.0), "y+": (0.0, 1.0), "y-": (0.0, -1.0)}


@dataclass
class Run:
    scene: int
    variant: int
    trial: int
    instruction: str
    target_object: str
    condition: str
    group: str
    seed: int
    flags: dict = field(default_factory=dict)  # extra observed_eval flags
    plan_from: "Run | None" = None  # replay the plan produced by this earlier run
    run_id: str = ""


def _flag_args(flags: dict) -> list[str]:
    out = []
    for k, v in flags.items():
        out += [f"--{k}", str(v)]
    return out


def _fmt_cm(d: float) -> str:
    return f"{d:g}cm"


def build_pilot(a) -> list[Run]:
    return [Run(a.scene, v, t, a.instruction, a.target_object, "baseline", "baseline",
                a.seed_base + 100 * v + t)
            for t in range(a.trials) for v in a.variants]


def build_displacement(a) -> list[Run]:
    runs = []
    for v in a.variants:
        fresh = Run(a.scene, v, 0, a.instruction, a.target_object, "fresh_nominal", "fresh",
                    a.seed_base + 100 * v)
        runs.append(fresh)
        for r in range(a.repeats):
            seed = a.seed_base + 100 * v + 10 + r
            runs.append(Run(a.scene, v, r, a.instruction, a.target_object, "replay_0cm", "0cm", seed,
                            plan_from=fresh))
            for d, (dname, (ux, uy)) in itertools.product(a.distances_cm, DIRECTIONS.items()):
                if dname not in a.directions:
                    continue
                runs.append(Run(a.scene, v, r, a.instruction, a.target_object,
                                f"disp_{_fmt_cm(d)}_{dname}", _fmt_cm(d), seed,
                                {"displace-dx-m": round(ux * d / 100, 6), "displace-dy-m": round(uy * d / 100, 6)},
                                plan_from=fresh))
            for y in a.yaws_deg:
                runs.append(Run(a.scene, v, r, a.instruction, a.target_object, f"yaw_{y:g}deg",
                                f"yaw{y:g}", seed, {"displace-yaw-deg": y}, plan_from=fresh))
    return runs


def build_sensor(a) -> list[Run]:
    runs = []
    for t in range(a.trials):
        for v in a.variants:
            for name, flags in SENSOR_CONDITIONS.items():
                if a.conditions and name not in a.conditions:
                    continue
                seed = a.seed_base + 100 * v + t
                f = dict(flags)
                if f:
                    f["probe-seed"] = seed
                runs.append(Run(a.scene, v, t, a.instruction, a.target_object, name, name, seed, f))
    return runs


def build_scenes(a) -> list[Run]:
    cfg = json.loads(Path(a.scenes_config).read_text())
    runs = []
    for t in range(a.trials):
        for s in a.scenes:
            sc = cfg[str(s)]
            flags = {}
            if sc.get("object_aliases"):
                flags["object-aliases"] = str((PROJECT_ROOT / sc["object_aliases"]).resolve())
            for v in a.variants if a.variants else sc["variants"][:a.variants_per_scene]:
                if v not in sc["variants"]:
                    continue
                runs.append(Run(s, v, t, sc["instruction"], sc.get("target_object") or "from-plan",
                                f"scene{s}", f"scene{s}", a.seed_base + 1000 * s + 100 * v + t, flags))
    return runs


def build_paper(a) -> list[Run]:
    """Paper-taxonomy comparison: the scene survey across all five scenes, sized for one or two GPUs.

    Defaults: 4 variants per scene x 2 trials = 40 runs. Trial-major order, so a run cut
    short by --max-hours still covers every scene evenly.
    """
    return build_scenes(a)


BUILDERS = {"pilot": build_pilot, "displacement": build_displacement, "sensor": build_sensor, "scenes": build_scenes,
            "paper": build_paper}


def shard_runs(runs: list[Run], shard: tuple[int, int] | None) -> list[Run]:
    """Runs for shard k of n. Whole (scene, variant) groups stay together, so replays keep their source run."""
    if shard is None:
        return runs
    k, n = shard
    groups = sorted({(r.scene, r.variant) for r in runs})
    mine = {g for i, g in enumerate(groups) if i % n == k}
    return [r for r in runs if (r.scene, r.variant) in mine]


def assign_ids(runs: list[Run]) -> None:
    for i, r in enumerate(runs):
        r.run_id = f"{i:03d}_s{r.scene}v{r.variant}_{r.condition}_t{r.trial}"


def schedule_entry(r: Run, suite: str) -> dict:
    return {
        "run_id": r.run_id, "suite": suite, "scene": r.scene, "variant": r.variant, "trial": r.trial,
        "seed": r.seed, "instruction": r.instruction, "target_object": r.target_object,
        "condition": r.condition, "group": r.group, "flags": r.flags,
        "plan_from_run": r.plan_from.run_id if r.plan_from else None,
        "scheduled_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _append(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _record(out: Path, run_id: str) -> dict | None:
    p = out / "runs" / run_id / "record.json"
    return json.loads(p.read_text()) if p.exists() else None


def replay_source(out: Path, r: Run) -> str | None:
    """Saved plan file of the run this replay depends on, or None if that run produced no plan."""
    src = _record(out, r.plan_from.run_id)
    plan = (src or {}).get("artifacts", {}).get("server_plan")
    if src and src.get("server_success") and plan and Path(plan).exists():
        return plan
    return None


def run_one(r: Run, a, out: Path, sim: Path, tiptop: Path) -> int:
    run_dir = out / "runs" / r.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "python", str(EVAL_PY), "--sim-root", str(sim), "--tiptop-dir", str(tiptop),
           "--scene", str(r.scene), "--variant", str(r.variant), "--trial", str(r.trial),
           "--seed", str(r.seed), "--instruction", r.instruction, "--target-object", r.target_object,
           "--run-id", r.run_id, "--out-dir", str(out), "--ws-host", a.ws_host, "--ws-port", str(a.ws_port),
           "--probe-label", r.condition] + _flag_args(r.flags)
    if a.provenance_file:
        cmd += ["--provenance-file", str(Path(a.provenance_file).resolve())]
    if r.plan_from is not None:
        cmd += ["--plan-json", replay_source(out, r)]
    goal = a.goals.get(str(r.scene))
    if goal:
        cmd += ["--goal-json", json.dumps(goal)]
    cmd += a.eval_args
    code = run_process(cmd, sim, run_dir / "console.log", a.run_timeout_s)
    if not a.skip_attribution:
        attribute(r, out, sim, tiptop)
    return code


def run_process(cmd: list[str], cwd: Path, log_path: Path, timeout_s: float | None) -> int | str:
    """Run cmd with output to log_path. On timeout, kill its whole process group (Isaac Sim spawns children)."""
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=(os.name == "posix"))
        try:
            return p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(p.pid, signal.SIGKILL)
            else:
                p.kill()
            p.wait()
            log.write(f"\n[run_suite] killed after {timeout_s:.0f} s timeout\n")
            return "timeout"


def attribute(r: Run, out: Path, sim: Path, tiptop: Path) -> None:
    run_dir = out / "runs" / r.run_id
    rec = _record(out, r.run_id)
    perceived = None
    if rec and rec.get("server_save_dir"):
        save_dir = Path(rec["server_save_dir"])
        perceived = out / "perceived" / f"{save_dir.name}.json"
        if not perceived.exists():  # replays share their source's perceived scene
            run_process(["pixi", "run", "python", str(EXTRACT_PY), "--save-dir", str(save_dir),
                         "--out", str(perceived)], tiptop, run_dir / "extract.log", 600)
    cmd = ["uv", "run", "python", str(ATTRIBUTE_PY), "--run-dir", str(run_dir),
           "--append-to", str(out / "attribution.jsonl")]
    if perceived is not None and perceived.exists():
        cmd += ["--perceived", str(perceived)]
    run_process(cmd, sim, run_dir / "attribution.log", 600)


def servers_healthy(a) -> tuple[bool, str]:
    from grasp_eval.preflight import http_ok

    for url in (f"http://{a.ws_host}:{a.ws_port}/health", a.m2t2_url.rstrip("/") + "/health"):
        ok, detail = http_ok(url)
        if not ok:
            return False, f"{url}: {detail}"
    return True, "ok"


def wait_for_servers(a) -> bool:
    """True once both servers answer; False if they stay down for --server-wait-s."""
    deadline = time.monotonic() + a.server_wait_s
    while True:
        ok, detail = servers_healthy(a)
        if ok:
            return True
        if time.monotonic() >= deadline:
            print(f"ERROR: server down for {a.server_wait_s:.0f} s ({detail}).", file=sys.stderr)
            return False
        print(f"    waiting for servers ({detail})", flush=True)
        time.sleep(30)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("suite", choices=sorted(BUILDERS))
    p.add_argument("--tiptop-dir", type=Path, help="parent of tiptop/ and droid-sim-evals/ (required unless --dry-run)")
    p.add_argument("--out-name", default=None, help="results/<out-name> (default: <suite>_scene<scene>)")
    p.add_argument("--provenance-file", default=None)
    p.add_argument("--ws-host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-attribution", action="store_true")
    # Running on a small GPU budget (docs/setup.md, section G)
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted suite in its existing results directory (same arguments)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="stop starting new runs after this long; continue later with --resume")
    p.add_argument("--shard", default=None, metavar="K/N",
                   help="run only shard K of N (e.g. 0/2 on GPU 0, 1/2 on GPU 1); results go to <out-name>_shardKofN")
    p.add_argument("--run-timeout-s", type=float, default=1200, help="kill a run that takes longer (default 20 min)")
    p.add_argument("--m2t2-url", default="http://localhost:8123")
    p.add_argument("--no-health-check", action="store_true", help="don't check the servers before each run")
    p.add_argument("--server-wait-s", type=float, default=600,
                   help="if a server is down, wait this long for it before stopping the suite")
    p.add_argument("--require-goals", action=argparse.BooleanOptionalAction, default=None,
                   help="refuse to start if goal names are unset (default: on for the paper suite)")
    p.add_argument("--eval-args", nargs=argparse.REMAINDER, default=[],
                   help="passed through to observed_eval.py, e.g. --eval-args --gripper-bodies a b")

    p.add_argument("--scene", type=int, default=1)
    p.add_argument("--variants", type=int, nargs="+", default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--target-object", default=None)
    # displacement
    p.add_argument("--distances-cm", type=float, nargs="+", default=[1, 2, 4])
    p.add_argument("--directions", nargs="+", default=list(DIRECTIONS), choices=list(DIRECTIONS))
    p.add_argument("--yaws-deg", type=float, nargs="+", default=[30])
    p.add_argument("--repeats", type=int, default=1)
    # sensor
    p.add_argument("--conditions", nargs="+", default=None, choices=list(SENSOR_CONDITIONS))
    # scenes
    p.add_argument("--scenes-config", default=str(PROJECT_ROOT / "configs" / "scenes.json"))
    p.add_argument("--scenes", type=int, nargs="+", default=None, help="default: 2-5 (scenes), 1-5 (paper)")
    p.add_argument("--variants-per-scene", type=int, default=None,
                   help="scenes/paper: first N variants of each scene (default 3 / 4)")

    a = p.parse_args(argv)
    defaults = {"pilot": ([0, 1, 2, 3, 4], 3), "displacement": ([0], None), "sensor": ([0], 2), "scenes": (None, 2),
                "paper": (None, 2)}
    dv, dt = defaults[a.suite]
    a.variants = a.variants if a.variants is not None else dv
    a.trials = a.trials if a.trials is not None else dt
    a.scenes = a.scenes or ([1, 2, 3, 4, 5] if a.suite == "paper" else [2, 3, 4, 5])
    a.variants_per_scene = a.variants_per_scene or (4 if a.suite == "paper" else 3)
    if a.require_goals is None:
        a.require_goals = a.suite == "paper"
    if a.shard:
        try:
            k, n = (int(x) for x in a.shard.split("/"))
            assert 0 <= k < n
        except (ValueError, AssertionError):
            p.error("--shard must look like K/N with 0 <= K < N, e.g. 0/2")
        a.shard = (k, n)
    if a.suite not in ("scenes", "paper") and not a.target_object:
        p.error(f"--target-object is required for the {a.suite} suite (see observed_eval.py --inspect-only)")
    if a.suite == "displacement" and a.target_object == "from-plan":
        p.error("the displacement suite needs an explicit --target-object (the object to move)")
    if not a.dry_run and not a.tiptop_dir:
        p.error("--tiptop-dir is required unless --dry-run")
    a.out_name = a.out_name or (a.suite if a.suite in ("scenes", "paper") else f"{a.suite}_scene{a.scene}")
    if a.shard:
        a.out_name += f"_shard{a.shard[0]}of{a.shard[1]}"
    return a


def load_goals(scenes_config: str) -> dict[str, list]:
    """Per-scene goal specs for the automatic task-success check (configs/scenes.json "goal")."""
    path = Path(scenes_config)
    if not path.exists():
        print(f"WARNING: {path} not found; task success will be unchecked", file=sys.stderr)
        return {}
    cfg = json.loads(path.read_text())
    return {k: v["goal"] for k, v in cfg.items() if isinstance(v, dict) and v.get("goal")}


def main(argv=None) -> int:
    a = parse_args(argv)
    a.goals = load_goals(a.scenes_config)
    runs = BUILDERS[a.suite](a)
    assign_ids(runs)  # before sharding, so a run has the same id whichever shard runs it
    runs = shard_runs(runs, a.shard)
    out = PROJECT_ROOT / "results" / a.out_name
    if a.require_goals:
        from grasp_eval.preflight import goal_problems

        scenes = sorted({r.scene for r in runs})
        problems = goal_problems(json.loads(Path(a.scenes_config).read_text()), scenes)
        if problems:
            print(f"{'WARNING' if a.dry_run else 'ERROR'}: goal names unset, so task success (and the paper "
                  "taxonomy) would be unchecked:\n  " + "\n  ".join(problems)
                  + "\nFill in configs/scenes.json or pass --no-require-goals.", file=sys.stderr)
            if not a.dry_run:
                return 2

    if a.dry_run:
        print(f"{a.suite}: {len(runs)} runs -> {out}")
        groups: dict[str, int] = {}
        for r in runs:
            groups[r.condition] = groups.get(r.condition, 0) + 1
        for cond, n in groups.items():
            print(f"  {cond:28s} x{n}")
        for r in runs[: min(len(runs), 8)]:
            src = f"  replay plan of {r.plan_from.run_id}" if r.plan_from else ""
            print(f"  {r.run_id}  seed={r.seed}  {_flag_args(r.flags)}{src}")
        if len(runs) > 8:
            print(f"  ... {len(runs) - 8} more")
        return 0

    tiptop = (a.tiptop_dir / "tiptop").resolve()
    sim = (a.tiptop_dir / "droid-sim-evals").resolve()
    done, scheduled = resume_state(out)
    if scheduled and not a.resume:
        print(f"ERROR: {out} already has a schedule. Use --resume to continue it, or a new --out-name.",
              file=sys.stderr)
        return 1
    if a.resume and scheduled:
        unknown = scheduled - {r.run_id for r in runs}
        if unknown:
            print(f"ERROR: --resume with different arguments: {len(unknown)} scheduled run(s) are not in this "
                  f"suite (e.g. {sorted(unknown)[:2]}). Use the original arguments.", file=sys.stderr)
            return 1
    (out / "runs").mkdir(parents=True, exist_ok=True)
    if not (out / "suite.json").exists():
        (out / "suite.json").write_text(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                                                    for k, v in vars(a).items()}, indent=2))
    skipped_ids = {s["run_id"] for s in _read_jsonl(out / "skipped.jsonl")}
    start = time.monotonic()
    durations: list[float] = []
    stopped = None
    for i, r in enumerate(runs, 1):
        if r.run_id in done or r.run_id in skipped_ids:
            continue
        if a.max_hours is not None and time.monotonic() - start > a.max_hours * 3600:
            stopped = f"--max-hours {a.max_hours} reached"
            break
        if r.plan_from is not None and replay_source(out, r) is None:
            # Not a failure of this condition: its source run has no plan to replay. Logged
            # separately so it stays out of the condition's denominators.
            _append(out / "skipped.jsonl", {**schedule_entry(r, a.suite),
                                            "reason": f"source run {r.plan_from.run_id} produced no plan"})
            print(f"==> [{i}/{len(runs)}] {r.run_id} skipped: no plan from {r.plan_from.run_id}", flush=True)
            continue
        if r.plan_from is None and not a.no_health_check and not wait_for_servers(a):
            stopped = "a server is down"  # this run was never scheduled, so denominators are unaffected
            break
        if r.run_id not in scheduled:  # an interrupted run is already in the schedule
            _append(out / "schedule.jsonl", schedule_entry(r, a.suite))
        remaining = sum(1 for x in runs[i - 1:] if x.run_id not in done)
        eta = f", ~{remaining * sum(durations) / len(durations) / 3600:.1f} h left" if durations else ""
        print(f"==> [{i}/{len(runs)}] {r.run_id}{eta}", flush=True)
        t0 = time.monotonic()
        code = run_one(r, a, out, sim, tiptop)
        durations.append(time.monotonic() - t0)
        with open(out / "exit_codes.tsv", "a") as f:
            f.write(f"{r.run_id}\t{code}\n")
        print(f"    exit code {code} ({durations[-1] / 60:.1f} min)", flush=True)

    from grasp_eval.summarize import main as summarize
    rc = summarize([str(out)])
    if stopped:
        print(f"\nStopped early: {stopped}. Continue with the same command plus --resume.", file=sys.stderr)
        return 3
    return rc


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def resume_state(out: Path) -> tuple[set[str], set[str]]:
    """(finished run ids, scheduled run ids). A scheduled run with no exit code and no record was
    interrupted (e.g. the machine went away) and is run again on --resume; one that exited but
    wrote no record crashed and stays in the denominators as an infrastructure failure."""
    scheduled = {s["run_id"] for s in _read_jsonl(out / "schedule.jsonl")}
    finished = set()
    if (out / "exit_codes.tsv").exists():
        finished = {line.split("\t")[0] for line in (out / "exit_codes.tsv").read_text().splitlines() if line.strip()}
    recorded = {r["run_id"] for r in _read_jsonl(out / "runs.jsonl")}
    return finished | recorded, scheduled


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
