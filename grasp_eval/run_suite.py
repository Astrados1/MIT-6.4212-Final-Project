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
import subprocess
import sys
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
            for v in a.variants if a.variants else sc["variants"][:3]:
                if v not in sc["variants"]:
                    continue
                runs.append(Run(s, v, t, sc["instruction"], sc.get("target_object") or "from-plan",
                                f"scene{s}", f"scene{s}", a.seed_base + 1000 * s + 100 * v + t, flags))
    return runs


BUILDERS = {"pilot": build_pilot, "displacement": build_displacement, "sensor": build_sensor, "scenes": build_scenes}


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
    cmd += a.eval_args
    with open(run_dir / "console.log", "w") as log:
        code = subprocess.run(cmd, cwd=sim, stdout=log, stderr=subprocess.STDOUT).returncode
    if not a.skip_attribution:
        attribute(r, out, sim, tiptop)
    return code


def attribute(r: Run, out: Path, sim: Path, tiptop: Path) -> None:
    run_dir = out / "runs" / r.run_id
    rec = _record(out, r.run_id)
    perceived = None
    if rec and rec.get("server_save_dir"):
        save_dir = Path(rec["server_save_dir"])
        perceived = out / "perceived" / f"{save_dir.name}.json"
        if not perceived.exists():  # replays share their source's perceived scene
            with open(run_dir / "extract.log", "w") as log:
                subprocess.run(["pixi", "run", "python", str(EXTRACT_PY), "--save-dir", str(save_dir),
                                "--out", str(perceived)], cwd=tiptop, stdout=log, stderr=subprocess.STDOUT)
    cmd = ["uv", "run", "python", str(ATTRIBUTE_PY), "--run-dir", str(run_dir),
           "--append-to", str(out / "attribution.jsonl")]
    if perceived is not None and perceived.exists():
        cmd += ["--perceived", str(perceived)]
    with open(run_dir / "attribution.log", "w") as log:
        subprocess.run(cmd, cwd=sim, stdout=log, stderr=subprocess.STDOUT)


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
    p.add_argument("--eval-args", nargs=argparse.REMAINDER, default=[],
                   help="passed through to observed_eval.py, e.g. --eval-args --gripper-bodies a b")

    p.add_argument("--scene", type=int, default=1)
    p.add_argument("--variants", type=int, nargs="+", default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--target-object", default=None)
    # displacement
    p.add_argument("--distances-cm", type=float, nargs="+", default=[1, 2, 3, 4, 6])
    p.add_argument("--directions", nargs="+", default=list(DIRECTIONS), choices=list(DIRECTIONS))
    p.add_argument("--yaws-deg", type=float, nargs="+", default=[15, 30, 45])
    p.add_argument("--repeats", type=int, default=2)
    # sensor
    p.add_argument("--conditions", nargs="+", default=None, choices=list(SENSOR_CONDITIONS))
    # scenes
    p.add_argument("--scenes-config", default=str(PROJECT_ROOT / "configs" / "scenes.json"))
    p.add_argument("--scenes", type=int, nargs="+", default=[2, 3, 4, 5])

    a = p.parse_args(argv)
    defaults = {"pilot": ([0, 1, 2, 3, 4], 3), "displacement": ([0], None), "sensor": ([0], 3), "scenes": (None, 2)}
    dv, dt = defaults[a.suite]
    a.variants = a.variants if a.variants is not None else dv
    a.trials = a.trials if a.trials is not None else dt
    if a.suite != "scenes" and not a.target_object:
        p.error(f"--target-object is required for the {a.suite} suite (see observed_eval.py --inspect-only)")
    if a.suite == "displacement" and a.target_object == "from-plan":
        p.error("the displacement suite needs an explicit --target-object (the object to move)")
    if not a.dry_run and not a.tiptop_dir:
        p.error("--tiptop-dir is required unless --dry-run")
    a.out_name = a.out_name or (f"{a.suite}" if a.suite == "scenes" else f"{a.suite}_scene{a.scene}")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    runs = BUILDERS[a.suite](a)
    assign_ids(runs)
    out = PROJECT_ROOT / "results" / a.out_name

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
    if (out / "schedule.jsonl").exists():
        print(f"ERROR: {out} already has a schedule. Choose a new --out-name.", file=sys.stderr)
        return 1
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                                                for k, v in vars(a).items()}, indent=2))
    for i, r in enumerate(runs, 1):
        if r.plan_from is not None and replay_source(out, r) is None:
            # Not a failure of this condition: its source run has no plan to replay. Logged
            # separately so it stays out of the condition's denominators.
            _append(out / "skipped.jsonl", {**schedule_entry(r, a.suite),
                                            "reason": f"source run {r.plan_from.run_id} produced no plan"})
            print(f"==> [{i}/{len(runs)}] {r.run_id} skipped: no plan from {r.plan_from.run_id}", flush=True)
            continue
        _append(out / "schedule.jsonl", schedule_entry(r, a.suite))
        print(f"==> [{i}/{len(runs)}] {r.run_id}", flush=True)
        code = run_one(r, a, out, sim, tiptop)
        with open(out / "exit_codes.tsv", "a") as f:
            f.write(f"{r.run_id}\t{code}\n")
        print(f"    exit code {code}", flush=True)

    from grasp_eval.summarize import main as summarize
    return summarize([str(out)])


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
