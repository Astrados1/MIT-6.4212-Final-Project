# Week-one runbook (Linux GPU machine)

For the teammate running the baseline. Every command is Bash on an Ubuntu 22.04+ machine with an NVIDIA GPU.
None of this runs on Windows or a machine without CUDA. Stop and report back at any step that fails. Don't work around it silently.

Throughout:

```bash
export PROJECT=/path/to/MIT-6.4212-Final-Project   # this repo
export TIPTOP_DIR=/path/to/tiptop-project          # parent of the three upstream repos
```

## A. Machine and provenance

```bash
cd "$PROJECT"
bash scripts/collect_env_info.sh          # writes provenance/machine_<host>.txt
```

Commit that file. It records the OS, GPU/VRAM, driver, disk and toolchain. Install `pixi` and `uv` from their official instructions if they are missing.

Gemini key: keep it out of shell history, logs and git. In the terminal that will run the TiPToP server:

```bash
read -rs GOOGLE_API_KEY && export GOOGLE_API_KEY   # paste, press Enter; nothing is echoed
```

Before spending time on installs, confirm the key can use `gemini-robotics-er-1.6-preview`.
If that model is unavailable and a substitute is used, record it in `docs/baseline_report.md`. A substitution means the run is no longer an exact reproduction.

## B. Install

```bash
TIPTOP_DIR="$TIPTOP_DIR" bash "$PROJECT/scripts/setup_upstream.sh"
```

This clones tiptop and droid-sim-evals at the pins in `provenance/pins.env`, plus M2T2 at its current default branch.
It never touches an existing checkout.

```bash
cd "$TIPTOP_DIR/tiptop" && pixi install && pixi run setup-planners
cd "$TIPTOP_DIR/M2T2" && pixi run setup && pixi run download-weights
cd "$TIPTOP_DIR/droid-sim-evals" && uv sync
```

Skip the Franka controller, ZED and FoundationStereo setup, because the simulator supplies depth.

Simulation assets: the two upstream docs give different URLs. Try them in this order and keep the archive:

```bash
cd "$TIPTOP_DIR/droid-sim-evals"
test -e assets && echo "assets/ already exists: do not overwrite, inspect it first"
curl -fO https://tiptop-sim-assets.s3.us-east-1.amazonaws.com/assets.zip \
  || curl -fO https://pi-sim-assets.s3.us-east-1.amazonaws.com/assets.zip
unzip assets.zip && ls assets/*.usd
```

Then record provenance (and again after any reinstall):

```bash
cd "$PROJECT"
TIPTOP_DIR="$TIPTOP_DIR" ASSETS_ZIP="$TIPTOP_DIR/droid-sim-evals/assets.zip" \
  ASSETS_URL=<the URL that worked> bash scripts/record_provenance.sh
```

Commit the generated `provenance/provenance_<timestamp>.txt`. After the first install, copy the M2T2 SHA and
cuRobo SHA from it into `provenance/pins.env` so that later installs are pinned.

## C. Geometry check and smoke tests

1. Work through each **CHECK** item in `docs/upstream_notes.md` (fr3 vs Panda kinematics, TCP frame, body names, cuRobo interpolation rate). Write the findings in `docs/baseline_report.md`.
2. Run the planner and M2T2 smoke tests from the official docs. Check `--help` for flags that disable visualization on a headless host.
3. Replay the fixed plan. This checks the simulator, assets and gripper only, not perception or planning:
   ```bash
   cd "$TIPTOP_DIR/droid-sim-evals"
   uv run replay_json_traj.py --json-path tiptop_assets/tiptop_scene1_plan.json --scene 1 --variant 0
   ```
   Watch the video. Does the arm reach the cube, close and lift it?
4. Inspect scene 1 to get the target object's rigid-body name and the finger body names:
   ```bash
   cd "$TIPTOP_DIR/droid-sim-evals"
   uv run python "$PROJECT/grasp_eval/observed_eval.py" --inspect-only --scene 1 --variant 0 \
       --out-dir "$PROJECT/results/inspect"
   ```
   Open `results/inspect/runs/*/scene_inspection.json`. Find the Rubik's cube entry and check that its position
   matches the video. Confirm `left_inner_finger` and `right_inner_finger` are in `robot_body_names`. If they are not, choose
   the two inner finger/pad links and pass them with `--gripper-bodies`. Record the choice in `docs/baseline_report.md`.

## D. One fresh end-to-end run

Terminal 1:
```bash
cd "$TIPTOP_DIR/M2T2" && pixi run server
```
Terminal 2 (with `GOOGLE_API_KEY` exported as above):
```bash
cd "$TIPTOP_DIR/tiptop" && pixi run tiptop-server --host 127.0.0.1 --port 8765
```
Terminal 3: first the unmodified upstream script, then the observed wrapper on the same scene:
```bash
cd "$TIPTOP_DIR/droid-sim-evals"
uv run tiptop_eval.py --scene 1 --variant 0 --instruction "Put the Rubik's cube in the bowl." \
    --episodes 1 --ws-host localhost --ws-port 8765

uv run python "$PROJECT/grasp_eval/observed_eval.py" --scene 1 --variant 0 --trial 0 \
    --instruction "Put the Rubik's cube in the bowl." --target-object <name from step C4> \
    --out-dir "$PROJECT/results/smoke" --tiptop-dir "$TIPTOP_DIR/tiptop"
```

Check that:
- the server log shows a fresh pipeline run, and `record.json → server_save_dir` points to the new
  `tiptop_server_outputs/<timestamp>/` directory with `tiptop_plan.json` in it;
- `results/smoke/runs/<id>/video.mp4` shows the plan executing;
- `trace.npz` and the record's `score` agree with what the video shows. Use this run to start validating the
  thresholds in `docs/scoring.md`.

If the simulator and servers cannot share the GPU, use the offline route in `docs/project_plan.md` §D. It is
not equivalent to the online route, so keep its results separate.

## E. Pilot: scene 1, variants 0–4, 3 trials each

With both servers running:

```bash
cd "$PROJECT"
TIPTOP_DIR="$TIPTOP_DIR" TARGET_OBJECT=<name> \
  PROVENANCE_FILE=provenance/provenance_<timestamp>.txt \
  bash scripts/run_pilot.sh
```

This schedules 15 trials, runs each in its own process with a fresh plan, and writes
`results/pilot_scene1/{schedule.jsonl,runs.jsonl,exit_codes.tsv,summary.md}` with per-run directories.
Then:

1. Watch every failure video, read the matching server log, and fill in `results/pilot_scene1/manual_labels.csv`.
   Watch a few successes too.
2. Re-run `python3 -m grasp_eval.summarize results/pilot_scene1`.
3. Write up failure modes in `docs/baseline_report.md`.
4. Commit the records, summary, labels and report. Videos and traces are gitignored, so share them separately and note where.

Keep TiPToP settings at their upstream defaults (RecGen off, experimental next-to goals off). Any change that
alters behavior gets its own entry in `docs/baseline_report.md` and its own results directory.
