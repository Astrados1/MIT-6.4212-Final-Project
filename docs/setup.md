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

Before spending time on installs, confirm the key can use `gemini-robotics-er-2-preview`, for example in Google
AI Studio. The pinned TiPToP code asks for `gemini-robotics-er-1.6-preview`, which Google shut down on Aug 31, 2026.
`setup_upstream.sh` patches TiPToP to use ER 2 instead (`patches/README.md`), so the baseline is "upstream TiPToP
with ER 2", not an exact reproduction. That is already recorded in `docs/baseline_report.md`.

## B. Install

```bash
TIPTOP_DIR="$TIPTOP_DIR" bash "$PROJECT/scripts/setup_upstream.sh"
```

This clones tiptop and droid-sim-evals at the pins in `provenance/pins.env`, plus M2T2 at its current default branch,
then applies our patches in `patches/` (the Gemini ER 2 switch, and planning for the Panda arm the simulator has). It never modifies an existing checkout.
For an existing checkout it reports whether each patch is applied and prints the command to apply it.

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

   The same file lists each object's geometry box (`rigid_object_local_bounds`). Check that the cube is about 6 cm on
   a side and the bowl looks bowl-sized; anything in `rigid_object_bounds_notes` means a box could not be read.
   Then put the bowl's name in `configs/scenes.json` → `"1"` → `goal[0].container`. Without it, task success stays
   "unchecked" and the paper-taxonomy table (`docs/probes.md` §5) can't separate successes from failures.

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
- the server log shows the Gemini call succeeding with `gemini-robotics-er-2-preview`. If it fails because of the
  thinking setting, follow the note in `patches/README.md`;
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

This schedules 15 trials and runs each in its own process with a fresh plan. After each run it extracts TiPToP's
perceived scene and computes stage attribution (`docs/probes.md` §1). It writes
`results/pilot_scene1/{schedule.jsonl,runs.jsonl,attribution.jsonl,exit_codes.tsv,summary.md}` with per-run
directories. `DRY_RUN=1` prints the schedule without running anything. Time the pilot. The probe suites in step F
are sized from it.
Then:

1. Open `results/pilot_scene1/review.html` in a browser. It groups runs by root cause, and shows each run's video
   (captioned with the plan step, gripper command and target height, with an outcome card at the end) next to the
   evidence and key measurements. To view it on a laptop, copy the whole results folder; videos aren't in git.
   Watch every failure video, read the matching server log, and fill in `results/pilot_scene1/manual_labels.csv`.
   Watch a few successes too. Confirm each run's automatic paper category (summary section "Paper failure taxonomy")
   and correct it in the `paper_category` column when it's wrong; resolve every `needs_review` run.
2. Re-run `python3 -m grasp_eval.summarize results/pilot_scene1`.
3. Write up failure modes in `docs/baseline_report.md`.
4. Commit the records, summary, labels and report. Videos and traces are gitignored, so share them separately and note where.

Keep TiPToP settings at their upstream defaults (RecGen off, experimental next-to goals off). Any change that
alters behavior gets its own entry in `docs/baseline_report.md` and its own results directory.

## F. Failure-mode probes

Once the pilot works end to end, run the probe suites in `docs/probes.md`, in this order:

1. **`displacement`**: the most important one for the project. It finds how much object-pose error the open-loop plan tolerates.
2. **`sensor`**.
3. **`scenes`**: fill in `configs/scenes.json` (target, aliases and every `goal` name, including how many cans/blocks
   there really are in scenes 4 and 5) and `configs/object_aliases.json` from `--inspect-only` first. For the
   comparison with the paper, use the `paper` suite in section G rather than this one.

Each suite writes its own results directory and `summary.md`. Check each suite's in-suite control before reading
the rest: `replay_0cm` for displacement, `nominal` for sensor.

## G. Running on the cluster (RTX A6000, optional RTX PRO 6000)

### Which GPU runs what

| component | RTX A6000 (Ampere, sm 8.6, 48 GB) | RTX PRO 6000 Blackwell (sm 12.0, 96 GB) |
|---|---|---|
| Simulator (Isaac Sim 5.0; PyTorch 2.7.0 built for CUDA 12.6) | yes | no: that PyTorch build has no Blackwell kernels |
| M2T2 server (PyTorch 2.4.1, nvcc 12.1) | yes | no: same reason, and nvcc 12.1 can't compile for Blackwell |
| TiPToP server (PyTorch 2.7.1, nvcc 12.9) | yes | expected to work |

Run **everything on the A6000**. If both cards are on the same node, TiPToP can move to the Blackwell card to
free A6000 memory. Running the simulator or M2T2 on Blackwell would mean upgrading their pinned PyTorch/CUDA
builds, a compatibility change that would need its own entry in `docs/baseline_report.md`. H100/A100-class
cards are out entirely: they have no RT cores, so Isaac Sim can't render the cameras. Preflight's `gpu` check
reports which components each visible GPU can run.

### Before you start: the cluster

Check these once, on a GPU node (not the login node):

1. **OS**: Ubuntu 22.04 or later (`cat /etc/os-release`). Isaac Sim doesn't support other distributions such as
   Rocky/RHEL; if that's what the cluster runs, ask whether it offers an Ubuntu container (Apptainer/Singularity).
2. **Internet from the GPU node**: TiPToP calls Gemini on every run, and SAM2 downloads its checkpoint on first
   use. Preflight's `network` checks test this. Many clusters block outbound traffic from compute nodes; ask
   for a proxy or an exception if so.
3. **A long enough job** with the A6000, holding three processes at once: M2T2, TiPToP and the suite runner,
   which starts the simulator for each run. Use `tmux` inside the job, or one job script that starts the
   servers in the background. Something like `srun --gres=gpu:a6000:1 --time=8:00:00 --pty bash`, but the
   GPU name and limits depend on your cluster.
4. **Disk**: Isaac Sim and the pixi/uv environments take tens of GB. Put `$TIPTOP_DIR` on a project or scratch
   filesystem if the home quota is small.
5. **Gemini key**: in the TiPToP server's shell, `read -rs GOOGLE_API_KEY && export GOOGLE_API_KEY`. Export it in
   the preflight shell too, so preflight can ask the API whether the key can see `gemini-robotics-er-2-preview`.
   The key never appears in output or files.

### Stages

GPU time is limited, so the plan is small and staged. Each stage starts only once the previous one is clean.
Nothing has been timed yet. The runner prints minutes per run and an ETA after the first run; use the pilot's
numbers to decide how much of stages 4-5 fits.

| stage | command (from `$PROJECT`) | runs | purpose |
|---|---|---|---|
| 0 | `python3 -m grasp_eval.preflight --tiptop-dir $TIPTOP_DIR` | 0 | GPU roles, OS, network, Gemini key, tools, scene files, pins, Gemini patch, servers, goal names, disk |
| 1 | `... preflight --tiptop-dir $TIPTOP_DIR --sim-check` | 5 inspections | the parts never run on a GPU: object boxes, finger links, goal names exist in the sim |
| 2 | `... preflight --tiptop-dir $TIPTOP_DIR --scenes 1 --smoke-run` | 1 | one full run plus attribution; checks record, trace, task check, paper category |
| 3 | `python3 -m grasp_eval.run_suite pilot --tiptop-dir $TIPTOP_DIR --target-object <name>` | 15 | baseline on scene 1 |
| 4 | `python3 -m grasp_eval.run_suite paper --tiptop-dir $TIPTOP_DIR` | 40 | paper-taxonomy comparison: 5 scenes x 4 variants x 2 trials |
| 5 (optional) | `run_suite displacement ...` / `run_suite sensor ...` | 15 / 18 | probes (docs/probes.md) |

That is about 60 runs for stages 2-4. Treat 40 runs as a check of the pipeline and a rough picture of the
failure mix, not a reproduction of the paper's numbers. The summary prints 95% intervals so the uncertainty is visible.
Watch `nvidia-smi` during stage 2: the simulator, TiPToP and M2T2 sharing 48 GB is an estimate, not a measurement.

### Servers on the A6000

Start the servers as in section D, pinned to the A6000 with `CUDA_VISIBLE_DEVICES`. Use the index `nvidia-smi`
shows for it inside your job; it's often 0 when the job gets one GPU. Start the suite in a third shell with
the same `CUDA_VISIBLE_DEVICES`.

To put TiPToP on the Blackwell card instead (same node), start only that server with the Blackwell card's
index. M2T2 and the suite stay on the A6000. Preflight then reports TiPToP-only for the Blackwell card, which
is expected.

### If you get a second simulator-capable GPU

With two cards that can both run the simulator (two A6000s, for example), split a suite into shards. Each GPU
runs its own simulator and TiPToP server; M2T2 is shared on the first.

```bash
# second TiPToP server, started from its own folder: the server writes tiptop_server_outputs/<time to the
# second>/ into its working directory, so two servers in one folder could overwrite each other's outputs
mkdir -p "$TIPTOP_DIR/server_gpu1" && cd "$TIPTOP_DIR/server_gpu1"
eval "$(pixi shell-hook --manifest-path "$TIPTOP_DIR/tiptop/pixi.toml")"
CUDA_VISIBLE_DEVICES=1 tiptop-server --host 127.0.0.1 --port 8766

# one shard per GPU, each in its own shell
CUDA_VISIBLE_DEVICES=0 python3 -m grasp_eval.run_suite paper --tiptop-dir "$TIPTOP_DIR" --shard 0/2 --ws-port 8765
CUDA_VISIBLE_DEVICES=1 python3 -m grasp_eval.run_suite paper --tiptop-dir "$TIPTOP_DIR" --shard 1/2 --ws-port 8766

# afterwards: one summary (put manual labels in the merged directory)
python3 -m grasp_eval.summarize results/paper_shard0of2 results/paper_shard1of2 --into results/paper
```

Start the second server only after the first has served one request. Both share one SAM2 checkpoint, which
TiPToP downloads straight to its final path on first use, so two simultaneous first downloads could leave a
corrupt file that is never fetched again.

### Unattended runs

- `--max-hours H` stops starting new runs after H hours. Runs go trial by trial, so a cut-short suite still covers every scene.
  Set it a little under the job's time limit.
- If the job ends (time limit, preemption), rerun the same command with `--resume` in a new job. Interrupted runs are
  redone. Runs that exited but wrote no record stay counted as infrastructure failures.
- Each run is killed after `--run-timeout-s` (20 min by default) and counts as an infrastructure failure.
- Before each fresh-plan run, the runner checks both servers' `/health`. If one is down, it waits up to
  `--server-wait-s` (10 min), then stops cleanly so the outage doesn't turn into a string of failed runs. Fix it and `--resume`.
- The `paper` suite refuses to start while any goal name in `configs/scenes.json` is unset (stage 1 tells you which).
