# Week two: grasp-checkpoint recovery (oracle-state baseline)

Status (November 4, 2026): implemented and unit-tested on CPU with stand-ins. **Nothing here has run on a GPU or
against the real simulator/TiPToP server yet.** Expect small fixes on first contact, as with week one.

## What it is

A bounded loop around TiPToP: execute the target's approach, close and post-close retraction; hold; decide whether
the target is held *now*; after a confirmed empty-hand miss, return to an observation pose and ask TiPToP for a
fresh plan from the changed scene. The trial stops once the target is confirmed held after the lift. Nothing is
placed.

This is a **grasp-checkpoint** variant, not the paper's proposed placement-level checkpoint. The first detector is
**`oracle-state`**: it reads simulator ground truth to decide whether to retry, so these are oracle-checkpoint
results, never observation-only or image-based recovery. Every trial and attempt record says so
(`detector`, `detector_privileged`, `outcome_source`, `outcome_privileged`).

| condition | how to run | what it is |
|---|---|---|
| Original | week-one `observed_eval.py` / `run_suite`, unchanged | upstream TiPToP + Gemini ER 2, full plan |
| Checkpoint, one attempt | `recovery_eval.py --max-attempts 1` | stops at the lift checkpoint and holds; never retries |
| Checkpoint recovery | `recovery_eval.py --max-attempts 2` | same checkpoint, one retry with a fresh plan |

The one-attempt mode is the matched control for recovery. Do not relabel it as unchanged upstream TiPToP: it
stops early and adds a hold.

## Week one is untouched

Week two only adds files. No existing module, config, script, pin or patch changed, and there are no new
dependencies (the new pure modules need only numpy; `recovery_eval.py` runs in the same droid-sim-evals uv env).
New code imports week-one helpers read-only: `LiftCriteria`/`score_first_grasp`/`count_close_commands`,
`append_jsonl`/`read_jsonl`, `displaced_pose`, `make_observed_client_class` and the provenance helpers from
`observed_eval.py`, `run_process`/`wait_for_servers` from `run_suite.py`, and `_frac`/`_frac_ci` from
`summarize.py`. Results go to `recovery_trials.jsonl` / `attempts.jsonl`, never `runs.jsonl`.

`observed_eval.main()` is one long function, so `recovery_eval.py` has its own copy of the environment setup
(~60 lines) rather than refactoring the week-one observer.

## Files

| file | role |
|---|---|
| `grasp_eval/checkpoint.py` | state machine (`RecoverySupervisor`), budgets, stop reasons, `OracleStateDetector`, `SensorDetector` interface (not implemented) |
| `grasp_eval/plan_segments.py` | finds the target's close and its retraction; executes through the final retraction waypoint |
| `grasp_eval/recovery_eval.py` | one trial in the simulator (`SimRecoveryBackend` + `IsaacWorld` adapter) |
| `grasp_eval/recovery_records.py` | `RecoveryTrialRecord`, `AttemptRecord` |
| `grasp_eval/recovery_summary.py` | attempt-aware `summary_checkpoint.md` |
| `grasp_eval/recovery_suite.py` | paired scheduler (one-attempt control + recovery per variant/trial/condition) |
| `tests/test_checkpoint.py`, `tests/test_plan_segments.py`, `tests/test_recovery_backend.py`, `tests/recovery_fakes.py` | CPU tests |

## Decisions grounded in the pinned upstream code

- **The retraction is the first `Place` trajectory.** cuTAMP 0.0.6 (`motion_solver.py`) emits, for `Place`, a
  retraction to `world_from_ee @ approach_offset` (−5 cm along the end-effector z axis), then transfer-to-approach,
  then the final approach. The scene-1 plan is `Pick, Pick, close, Place, Place, Place, open, GoToInitial,
  GoToInitial`. A pick-only plan ends `close → GoToInitial`, whose first trajectory is the same retraction; it is
  also accepted. Plans with more than one close, more than one picked object, a non-Pick step before the close, or
  anything else after the close are refused (`unsupported_plan`) before any motion.
- **The final retraction waypoint is applied.** The upstream client advances `_current_plan_step` when a trajectory
  starts and `_current_waypoint_idx` before the step is applied, so `plan_done` turns true while the last
  waypoint is still unapplied. Upstream `tiptop_eval.py` (and the week-one observer) check `plan_done` before
  `env.step`, so they drop that waypoint. `run_segment` tags every returned action with its plan step and stops only
  after `env.step` has applied the last retraction waypoint. It never consults `plan_done` and refuses to apply
  anything past the retraction. The full plan stays loaded in the client (not truncated).
- **Hold.** After the retraction it steps physics for `--confirm-hold-s` (default 1.0 s = 15 control steps) with the
  final commanded joint target and the gripper closed. Identical in both modes.
- **Fresh plan on retry.** The client's remaining plan and counters are cleared, any replay is switched off, and the
  client reconnects (`reset()`: fresh server handler). The request is built from the latest observation, so
  `q_init` is the current measured joint state and the RGB-D is new; each attempt records a fingerprint
  (`request.q_init`, `rgb_sha256_16`, `depth_sha256_16`, `world_from_cam`). A retry whose server output directory
  equals an earlier one stops as infrastructure. A new connection is not proof that every server cache was cleared:
  **CHECK** on the first GPU retry that the second `tiptop_run.log` shows a fresh perception/cuTAMP world.
- **No resets.** Objects, robot and the simulation clock carry over. The env episode length is set to
  `max_sim_s + 10 s` so the trial's own limit fires first; the episode buffer is never cleared between attempts.
- **Displacement probe** is applied once, after attempt 1's plan exists and before its first motion step. The retry
  plans for the displaced object; it is not shifted again.

## Outcome detection (`oracle-state`)

Evaluated on the window from the start of the close to the end of the hold:

- `holding_target`: the target rose with the hand (≥ 0.6 × the hand's rise), stayed within 20 cm of the finger
  midpoint for the whole hold (the value frozen for the baseline on 2026-11-05; see below), and its offset in the hand drifted ≤ 1 cm during the hold. Another object
  following the hand at the same time makes it `uncertain`.
- `failed_empty`: the hand rose ≥ 3 cm, the target's height changed ≤ 1 cm and it moved ≤ 5 mm during the hold
  (settled), no other object followed the hand, and no disturbed object is within 8 cm of the hand.
- otherwise `uncertain` with a reason (`wrong_object_held` stops as its own reason). Uncertain never opens the
  gripper and never retries.

`score_first_grasp(...).lifted` (week one) is still computed per attempt as `historical_lift`, and kept separate:
it means "held long enough at some earlier point", not "held now".

**Known threshold issue:** the cuTAMP retraction lifts only ~5 cm, the same value as week one's `min_rise_m`.
At this checkpoint the week-one stable-lift criterion is therefore borderline even for a good grasp, which is why
the checkpoint decision uses "followed the hand" instead. Expect `first-attempt stable lift` and
`first-attempt confirmed held` to disagree until both are checked against video. Do not lower the week-one
threshold for week-one results.

All detector thresholds are first guesses (`OracleCheckCriteria`, CLI flags of the same names). Validate on the
GPU smoke traces, then freeze them before any comparison and record the frozen values.

### Changes after the November 5 baseline run

The A6000 baseline run ([baseline_report.md](baseline_report.md), [scoring.md](scoring.md)) found measurement
problems that also applied to this code. They are fixed here the same way:

- **Proximity 10 → 20 cm.** The finger "midpoint" is the inner-finger *link-origin* midpoint, about 15.5 cm from
  a held cube. The old 10 cm limit rejected every real grasp. `OracleCheckCriteria.max_obj_to_gripper_m` and the
  `--max-obj-to-gripper-m` default (shared with `historical_lift`) are now the frozen 0.20 m.
- **Geometry-box centers.** Some scene-4/5 meshes sit far from their rigid-body origin (one cube by ~20 cm).
  The trace keeps raw `objects_pos`/`objects_quat` and adds `objects_center`. The checkpoint, `historical_lift`
  and the return-motion disturbance check use the centers. Records add `object_centers`,
  `initial/final_object_centers` and the USD boxes. `historical_lift.position_reference` says which was used.
- **API quota is infrastructure.** A Gemini `429 RESOURCE_EXHAUSTED` used to stop a trial as
  `planning_failure`. It now stops as `infrastructure`, and `recovery_suite` stops before starting another
  simulator (exit 3; fix the quota, then `--resume`).

Still to check on the GPU: `other_near_m` (8 cm) is measured from the same finger-origin reference, which sits
well behind the contact region, so the "disturbed neighbour" check may never fire. Validate it against video
before trusting `failed_empty` in cluttered scenes.

## Empty-hand return (first version)

Open in place (20 control steps, like the client's gripper dwell), cosine joint-space interpolation to the
observation pose (default: the trial's settled start pose, which is also TiPToP's `q_home`; peak 0.5 rad/s), settle
0.5 s. It is accepted only if no object moves more than 1 cm during the whole sequence (a privileged monitor) and
the arm ends within 0.05 rad of the target; otherwise `return_failed`. This is **not a collision-free plan**:
it relies on the hand being empty and above the table after a 5 cm retraction. Check on video that the path is
clear in each scene before trusting it; a cuRobo-planned return is the obvious upgrade.

## Running on the GPU machine

Same environment and servers as week one (docs/setup.md). Use the target name from
`observed_eval.py --inspect-only`.

1. Smoke test, one trial per mode. Watch the video and check the trace:

   ```bash
   cd $TIPTOP_DIR/droid-sim-evals
   uv run python $PROJECT/grasp_eval/recovery_eval.py --tiptop-dir $TIPTOP_DIR/tiptop \
     --scene 1 --variant 0 --trial 0 --target-object <SIM_NAME> --target-label rubiks_cube \
     --instruction "Put the Rubik's cube in the bowl." --detector oracle-state --max-attempts 1 \
     --out-dir $PROJECT/results/week2_smoke
   ```

   Check: the `retract` phase is an upward withdrawal for this grasp; the robot stops before transport; the hold
   is still; `attempts.json` `checkpoint.evidence` (hand rise, target rise, drift) matches what the video shows;
   `final_joint_error_max_rad` is small.
2. Recovery demo. Natural misses may be rare. Use the displacement fixture (clearly labelled) to force one:

   ```bash
   uv run python $PROJECT/grasp_eval/recovery_eval.py ... --max-attempts 2 \
     --displace-dx-m 0.02 --probe-label disp_fixture --out-dir $PROJECT/results/week2_smoke
   ```

   Check: attempt 1 `failed_empty`, a clean return in the video, attempt 2's `request.q_init` equals its
   `start_joint_pos`, a different `server_save_dir`, the displaced pose in attempt 2's `object_positions`. Also
   run a two-failure case (e.g. a larger displacement, or a scene where misses are common) to see
   `attempts_exhausted`.
3. Small paired comparison, after freezing thresholds:

   ```bash
   python3 -m grasp_eval.recovery_suite --tiptop-dir $TIPTOP_DIR --target-object <SIM_NAME> \
     --target-label rubiks_cube --variants 0 1 2 3 4 --trials 2 --conditions baseline disp:0.02,0,0 \
     --paired --out-name week2_oracle_scene1
   python3 -m grasp_eval.recovery_suite ... --resume   # after an interruption
   ```

   With `--paired`, the recovery trial replays the control's first plan for attempt 1 (same scene and
   disturbance), so only the retry gets a fresh plan. Matching seeds alone do not make first plans identical,
   because Gemini, M2T2 and cuTAMP run in other processes.

`python3 -m grasp_eval.recovery_summary results/<dir>` regenerates `summary_checkpoint.md`.

## What the summary reports

Per condition (mode × detector × probe × instruction), each with its denominator:

- first-attempt stable lift (week-one criterion) per scheduled trial;
- first-attempt confirmed held at the checkpoint;
- final confirmed success per scheduled trial (planning, infrastructure, uncertain and crashed trials count as
  failures);
- recovery success among first-grasp failures, and among retry-eligible trials (those that got a second attempt);
- attempts, planning calls and closes per trial; time to success (simulated and wall);
- attempts that lifted earlier but were not held at the checkpoint;
- detector disagreement with the oracle, once a sensor detector exists;
- a paired table (one attempt vs recovery for the same scene, variant, trial and probe).

Trials are grasp-only: do not feed them into the paper's whole-task taxonomy.

## Timing semantics

The trial clock starts after settling, just before the first planning request. `sim_duration_s` counts control
steps (1/15 s each) from then on, including hold, return and settle. `wall_duration_s` includes planning;
the simulator does not step during the blocking planning call (logged per attempt as `planning_wall_s`).
`process_startup_s` is process start to trial clock start. Defaults: 2 closes, 2 planning calls, 180 s
simulated, 600 s wall, plus the suite's 1500 s outer process timeout. A no-plan reply is `planning_failure`,
connection/API errors are `infrastructure`; neither gets an extra budget.

## Not done / known limitations

- No GPU validation of anything: frames, target names, the retraction boundary, thresholds, the return motion,
  timing, or the server-cache reset.
- `SensorDetector` is an interface only. Any sensor-based recovery claim needs it implemented and validated
  against the oracle (false-success and false-empty rates).
- Return motion is interpolated and monitored, not planned.
- Single target, single grasp, scene 1-style plans only. Multi-object plans are refused.
- The optional finger-joint evidence (`--empty-min-finger-closure`) is off until the Robotiq joint scaling is
  checked (`finger_joint` ≈ π/4 rad when closed).
- Per-attempt stage attribution (week one's `attribution.py`) is not wired in yet.
- No learning, failed-grasp memory, IIWA work or placement continuation, by design.
- The displacement probe rotates yaw about the rigid-body origin, not the geometry center. That only matters for
  assets whose mesh is offset from the origin (scenes 4-5).
- Scene 4 variants 1-3 initialise the cube far below the table (baseline report); exclude them until repaired.
- TiPToP's default FR3 model allows more joint-6/7 range than the simulated Panda, and a plan saturated a joint
  near release (baseline pilot run 002). New runs plan for `panda_robotiq`
  (`patches/tiptop/0002-panda-robot-model.patch`). Do not compare them with the FR3 baseline as the same condition.
