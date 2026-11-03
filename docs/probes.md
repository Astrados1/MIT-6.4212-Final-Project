# TiPToP failure-mode probes

The pilot (`run_suite pilot`) answers one question: how often does TiPToP get a stable lift in clean scenes?
On its own it says little about *why* runs fail or how close successes were to failing. The probes below
answer those questions. Each one is either a measurement (stage attribution) or a controlled experimental
condition with its own results directory, so baseline numbers stay uncontaminated.

None of this has run on a GPU yet. The pure-Python parts (scoring, attribution math, perturbation math, suite
scheduling, replay chaining, summaries) are unit-tested, including an end-to-end suite run with stand-ins for
the simulator and TiPToP (`tests/test_probes.py`). The Isaac Lab calls (`write_root_pose_to_sim`, rigid-object
tracing) and the cuTAMP unpickling are untested until the first real run.

| probe | question | changes the experiment? | suite | runs (defaults) |
|---|---|---|---|---|
| Stage attribution | Which stage caused each failure? | No, measurement only | every suite | 0 extra |
| Object displacement | How much object-pose error does the open-loop plan tolerate? | Yes: moves the object after planning | `displacement` | 49 per variant |
| Sensor corruption | How sensitive are perception and grasping to depth noise and calibration error? | Yes: corrupts the observation sent to TiPToP | `sensor` | 27 |
| Scene survey | Does performance hold across other objects and clutter? | No (different scenes) | `scenes` | 24 |

Run time per episode has not been measured yet. Each run starts its own Isaac Sim process; replay runs skip
perception and planning. Time the pilot first and scale the other suites from that.

## 1. Stage attribution (every run)

After each run, `run_suite` does two things:
1. It extracts the scene TiPToP perceived from `tiptop_server_outputs/<ts>/perception/cutamp_env.pkl` (via
   `grasp_eval/extract_perceived_scene.py` in the tiptop pixi environment).
2. It compares that scene and the plan with simulator ground truth (`grasp_eval/attribute_run.py`).

Results go to `runs/<id>/attribution.json` and `attribution.jsonl`, and `summary.md` gets a stage-attribution section.

| measurement | meaning | source |
|---|---|---|
| `first_pick_maps_to_sim_object`, `first_pick_matches_target` | Grounding: is the object TiPToP chose to pick (first `Pick(...)` in the plan) actually the target? Its perceived position is matched to the nearest true object, so no name mapping is needed. | plan labels, perceived scene, sim poses |
| `perception_xy_error_m`, `perception_z_error_m` | How far the perceived object (center of its bounding box) is from the true object origin. xy is the meaningful one; z depends on where each asset's origin sits. The client's 1.5 cm camera z-offset is corrected. | perceived scene, sim poses |
| `sim_objects_without_nearby_detection` | True objects with no perceived object within 5 cm (missed detections). | same |
| `gripper_xy_offset_at_close_m` | Where the fingers actually were relative to the object when the close command started. This combines perception, grasp choice, and any planner/simulator robot-model mismatch. | trace |
| `target_pushed_before_close_m` | Did the approach knock the object before closing? | trace |
| `joint_tracking_error_at_close_rad`, `max_joint_tracking_error_rad` | Did the arm actually reach the planned joints? | trace (`joint_cmd` vs `joint_pos`) |
| `plan_timing.execution_time_scale` | Executed vs planned duration given the client's 50 Hz waypoint assumption (1.0 = assumption holds). Resolves a CHECK item in `docs/upstream_notes.md`. | plan `dt` |
| `stage_hint` | Earliest stage whose measurement crosses a fixed threshold (`ATTRIBUTION_THRESHOLDS` in `grasp_eval/attribution.py`). Order: grounding → perception_pose → execution_tracking → approach_contact → grasp_pose_or_frame → closure_or_lift_dynamics. A guide for video review, not a label. | the above |

Read it like this: if `gripper_xy_offset_at_close_m` is large while `perception_xy_error_m` is small and tracking is fine,
the error comes after perception. Either grasp choice is the cause, or the planner's robot model (`fr3_robotiq` by default)
doesn't match the simulated Panda. Check the latter first (see `docs/upstream_notes.md`).

## 2. Object displacement sweep

```bash
python3 -m grasp_eval.run_suite displacement --tiptop-dir "$TIPTOP_DIR" --target-object <name> \
    --provenance-file provenance/provenance_<ts>.txt [--variants 0 1] [--distances-cm 1 2 3 4 6] \
    [--directions x+ x- y+ y-] [--yaws-deg 15 30 45] [--repeats 2]
```

For each variant:
1. **`fresh_nominal`**: one normal run with a fresh plan. Its saved `tiptop_plan.json` becomes the plan for every other condition of that variant.
2. **`replay_0cm`**: replays that plan with no displacement. This control shows that replay reproduces the fresh run.
   If it doesn't, the rest of that variant can't be interpreted.
3. **`disp_<d>cm_<dir>`**: replay, but after the plan is loaded and before the first motion step, the target is
   moved `d` cm along world ±x / ±y (robot base frame; the robot sits at the origin). Height and orientation stay the same.
4. **`yaw_<a>deg`**: same, but rotated about world z through the object origin.

`summary.md` has a **By group** table that pools the directions at each distance. That table is the success-versus-error curve.
The distance where success collapses is the open-loop tolerance, which is the error range the learned closed-loop policy
must handle. It also sets the perturbation range for collecting training corrections (plan, learning-procedure step 3).

Caveats:
- **Repeats are near-duplicates.** Replays of the same plan are close to deterministic, so repeats mostly measure physics
  noise. Plan-to-plan variation is the pilot's job. To get it here, add more variants (`--variants 0 1 2`); each adds one fresh plan.
- **Moves can be blocked.** A move can push the object into or next to another object. Each record stores the nearest other
  object and its xy distance after the move (`probe.displacement.nearest_other_xy_m`). Treat runs where that is under about 6 cm with care.
- **Scope.** This tests open-loop execution against a stale world. It does not test whether TiPToP would re-perceive
  correctly; it never gets the chance.
- **Skipped replays.** If a variant's fresh plan fails, its replays aren't scheduled. They are listed in `skipped.jsonl`
  and left out of every denominator.

## 3. Sensor corruption

```bash
python3 -m grasp_eval.run_suite sensor --tiptop-dir "$TIPTOP_DIR" --target-object <name> \
    [--variants 0] [--trials 3] [--conditions nominal depth_noise_5mm cam_rot_2deg ...]
```

The simulator renders perfect depth and knows the exact camera pose. A real IIWA setup will have neither. Each condition
corrupts only the observation sent to TiPToP; the simulated world and robot are untouched. Every run gets a fresh plan.

| condition | corruption |
|---|---|
| `nominal` | none (in-suite control) |
| `depth_noise_2mm`, `depth_noise_5mm` | additive Gaussian noise on valid depth pixels |
| `depth_dropout_10pct`, `depth_dropout_30pct` | random pixels set to 0 (TiPToP treats 0 as missing) |
| `cam_trans_5mm`, `cam_trans_10mm` | camera pose translated along a random camera-frame direction |
| `cam_rot_1deg`, `cam_rot_2deg` | camera pose rotated about a random camera-frame axis |

Directions and noise come from `--probe-seed` (= the run seed), and the exact values are recorded in `record.probe.sensor_applied`.
Calibration errors shift the whole perceived scene, so expect `perception_xy_error_m` and `gripper_xy_offset_at_close_m` to grow together.
Independent pixel noise mainly tests segmentation and grasp prediction. Magnitudes are first guesses (`SENSOR_CONDITIONS` in `grasp_eval/run_suite.py`).
Adjust them once the first sweep shows where failures start.

## 4. Scene survey

```bash
python3 -m grasp_eval.run_suite scenes --tiptop-dir "$TIPTOP_DIR" [--scenes 2 3 4 5] [--variants 0 1 2] [--trials 2]
```

This runs the README example instruction for each scene (`configs/scenes.json`). With `target_object: null`, the scored object
is the plan's first `Pick(...)`:
- It is matched to a simulator rigid body by name, or through `configs/object_aliases.json` (`{"perceived label": "sim name"}`).
- If neither works, the fallback is the object nearest the gripper at the first close. These runs have
  `target_source = nearest_at_close`, and the summary lists them for review.

Before trusting results, run `observed_eval.py --inspect-only` for each scene, then fill in `target_object` or the aliases.
Multi-object instructions (scenes 4 and 5) are scored on their first grasp only.

## Outputs

Each suite writes `results/<suite>_scene<N>/` (or `results/scenes/`) containing:
- `schedule.jsonl`, `runs.jsonl`, `attribution.jsonl`, `skipped.jsonl`, `exit_codes.tsv`, `suite.json` (the exact arguments);
- `perceived/` (extracted perceived scenes);
- `runs/<id>/` (record, attribution, trace, video, logs);
- `summary.md`.

Use `manual_labels.csv` exactly as for the pilot (`docs/scoring.md`). Re-run `python3 -m grasp_eval.summarize results/<dir>` after labeling.

## Not covered yet

- **Mid-approach displacement.** The current probe moves the object before any motion. Moving it during the approach is closer to a live disturbance.
- **Object physical properties.** Mass and friction changes would test slip after closure.
- **Planner robot model.** A `panda_robotiq` vs `fr3_robotiq` config probe. Do this right after the geometry check if the gripper offset at close is systematically non-zero.
