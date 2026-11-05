# Grasp success definition and run accounting

## Observation-only rule

The evaluator (`grasp_eval/observed_eval.py`) executes the upstream plan exactly as `tiptop_eval.py` does
and only reads state. It never changes actions, pauses, retries, or feeds simulator state to perception or planning.

After each control step it logs every rigid object's world position, the gripper reference point (midpoint of
the two inner finger links), the commanded gripper value, the commanded and measured arm joints, and the time
(`trace.npz`). Probe runs (`docs/probes.md`) change the experiment on purpose and are recorded in `record.probe`.

The scored target is the `--target-object` name. With `--target-object from-plan` (scene survey), it is the plan's
first `Pick(...)` object matched to a simulator name (`record.target_source = plan_label`). If that fails, it is the
object nearest the gripper at the first close (`nearest_at_close`, flagged in the summary for review).

## Stable lift (first grasp only)

1. **Baseline height**: the target's z after `settle_sim`, before the first plan action.
2. **Scored segment**: from the first commanded close until the next commanded open (the plan's intentional
   release), or the end of the episode.
3. A sample **holds** when, within that segment, the target is
   - at least `min_rise_m` above the baseline height, and
   - within `max_obj_to_gripper_m` of the gripper reference point,
   while the gripper is commanded closed.
4. **Stable lift** = holding samples span at least `min_hold_s` without a break.

A pushed or tossed object fails the proximity condition. A lift that happens only after a release and re-close is
not scored as the first grasp.

| parameter | frozen value | status |
|---|---|---|
| `min_rise_m` | 0.05 m | frozen 2026-11-05 |
| `min_hold_s` | 0.5 s | frozen 2026-11-05 |
| `max_obj_to_gripper_m` | 0.20 m | frozen 2026-11-05; inner-finger origin reference |

**Before comparing anything**, validate these on pilot traces. Check that the native motion lifts the object
more than 5 cm and keeps it there more than 0.5 s before release, and look at the real object-to-finger-midpoint
distance during a good hold. Then set the final values here, mark them frozen with the date, and pass the same
values to every later run. Do not add a dwell or stop to the baseline to make scoring easier.

## Task success (whole instruction)

Stable lift is our project's metric. The TiPToP paper counts a trial as a success only when the whole rearrangement
worked, so its failure taxonomy (`docs/probes.md` §5) needs task success too. The attribution step checks it
automatically (`grasp_eval/task_goal.py`) from the final resting poses and each object's true geometry box, which
`observed_eval.py` reads from the USD stage:

- `in`: the object's center is inside the container's xy box, and its bottom is between the container's bottom
  (−1 cm) and top (+2 cm).
- `on`: the object's center is inside the container's xy box, and its bottom is within 2 cm of the container's top.
- The gripper must be open at the end of the episode.

The goal per scene lives in `configs/scenes.json` → `goal`. Until its object/container names are filled in from
`--inspect-only`, task success is "unchecked". A `task_success` column in `manual_labels.csv` overrides the
automatic check.

The task check uses the objects' boxes, so a bowl counts as "in" anywhere inside its rim. Check the first few results
against the video before trusting the numbers.

## Failure labels

`perception_grounding`, `planning`, `missed_grasp`, `drop_after_closure_or_lift`, `execution_collision`,
`timeout`, `infrastructure`, `unknown`.

`records.auto_label` assigns a first guess:

| condition | auto label |
|---|---|
| harness or simulator exception, server unreachable, non-JSON reply | infrastructure |
| server error containing `cuTAMP failed` | planning |
| other server error | unknown (read `tiptop_run.log`, then relabel) |
| plan executed, no close command, episode hit time limit | timeout |
| plan executed, closed, never held near gripper above threshold | missed_grasp |
| held briefly but shorter than `min_hold_s` | drop_after_closure_or_lift |

Collisions and wrong-object grounding cannot be detected from the trace. A reviewer watches each failure video,
reads the server log, and writes corrections to `results/<pilot>/manual_labels.csv`:

```
run_id,stable_lift,failure_category,task_success,note
002_s1v2_baseline_t0,,perception_grounding,,Gemini grounded the bowl as the target
005_s1v0_baseline_t1,,,true,cube ended inside bowl
```

Raw records are never edited. `summarize.py` applies these labels on top and reports how many runs were overridden.
A `stable_lift=true` override is ignored unless the run had a valid executed plan.

After reviewing a confirmed task failure, use paper category
`unattributed_task_failure/cause_not_established` if the responsible module cannot be established.
This keeps the observed failure in the denominator without forcing a causal claim. Use `needs_review`
for trials still awaiting review, not as a way to remove known failures from the reported success rate.
Shape errors and in-hand rotation are diagnostic indicators; crossing their thresholds alone is not
proof that they caused a failed task.

## Denominators

`python -m grasp_eval.summarize results/<pilot>` reports:

- **Overall stable-lift rate** = stable lifts / scheduled trials. Every scheduled trial counts, including API
  failures, no-plan outcomes, crashes, and runs that never wrote a record (counted as infrastructure).
- **Conditional stable-lift rate** = stable lifts / valid executed plans (server returned a plan, at least one
  step executed, no harness exception).
- Plan-returned rate, per-variant counts, failure-category counts, planning-time range, and final task success
  over manually labeled runs only.

`server success` and `plan_done` never count as grasp success.

## November 5 measurement validation

The first visually successful smoke run found that the inner-finger **link origins** are about
15.47 cm from a held cube. The 10 cm pilot proximity limit therefore rejects a real 6.87 s carry.
The completed 15-trial pilot used `--max-obj-to-gripper-m 0.20`, with the other thresholds unchanged.
All 15 outcomes were reviewed. The values were frozen at 2026-11-05 06:04:57 UTC before the paper suite.
Always pass the explicit frozen arguments; the generic scoring utility retains its original defaults.

The 11 positive traces hold for 1.333–11.133 s; the four negatives rise only 3.42–10.68 mm.
One positive has only 5.132 cm maximum rise, so height sensitivity remains a limitation.
Proximity limits 17.5, 20 and 22.5 cm all give 11/15 lifts; 10 cm gives 0/15 and 15 cm gives 1/15.
The greatest raised-and-closed distance in these traces is 18.41 cm.
[Freeze record](../provenance/scoring_frozen_20261105.json) includes the reviewed-label and audit hashes.

Some scene-4/5 assets have geometry offset substantially from the body origin. New runs preserve
`objects_pos` and `objects_quat` as raw poses and add `objects_center` to the trace. Lift/proximity
scoring and perception matching use geometry-box centers, with corresponding initial/final centers
in the record. `score.position_reference` records this choice. Existing raw results are unchanged.
