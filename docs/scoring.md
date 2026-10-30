# Grasp success definition and run accounting

## Observation-only rule

The evaluator (`grasp_eval/observed_eval.py`) executes the upstream plan exactly as `tiptop_eval.py` does
and only reads state. It never changes actions, pauses, retries, or feeds simulator state to perception or planning.

After each control step it logs the target object's world position, the gripper reference point (midpoint of
the two inner finger links), the commanded gripper value, the measured gripper and arm joints, and the time.

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

| parameter | pilot value | status |
|---|---|---|
| `min_rise_m` | 0.05 m | to validate |
| `min_hold_s` | 0.5 s | to validate |
| `max_obj_to_gripper_m` | 0.10 m | to validate; depends on object size and finger geometry |

**Before comparing anything**, validate these on pilot traces. Check that the native motion lifts the object
more than 5 cm and keeps it there more than 0.5 s before release, and look at the real object-to-finger-midpoint
distance during a good hold. Then set the final values here, mark them frozen with the date, and pass the same
values to every later run. Do not add a dwell or stop to the baseline to make scoring easier.

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
s1v2t0_20261101-141500,,perception_grounding,,Gemini grounded the bowl as the target
s1v0t1_20261101-142210,,,true,cube ended inside bowl
```

Raw records are never edited. `summarize.py` applies these labels on top and reports how many runs were overridden.
A `stable_lift=true` override is ignored unless the run had a valid executed plan.

## Denominators

`python -m grasp_eval.summarize results/<pilot>` reports:

- **Overall stable-lift rate** = stable lifts / scheduled trials. Every scheduled trial counts, including API
  failures, no-plan outcomes, crashes, and runs that never wrote a record (counted as infrastructure).
- **Conditional stable-lift rate** = stable lifts / valid executed plans (server returned a plan, at least one
  step executed, no harness exception).
- Plan-returned rate, per-variant counts, failure-category counts, planning-time range, and final task success
  over manually labeled runs only.

`server success` and `plan_done` never count as grasp success.
