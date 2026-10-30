# Week-one baseline report

_Template. Fill in as the runbook (`docs/setup.md`) progresses. Leave a section marked "not done" rather than deleting it._

## Environment

- Machine record: `provenance/machine_<host>.txt`
- Provenance record(s): `provenance/provenance_<timestamp>.txt`
- Gemini model actually used: (exact ID; note any substitution)
- Differences from the paper's release: current upstream code at the pins, not the original paper release.

## Geometry check (runbook C1)

- Planner robot model vs simulator asset (fr3_robotiq vs Panda + Robotiq 2F-85): result, how checked
- TCP / grasp frame:
- Camera transform (incl. the 1.5 cm client offset):
- cuRobo interpolation rate vs the client's assumed 50 Hz:
- Gripper reference bodies used by the evaluator:
- Target object mapping (instruction → rigid body name), per scene:

## Smoke tests

- Fixed-plan replay (scene 1 / variant 0): pass/fail, video path, notes
- M2T2 / planner smoke tests:
- Fresh end-to-end run: record path, server output dir, video path

## Frozen scoring thresholds

| parameter | value | evidence |
|---|---|---|
| min_rise_m | | |
| min_hold_s | | |
| max_obj_to_gripper_m | | |

Frozen on: (date)

## Pilot results

Paste `results/pilot_scene1/summary.md` here, or link it, along with where videos and traces are stored.

## Failure modes observed

## Behavior-changing modifications

None so far. List every change with its reason and the results directory it affects.

## What the learned skill should replace

Which part of the executed plan (approach → close → lift) failed, and the plan step indices that bound it.
