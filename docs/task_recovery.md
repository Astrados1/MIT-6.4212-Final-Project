# Task-level checkpoint recovery (week two, full version)

Status (November 6, 2026): implemented and CPU-tested (188 tests, 42 new). **Not yet run on a GPU.** It
supersedes the grasp-only skeleton described in [week2_recovery.md](week2_recovery.md), which stays in
the repo unchanged. The design follows the discussion of November 5-6 and the safety principles of
[checkpoint_recovery_implementation_guide.md](checkpoint_recovery_implementation_guide.md).

## What it does

TiPToP plans a whole task from one observation and executes it open-loop. This adds a checkpoint
after **every** grasp of the plan, for single- and multi-object tasks:

```
plan step:  approach -> close -> MONITORED LIFT -> HOLD -> verdict
  held       -> back down onto the plan, continue it unchanged
  empty      -> open, go to the observation pose, re-observe, NEW TiPToP plan (a retry)
  slipped    -> stop, lower until the object rests ~1 cm above its start, open, retry
  uncertain  -> continue the plan as the baseline would (doubt never triggers a retry)
  lost       -> stop the trial (the object left the table)
```

A retry plans the **remaining** task from the current scene with the same instruction (cuTAMP does not
re-pick objects whose goal already holds; verify on GPU). Budget: 2 retries per trial, 1 per object,
3 planning calls, 400 s simulated. `--max-retries 0` is the **checkpoint-only control**: same lift and
check, never a retry. It separates the effect of the check motion from the effect of retrying.

The detector is the **oracle**: it reads simulator truth (object geometry centers). Results are
oracle-checkpoint results, never observation-only.

## Two lift modes (`--lift-mode`)

| | `reverse_approach` | `vertical_ik` |
|---|---|---|
| motion | retrace the plan's own approach, slowly (half speed), back toward its start pose | the plan's 5 cm retraction, then straight up with orientation held |
| new motion code | none: the path the arm just came down | damped least-squares IK on the simulator Jacobian (`base_link`) |
| rejoining the plan | the plan's retraction ends exactly at the pre-grasp (verified: 0.0001 rad), which is on the path; descend to it and resume at the next step | replay the recorded lift commands in reverse to the retraction end; the client resumes there |
| after a miss | continue up the approach (empty) toward the start pose, then the observation pose | open low, back up the lift, interpolate to the observation pose |
| lateral motion while unconfirmed | none for a top-down grasp until the pre-grasp; angled grasps follow their approach axis | none |

Both stop lifting as soon as the verdict is clear and cap the hand's rise at 10 cm, so an unconfirmed
object is never more than ~10 cm up. The decision is the same monitor for both (below).

## The monitor (`grasp_eval/task_checkpoint.py`, `TaskCheckCriteria`)

Per simulation step, from the object's and the hand's rise since the close:

| verdict | rule (first values) |
|---|---|
| empty | hand rose >= 5 cm, object rose <= 1 cm and stayed still (0.3 s, 5 mm) |
| following | object rose >= 2 cm: the in-hand offset is recorded (lag allowed before this; ~2 cm observed) |
| held | following, object >= 4 cm up, then a 1 s hold with the hand still: offset change <= 1 cm, sink <= 3 mm |
| slipped | after following: falls back > 8 mm from its highest point, offset changes > 1 cm, or sinks > 3 mm in the hold |
| uncertain | the lift ends (path or 10 cm cap) without a verdict; another object rises with the hand; the hand itself never rose (blocked) |
| lost | object > 5 cm below its start or > 30 cm away |

## Pairing and scoring

`grasp_eval/task_suite.py` runs one recovery trial per baseline trial (same scene, variant, seed,
instruction, goal). **Attempt 0 replays the baseline trial's saved `tiptop_plan.json`**, so only retries
call TiPToP and Gemini. Task success uses the **same goal check** as the baseline
(`task_goal.check_goal`), on the final state and again on the state at the baseline's 90 s episode end
(no credit for extra time). Baseline trials without a plan count as failures in both arms.
`grasp_eval/task_summary.py` reports rescued (fail -> success) and harmed (success -> fail) pairs, an
exact sign test, verdict counts, and *replay fidelity* (trials without a retry that ended like their
baseline trial; differences come from the check motion or simulator non-determinism).

Carry drops (an object leaving the closed gripper during transport) are **recorded, not acted on**:
that is the paper's placement-level failure, outside grasp recovery.

## Calibration (`grasp_eval/checkpoint_calibration.py`)

Replays the monitor over the near-vertical part of every baseline grasp and compares its verdict with
the truth from the rest of the trial (object rose >= 5 cm near the hand). First result on 36 baseline
grasps: **0 false "held"**, 0 false "empty", but many held grasps judged slipped/uncertain, because the
baseline's motion is a fast retraction into an immediate lateral transfer with no hold. The baseline
gives the *separation by height* (held vs missed object rise at 5 cm hand rise: >= 2.7 cm vs <= 2.4 cm,
barely separated; at 8 cm: >= 4.4 vs <= 1.6 cm), not the final thresholds. **Calibrate the thresholds on
GPU checkpoint-only trials** (real slow lift) before the paired run, then freeze them.

## Running on the GPU (planned order)

1. Replay fidelity: `observed_eval.py --plan-json <baseline plan>` on 2-3 baseline trials. It should
   reproduce the baseline outcome; otherwise pairing is approximate.
2. Each lift mode on one held and one naturally missed baseline trial (scene 1 v0/v2/v6/v9 missed
   tonight): `python3 -m grasp_eval.task_suite --baseline results/baseline_panda_20261106 --runs <ids>
   --lift-mode <mode> --out-name task_smoke_<mode>`. Watch every video.
3. One scene-5 trial (three grasps) and one forced clean miss (`--eval-args --displace-dy-m 0.15
   --probe-label disp_fixture`, a displacement far enough that the fingers close on air).
4. Checkpoint-only calibration set (`--max-retries 0`, ~10-15 trials), tune `TaskCheckCriteria`, freeze.
5. Paired run with the chosen mode over all baseline trials (`--resume` after an interruption).

## Files

| file | role |
|---|---|
| `grasp_eval/plan_segments.py` | `find_grasp_units`, `run_actions`, `reverse_approach_path`, `skip_client_to` (added) |
| `grasp_eval/task_checkpoint.py` | `LiftMonitor`, `TaskCheckCriteria`, `TaskSupervisor`, `TaskLimits`, verdicts and stops |
| `grasp_eval/task_recovery_eval.py` | `TaskSimBackend` (both lift modes), the Isaac glue, one-trial CLI |
| `grasp_eval/task_suite.py`, `task_summary.py` | paired suite and summary |
| `grasp_eval/checkpoint_calibration.py` | monitor replay and height separation on baseline traces |
| `grasp_eval/recovery_records.py` | `TaskTrialRecord` (added) |
| `tests/task_fakes.py`, `tests/test_task_*.py`, `tests/data/plans/` | toy world (lag, miss, slip, carry drop, loss, multi-object), real scene 1/3/5 plans |

## Invariants (tested on CPU)

- No lateral end-effector motion while a grasp is unconfirmed (top-down approach / vertical IK).
- While unconfirmed, the object never rises more than the 10 cm hand cap.
- On a slip the object is lowered to <= 1 cm above its start before the gripper opens.
- `uncertain` never retries; a lost object never retries; the checkpoint-only control never retries.
- Attempt 0 replays the paired baseline plan; retries call the server and plan from the observation pose.
- Week-one baseline code paths are unchanged (only read-only `plan_step` logging was added).

## Known limitations

- Nothing has run on Isaac Sim. The IK uses `base_link` and the PhysX Jacobian index `body - 1`
  (fixed base); verify verticality on video.
- `reverse_approach` with an angled grasp retreats along the approach axis, not vertically.
- Return motions to the observation pose are joint interpolation, not collision-checked (as in the
  skeleton); the privileged disturbance monitor aborts if another object moves > 1 cm.
- Thresholds are first values (see Calibration).
