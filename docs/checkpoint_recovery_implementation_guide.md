# Checkpoint-based grasp recovery: implementation guide (week 2)

Written November 6, 2026, after the first GPU smoke test of the week-two skeleton. Hand this file to the chat
that implements the full version. It is self-contained: background, what broke, the target design, the
parameters to start from, how to calibrate them, and how to test.

---

## 1. Goal for week 2

Turn the existing **skeleton** (grasp, short lift, check, and on a miss, retry once) into a **complete,
credibly tested** checkpoint-recovery system:

1. **Continue the task after a confirmed grasp.** Today the skeleton stops at the checkpoint and never places
   anything.
2. **Handle multi-object plans.** Today any plan with more than one grasp is refused, so scenes 4 and 5 cannot
   run.
3. **Score whole-task success exactly as the baseline does**, so baseline and recovery compare directly.
4. **Pair every recovery trial with a baseline trial.** Attempt 1 replays the baseline trial's saved plan, so
   only a retry produces a new plan.
5. **Make the check safe and unambiguous** (sections 4 and 5). This is the main redesign.
6. **CPU tests plus a GPU smoke test** before any large run.

---

## 2. Background: how TiPToP works and where the checkpoint goes

TiPToP is a **one-shot, open-loop** planner. It takes **one** observation, plans the **whole** task, and the
simulator client then executes the plan **without looking again**.

Per planning request (timings from the Nov-6 scene 1 trial on an L40S):

| stage | what it does | time |
|---|---|---|
| observation | client sends the wrist-camera RGB + depth, camera pose, current joints (`q_init`), instruction | — |
| Gemini Robotics-ER 2 | **one call**: 2D boxes for objects + instruction translated into goal atoms, e.g. `on(Rubik's_cube, bowl)` | ~5 s |
| SAM2 | box → pixel mask per object | ~12 s (first load) |
| 3D scene | masks + depth → object point clouds; table plane (RANSAC + DBSCAN); approximate object shapes from partial views | ~4 s |
| M2T2 (port 8123) | hundreds of candidate 6-DoF grasps from the point cloud | — |
| cuTAMP | samples 256 particles (grasp, placement, configurations), optimizes them on the GPU, keeps those satisfying all constraints | ~1 s |
| cuRobo | collision-free joint trajectories for each motion (arm model: `panda_robotiq`, patch 0002) | ~6 s |
| reply | plan sent back; everything saved to `tiptop_server_outputs/<timestamp>/` (`tiptop_plan.json`, `tiptop.yml`, perception) | 27 s total |

A single-object plan (scene 1, 9 steps; cuTAMP 0.0.6 structure, see `grasp_eval/plan_segments.py`):

| step | label | motion |
|---|---|---|
| 1–2 | `Pick(X)` trajectory ×2 (sometimes ×3) | home → above X → grasp pose |
| 3 | `Pick(X)` gripper close | fingers close (client holds 20 ticks) |
| 4 | `Place(X)` trajectory | **retraction**: ~5 cm along the end-effector's −z (`approach_offset`), straight up for a top-down grasp |
| 5 | `Place(X)` trajectory | transfer to above the target |
| 6 | `Place(X)` trajectory | final approach |
| 7 | `Place(X)` gripper open | release |
| 8–9 | `GoToInitial` trajectory ×2 | back home |

Execution in droid-sim-evals: waypoints every 0.02 s, simulator at 15 Hz, so every 3rd waypoint (stride 3,
1.111× planned duration). **No feedback at all.** A missed grasp still carries air to the bowl.

**The checkpoint sits between step 4 (retraction) and step 5 (transfer).** For multi-object plans it repeats
after every `Pick` close.

---

## 3. What exists today (skeleton) and what the GPU smoke test showed

### Files

| file | role |
|---|---|
| `grasp_eval/checkpoint.py` | `OracleCheckCriteria`, `OracleStateDetector` (privileged: reads simulator truth), `RecoverySupervisor` state machine, `Stop` reasons, `is_api_quota_error` |
| `grasp_eval/plan_segments.py` | `find_grasp_segment` (approach, close, retraction; **refuses >1 close or >1 picked object**), `run_segment`, `clear_client_plan` |
| `grasp_eval/recovery_eval.py` | `SimRecoveryBackend`, `IsaacWorld`, `joint_interpolation`, `ReturnConfig`, empty-hand return (`prepare_retry`) |
| `grasp_eval/recovery_suite.py` | schedules trials; `--paired` makes recovery attempt 1 replay the one-attempt control's plan; `disp:DX,DY,YAW` conditions; quota stop; `--resume` |
| `grasp_eval/recovery_records.py`, `recovery_summary.py` | records and summary |
| `tests/test_checkpoint.py`, `tests/test_recovery_backend.py`, `tests/recovery_fakes.py` | CPU tests with an upstream-faithful fake client and a toy world |

Current criteria (`OracleCheckCriteria`): `max_obj_to_gripper_m 0.20`, `min_gripper_rise_m 0.03`,
`held_min_follow_frac 0.6`, `held_max_rel_drift_m 0.01`, `empty_max_target_rise_m 0.01`,
`empty_max_target_motion_m 0.005`, `other_near_m 0.08`; hold 15 steps (1 s). Empty-hand return: open in place
20 steps, cosine joint interpolation to the start pose (= TiPToP `q_home`) at ≤ 0.5 rad/s, settle 0.5 s; abort
if any object moves > 1 cm or the final joint error is > 0.05 rad.

Notes:
- The gripper reference point is the **inner-finger link-origin midpoint**, about **15.5 cm behind** a held
  cube's center. That is why distance limits look large.
- Positions use **geometry-box centers**, not body origins (some scene-4/5 meshes are offset).

### Smoke test, scene 1 variant 0 (results in `results/smoke_g5/` on the pod)

| run | intended | what happened | verdict | real cause |
|---|---|---|---|---|
| one attempt | normal grasp | hand rose **3.9 cm**, cube **2.0 cm**, then 0.1 mm drift in a 1 s hold (clearly held) | `uncertain: target_displaced_vertically_but_not_held` | the cube lags while the grip settles; a 5 cm lift is mostly lag, so 2.0/3.9 = 51% < the 60% rule. Same grasp in a full baseline trial lifted 10.4 cm and held 6.2 s |
| forced miss | cube moved +2 cm in x after planning | hand landed **on top of** the cube (stopped 2.6 cm high, tilted it); retraction raised the hand **−0.3 cm** | `uncertain: retraction_did_not_raise_gripper` | the fixture shifted the cube under the fingers, not away from them |

What worked: plan splitting on a real plan (close at step 2, retraction at step 3, 155-waypoint retraction),
segment execution, hold (final joint error 0.004 rad), traces, records, safe stops.

---

## 4. Safety principles (from the design discussion)

A failed or marginal grasp must not drop the object from height or carry it off the table:

1. **Low check height.** Just high enough to make held vs not-held unambiguous, so a drop is a few centimetres
   onto the object's own footprint.
2. **No lateral motion before the grasp is confirmed.** Lift strictly vertically, keep the wrist orientation.
   A "standard" check pose means standard **relative to the object and table**, not a fixed spot in the room.
   (The wrist camera is fixed to the gripper, so it still gets a consistent view of the gripper's contents, which
   a future camera-based detector needs.)
3. **Slow lift.** Fast motion shakes loose marginal grasps.
4. **Watch continuously, not just at the end.** On a slip, stop, lower back down and open: set the object down,
   don't drop it.
5. **A lost object is recorded, not retried.** If it rolls or falls off the table, stop with `object_lost`.

---

## 5. Target design: three-stage checkpoint

```
plan: [approach] → [close] → STAGE 1 slow vertical lift (monitored) → STAGE 2 low check pose (hold + check)
                                   │                                         │
                                   ├─ clean miss  → open in place → RETRY    ├─ held  → STAGE 3 continue plan (monitored carry)
                                   └─ slipping    → lower, open    → RETRY   └─ empty → open in place (≤ ~5 cm) → RETRY
RETRY = return to start pose (q_home) → new observation → NEW TiPToP plan (new Gemini call, new grasps) → attempt n+1
```

### Stage 1: slow, vertical, monitored lift
- Replace the plan's quick retraction with a **slow vertical lift** of the end effector. Options, simplest first:
  - **Time-scale the plan's retraction waypoints** (e.g. 2–3× slower), then continue vertically to the check
    height (below).
  - The sim side has no planner IK. For the extra vertical lift, use **Isaac Lab's `DifferentialIKController`**
    (end-effector z-up, orientation held) or a small Jacobian step loop. Keep the motion strictly vertical.
- **Every step**, compute the object's rise, the hand's rise, and the object's offset relative to the hand
  (geometry centers):
  - **Clean miss (early exit):** hand rise ≥ ~2.5–3 cm, object rise < ~0.5 cm, object stationary → `failed_empty`.
    Open in place, retry.
  - **Lag allowance:** allow the object to trail the hand by up to ~2.5 cm of hand rise before it starts
    following. Observed lag: ~2 cm.
  - **Slip:** after the object started following (object rise ≥ ~1 cm), if it descends > ~5 mm from its maximum,
    or its offset in the hand changes by > ~1 cm → **stop lifting, lower until the object bottom is ~1 cm above
    the table, open**. Record `slipped_during_lift`, retry.

### Stage 2: low check pose
- **Where:** directly above the grasp location (same x/y), same wrist orientation.
- **Height:** lift until the **object's bottom is ~5 cm above its starting bottom** (geometry box). **Cap the
  hand's rise at ~8–10 cm.** If the cap is reached and the object hasn't risen 5 cm, it's not held.
- **Hold ~1 s**, then decide:
  - **held:** object rise ≥ 5 cm, it follows the hand (ratio measured **after** the lag window, or simply
    object-to-hand offset stable during the hold), drift ≤ ~1 cm, no other object following the hand;
  - **empty:** object rise ≤ ~1 cm and object settled;
  - **uncertain:** anything else (keep the conservative "uncertain never retries" rule, but it should now be rare).

### Stage 3: continue the plan, with a monitored carry
- **Getting back onto the plan.** The plan's transfer (step 5) was planned from the retraction end
  configuration, which is directly below the check pose on the same vertical line:
  - **Option A (recommended first):** move **back down vertically** from the check pose to the retraction end
    configuration (retracing the lift, so collision-free), then execute steps 5–9 unchanged. No server changes.
  - **Option B (upgrade):** a cuRobo motion from the check pose to the plan's pre-place pose (needs a planner
    endpoint on the TiPToP server). Smoother, more work.
- **Monitor the carry:** if the object detaches mid-carry, **stop the arm** instead of finishing an empty carry.
  Record `dropped_during_carry`. This is the paper's placement-level failure; measure it, don't recover it yet.

### Multi-object plans (scenes 4 and 5)
- Generalize `find_grasp_segment` to **split a plan into one segment per `Pick`**: [approach_i, close_i,
  retraction_i] ... [place_i] ..., and run stages 1–3 **after every close**.
- On a miss at pick i: return to the start pose and **re-plan the remaining task** with the same instruction.
  Goals already satisfied (objects already placed) should be recognized by cuTAMP. **Verify that on GPU**: a
  replan must not re-pick an already-placed object.
- Budget: **one retry per grasp** (`max_attempts=2` per pick), plus a global planning-call cap and time limits
  scaled per scene. Scene 5's 25-step plan took 6.8 min wall clock in the baseline, so the current
  180 s sim / 600 s wall limits are too small for multi-object trials.

### Retry policy (unchanged in spirit)
- Always a **fresh observation and fresh plan** from the start pose. It usually picks a different grasp and sees
  where the object actually is now. Never replay the failed plan.
- If the empty-hand return disturbs an object > 1 cm, stop with `return_failed` (existing behaviour).

---

## 6. Calibrating the thresholds (do this before the GPU smoke test)

**Data:** the overnight baseline run (150 trials: 5 scenes × 10 variants × 3 trials, TiPToP + Panda planner).
Every baseline trial passes through approach → close → retraction, so its trace contains exactly the window
the checkpoint sees.

- `grasp_eval/observed_eval.py` now records **`plan_step`** per trace sample (the client's
  `_current_plan_step`). This was added in the working tree of branch `week2/stage0-cuda-check`, copied to the
  pod, and **not committed yet**. Commit it with the guide's work.
- **Ground truth:** the week-one `stable_lift` label (whole-trial lift and hold, frozen criteria: rise ≥ 5 cm,
  hold ≥ 0.5 s, distance ≤ 0.20 m), plus task success, plus video for borderline cases.
- **Procedure:** for each trial, cut the close-to-retraction-end window using `plan_step` and `gripper_cmd`.
  Compute candidate features: object/hand rise at each hand-rise level, lag, post-lag follow ratio, in-hand
  drift, object velocity. Choose thresholds that separate held from missed. Report a confusion table:
  false-held is the dangerous error (it carries air); false-empty wastes a retry.
- Raj's pilot found **scene 1 variant 4 misses its grasp 3/3**: a natural source of real misses for both
  calibration and the GPU smoke test.

---

## 7. Fix the forced-miss fixture

The 2 cm shift put the cube **under** the fingers. For a clean miss:
- shift the object **perpendicular to the fingers' closing direction**, using the grasp pose from the plan, by
  more than the object half-width plus finger-pad half-width (start at **6–8 cm**), keeping it on the table;
- or move it well away (e.g. 15 cm) for an unambiguous empty grasp;
- confirm on video that the fingers close on air. Label results `probe: disp_fixture`, never baseline.

---

## 8. Records and metrics to add

Per attempt: `stage1_outcome` (`following` / `clean_miss` / `slipped`), lift profile (hand and object rise
over time, lag), check-pose evidence, `continued_plan` (true/false, option A/B), carry monitoring result.

Per trial:
- **task success** using the same goal checks as the baseline (`grasp_eval/task_goal.py`, `configs/scenes.json`);
- first-attempt confirmed held; recovered after retry; number of retries; `object_lost`;
  `dropped_during_carry`; `slipped_during_lift`;
- the paired baseline trial id and whether attempt 1 replayed its plan.

Stop reasons to add to `Stop`: `slipped_during_lift`, `object_lost`, `dropped_during_carry` (keep existing ones).

---

## 9. Testing plan

### CPU (before touching the GPU)
- Extend `tests/recovery_fakes.py` `ToyWorld` with: **grasp lag** (object starts following after N cm of hand
  rise), **slip** (detaches after following), **push** (hand lands on the object and stalls), **multi-object
  plans** (several Pick/Place blocks), **object falls off table**.
- Test cases:
  - lag then held → confirmed at the check pose → plan continues → task scored;
  - clean miss → early exit → open → return → fresh plan → success on retry;
  - slip mid-lift → stop, lower, open (object never released above the set-down height) → retry;
  - push/stall → `uncertain` (no retry);
  - multi-pick: miss on pick 2 → replan the remaining task only;
  - carry drop → arm stops, `dropped_during_carry`;
  - lateral motion never occurs before confirmation (assert the end-effector x/y is constant through stages 1–2);
  - drop height never exceeds the cap (assert object bottom ≤ start + cap while unconfirmed);
  - attempt 1 replays the paired baseline plan; retries call the server.

### GPU smoke test (L40S pod, after the CPU tests pass)
1. **Replay determinism:** replay a baseline trial's plan with the checkpoint disabled. It should reproduce the
   baseline outcome. If not, pairing must be reported as approximate.
2. **Normal grasp, scenes 1–5 (variant 0):** confirmed held, plan continues, task scored like the baseline.
3. **Natural miss:** scene 1 variant 4: clean-miss or slip path, retry with a new plan.
4. **Forced clean miss** with the fixed fixture: retry path end to end.
5. **Multi-object miss** (scene 4 or 5): replan of the remaining task only.
6. Watch every video; check the evidence in `attempts.json` matches what the video shows.

### Then the paired run
150 recovery trials, one per baseline trial, attempt 1 replaying the baseline's saved plan. Compare task success
with a paired sign test on discordant pairs. With about 13% of pairs flipping, 100–150 pairs give roughly 90%+
power.

---

## 10. Invariants (must always hold)

- No lateral end-effector motion while the grasp is unconfirmed.
- While unconfirmed, the object bottom never exceeds start + ~5 cm (hand rise capped at ~8–10 cm).
- On a slip, the object is set down (lowered, then opened), never released at height.
- `uncertain` never retries; a lost object never retries.
- The oracle detector reads simulator truth: label results "oracle-checkpoint", never observation-only.
- Week-one baseline code paths stay unchanged apart from read-only logging (e.g. `plan_step`).

---

## 11. Open decisions

1. Stage 3 option A (descend back onto the plan) vs option B (cuRobo carry from the check pose). Recommended:
   A first.
2. Exact check height and lift speed. Start at a 5 cm object rise and ~3 cm/s, then calibrate (section 6).
3. Whether to add a camera-based detector this week or keep the oracle and design the check pose for it later.
4. Time limits per scene for multi-object trials.

---

## 12. Environment status (for the implementing chat)

- Runpod Community Cloud **L40S, driver 550.144.03**, pod `jhjif4nvjpexzy`, SSH `root@193.183.22.62 -p 1490`
  with key `~/.ssh/runpod_ed25519`. Everything persistent under `/workspace`; settings in `/workspace/env.sh`.
  Gemini key in `/workspace/.gemini_key` (root-only, never print it).
- Servers: M2T2 on :8123, TiPToP on 127.0.0.1:8765, started by `/workspace/start_servers.sh`; logs in
  `/workspace/logs/`.
- Verified: stages 0–3 pass; scene 4 v1–3 repaired (`scripts/fix_scene_assets.py`); preflight all PASS
  (`planner-robot: panda_robotiq`); full baseline trials work on all 5 scenes (scene 4 can fail at planning, a
  genuine TiPToP outcome); about 3.2 min per single-object trial, 6.8 min for scene 5.
- Open PR: Astrados1/MIT-6.4212-Final-Project#12 (stage-check fixes, scene-4 repair, results). Not merged.
- Runpod balance: about 16 h at $0.81/hr as of Nov 6 evening. A pod without a network volume is **deleted** at
  $0, so top up before running out.
