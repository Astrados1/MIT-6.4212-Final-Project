# Baseline failure analysis and method (`baseline_panda_20261106`)

**This is the authoritative analysis of the November 6, 2026 baseline.** It supersedes `first_pass.md` in this
folder and the automatic labels in the baseline's `summary_auto_20261106.md`.

TiPToP (Gemini Robotics-ER 2, SAM 2.1, M2T2, cuTAMP/cuRobo) ran on a simulated Panda with a Robotiq 2F-85
gripper in Isaac Sim 5.0. There were 147 trials: 5 scenes × up to 10 variants × 3 trials (scene 4 variant 9
excluded), with 150 s episodes.

| | |
|---|---|
| Plans returned | 137 / 147 (one Gemini 503 outage excluded) |
| Stable lift (grasp worked) | 117 / 137 executed plans |
| **Task success** | **84 / 137 executed plans (61%)**; 84 / 146 counting no-plan trials as failures (58%) |
| Automatic scoring before corrections | 65 / 137 (47%) |

Applied labels: the baseline's `manual_labels.csv` (75 overrides, each with a note). This folder's
`manual_labels.csv` is an identical copy.

---

## Part A. Method

### A1. What each trial records
`grasp_eval/observed_eval.py` runs a trial and saves three kinds of record. Everything after this works on
these files after the trial, read-only.
- **Simulator ground truth:**
  - every object's true pose before the plan and at the end, and its true size from the USD bounds;
  - at each step: object poses, gripper position and orientation, joint commands and readings, and the
    current plan step.
- **TiPToP's own outputs:** the wrist-camera image it planned from, Gemini's boxes and grounded goal,
  SAM's masks, depth and point cloud, M2T2's grasps, cuTAMP's plan and log, and the camera pose
  (`metadata.json`).
- **A captioned video.**

### A2. Success
- **Task success** (`grasp_eval/task_goal.py`) is a geometric check of the goal in `configs/scenes.json`,
  run on the final state.
  - "In": the object's geometric center lies inside the container's footprint, and its bottom sits
    between the container's floor and just above its rim.
  - "On": the object's center is over the container, and its bottom rests on the container's top.
  - The gripper must be open at the end.
  - Centers are geometry-box centers, not body origins: some asset origins are more than 10 cm off.
  - Scene 5 needs any 3 of its 5 blocks.
  - Scene 3 uses `"frame": "container"`: the test is made in the bin's own frame (see A4).
- **Stable lift** measures the grasp alone. The object's center rose at least 5 cm above its pose at the
  close, stayed within 20 cm of the finger reference (the midpoint of the two inner-finger bodies), and
  held for at least 0.5 s before the gripper reopened. These thresholds were fixed before the runs.

### A3. Category, automatic first pass
`grasp_eval/attribution.py` and `grasp_eval/paper_taxonomy.py` check the pipeline in order and take the
**first** failing stage. The rules are in `docs/probes.md` §5, with the paper's categories:
1. Infrastructure: a server or API error. Excluded from the counts.
2. Gemini: the grounded goal differs from the expected goal, or a goal object has no detection within 5 cm.
3. Scene completion: any perceived object box under 70% of its true size on some axis (or 3D IoU < 0.5).
   The label is partial observability.
4. cuTAMP: no plan.
5. Grasp: no stable lift, or the object dropped while the gripper was still closed. Split into M2T2 vs
   heuristic by whether TiPToP had M2T2 grasps for the picked object (`perception/grasps.pt`).
6. Rigid grasp assumption: lifted and carried, the task failed, and the object moved more than 1.5 cm or
   10° relative to the hand.
7. Otherwise: needs review.

### A4. Second pass: how the automatic result was checked and corrected
It ran in five rounds. Each started from something in the automatic results that didn't add up.

**Round 1: look for contradictions.** All failures went into one table with their labels, grasp offsets,
carry state and goal correctness. Three were impossible as stated:
- Scene 5 trials were scored "lifted and carried" while **no block had moved** between start and end.
- Scene 3 bananas had their center inside the bin, but their computed bottom sat **below the table**.
- The scene 4 tomato soup can was "undetected" in **every** trial.

**Round 2: test each contradiction against ground truth.**
- **Scene 5: scored on a reset scene.** Isaac Lab resets the scene inside the step that ends the episode,
  so the "final" state was the reset scene.
  - All 15 trials had hit the 90 s limit. The trace shows the objects jump back to their starts at
    exactly 90 s.
  - They were re-scored from the last frame before the reset. The 15 plans were then **replayed** at
    150 s: a replay of a saved plan is exact (0.000 m difference in a check run), so it finishes the same
    trial without calling Gemini.
  - Code fix (PR #16): score the state before the reset; episode limit `--episode-length-s`, default 150 s.
- **Scene 3: world-axis boxes.** The check boxed objects along the world axes. A tilted banana inside the
  bin was rejected, and a banana beside a turned bin was accepted.
  - End frames of all 29 trials were laid out as contact sheets and called in or out by eye.
  - A check in the bin's own frame matched every frame that could be judged (7 in, 14 out) and resolved
    the 6 where the wall hid the banana.
  - Code fix (PR #16): `"frame": "container"` for scene 3. Round containers (bowl, mug) keep the
    world-axis test, which end frames confirmed is correct for them.
- **Scene 4: out of view.** TiPToP's saved image, Gemini's boxes and the camera pose were pulled for every
  trial. Each goal object's true position was projected into the image. Every "missing" object was out of
  frame or behind the gripper (§B1).

**Round 3: does a cause separate failures from successes?** Each candidate cause was counted in successful
and failed trials of the same scene.
- Under-sized banana: **100% in both** in scene 3, so it explains nothing on its own there. Under-sized
  shape in scene 1 v9 and scene 2: **only in failures**, a real cause.
- Banana swing in the hand: also present in successes (44–133°), so it can't be the label on its own.
- Grasp approach angle: 7 of 8 steep grasps failed vs 5 of 22 near-vertical ones, a real pattern (§B3).
- For partial observability, SAM's saved masks were checked against the image. The masks covered every
  visible pixel; the missing parts were hidden behind the gripper or out of view, so SAM was not at fault.

**Round 4: human video review**, for the 27 trials the numbers couldn't settle. Each got a specific
question, such as "banana in the bin or behind it?" or "why did a centered grasp miss?". The reviewer
watched the trial in the baseline's `review.html` and described what physically happened.

**Round 5: event + measurement → label.**
- The reviewer's event plus a targeted ground-truth measurement gave the category. Example: "caught the
  rim" plus a banana that had swung 46–180° in the hand before contact is rigid grasp; with only 12–15° of
  swing it is partial observability.
- In-hand swing is the banana's orientation relative to the gripper's, both from the trace, from just
  after the grasp until first contact. Contact is the moment the bin's true position moved more than 1 cm.
- In a chain of events, the **first** failure decides the label. Example: 028 swung, then hit the rim,
  then slipped, so it is rigid grasp.
- Each label went into `manual_labels.csv` with a note giving its basis ("reviewed: …", a measurement, or
  the replay). `grasp_eval/summarize.py` then regenerated `summary.md`, `review.html` and
  `paper_taxonomy.json`. The automatic versions are kept as `*_auto_20261106.*`.

### A5. How grasp failures specifically were classified
1. **Did the grasp work?** From ground truth: stable lift (A2), and dropped while carried (separated from
   the fingers while still closed).
2. **Is the grasp the cause?** Only if no earlier stage explains the failure (A3 order). A cube modeled at
   half its width can *cause* a miss, so that miss is labeled scene completion, not grasp.
3. **Subcategory:** M2T2 vs heuristic, from TiPToP's own grasp file. All of ours were M2T2.

Where every failed grasp landed:

| trials | grasp outcome | final label | basis |
|---|---|---|---|
| 000, 002, 006, 049, 056, 100, 104, 106 (scene 1) | no stable lift | **M2T2 unstable grasp** | automatic; grasp frames spot-checked |
| 005 (scene 1) | dropped while carried | **M2T2 unstable grasp** | automatic; video: slips at ~29 s |
| 025 (scene 3) | lifted, slipped out at height | **M2T2 unstable grasp** | video: no bin contact (the automatic label was partial observability) |
| 009, 058, 107 (scene 1 v9) | no stable lift | partial observability | cube modeled half its width; masks: cube behind the gripper |
| 035, 038, 087, 133, 135, 136 (scene 4) | no stable lift | partial observability | the goal was already wrong: objects out of view |
| 029, 125 (scene 3) | no stable lift | motion-planning collision | video: the arm knocked the bin over the banana first |
| 076 (scene 3) | dropped while carried | rigid grasp assumption | video + swing: hit the rim after swinging |

The approach-angle result in §B3 is an analysis *within* the M2T2 group, not part of the classification.

### A6. Limitations of the method
- One reviewer, with no second rater.
- Thresholds (70% shape, 5 cm lift, 1 cm bin contact, the swing cutoff between 15° and 46°) are judgment
  calls, not tuned. The swing cutoff was set after looking at the data.
- "The object swung" is measured; "the planner assumed it rigid" is the interpretation. Swing size depends
  on the simulated friction and contact model.
- First-cause precedence hides second causes. Example: a scene 4 trial with a wrong goal *and* a missed
  grasp is counted once.
- The episode limit is 150 s, not upstream's 90 s, because 3-block plans need 92–115 s.
- Unexplained: 049 and 104 (near-vertical misses) and 143.

---

## Part B. Results by failure mode

| category (paper taxonomy) | trials | share of failures |
|---|---|---|
| Success | 84 | — |
| Partial observability | **33** | 53% |
| M2T2 unstable grasp | 10 | 16% |
| Rigid grasp assumption | 10 | 16% |
| cuRobo motion-planning collision | 6 | 10% |
| cuTAMP no feasible plan | 2 | 3% |
| Cause not established | 1 | 2% |
| Gemini VLM failure | 0 | — |

### The common root cause: one wrist-camera view
TiPToP plans from **one RGB-D image from the wrist camera** at a fixed observation pose, the same joint
configuration in every trial. In this setup:
1. The robot's own gripper covers about the bottom third of the image.
2. The frame covers only part of the table.

Anything unseen doesn't exist for the planner, and anything partly seen is modeled too small. This explains
all of scene 4 and 4 of the other 6 partial-observability failures.

### B1. Partial observability: whole objects out of view (27, scene 4)
Gemini named only what it saw and wrote a goal for that, usually "spam can in bowl". Each goal object's true
position, projected into TiPToP's saved image:

| goal object | out of frame | behind the gripper | visible and detected |
|---|---|---|---|
| tomato soup can | **27 / 27** | 0 | 0 |
| Rubik's cube | 12 | 6 | 9 |
| mug | 0 | 13 | 14 |
| potted-meat can | 0 | 0 | 27 / 27 |

(Bowl centers sit just past the image edge in 9 trials, but the rim is visible and was detected.)
**Gemini detected every goal object that was visible.** The automatic label was "Gemini object detection";
it was relabeled because the error is in what the camera can see, not in Gemini.

### B2. Partial observability: object shape partly hidden (6)
| runs | what the camera saw | effect |
|---|---|---|
| 009, 058, 107 (scene 1 v9) | cube mostly **behind the gripper**, a thin strip visible | modeled 3.1 cm wide vs 7.2 cm; a finger landed on the cube |
| 109 (scene 2) | can seen straight down: lid only | modeled at 7% of its height; the placement tipped the mug |
| 026, 073 (scene 3) | banana from above, partly behind the gripper (073) | modeled ~40% as thick; the banana rigid in the hand (12–15°) still hit the rim |

### B3. M2T2 unstable grasp (10)
The grasp didn't hold. Why, measured from the gripper's true orientation at the close in scene 1:

| approach axis from vertical | grasps | held | missed or slipped |
|---|---|---|---|
| > 30° (side-angled) | 8 | 1 | **7** (000, 002, 006, 056, 100, 106; 005 slipped) |
| ≤ 30° (near top-down) | 22 | 17 | 5 (3 are v9, §B2; **049, 104 unexplained**) |

M2T2 proposes 100–1600 grasps per trial, and cuTAMP picks one that is reachable and collision-free. That
choice is sometimes a steep side grasp, which the cube slides out of. Also 025: the banana slid out at
height.

### B4. Rigid grasp assumption (10, scene 3)
The grasp held, but the banana **swung 46–180° in the hand** before touching the bin. cuTAMP plans the carry
with the object fixed in the hand, so the path cleared only the unswung banana. The banana **caught the bin
rim**: 028, 069, 070, 076, 077, 118, 120, 121, 122, 126. It swings in successes too; in these trials the
swing put it on the rim. Grasps high on the banana dangle more (076).

### B5. cuRobo motion-planning collision (6)
The **arm or gripper** hit an object it had perceived accurately (within 3% of its true size):
- the bin: 020, 022, 024, 029, 125. In 029 and 125 the bin was knocked over the banana before the grasp.
- the bowl: 144, approaching block 2. The disturbed scene then no longer matched the plan.

Likely why (unconfirmed): tight collision margins for the Robotiq fingers near container walls. Execution
is open-loop, so one touch is never corrected.

### B6. cuTAMP no feasible plan (2, both scene 5 variant 5)
- 044: the 60 s search budget ran out (7 of 10 candidate plans optimized).
- 142: a placement motion failed to plan, and then **TiPToP's server crashed** with `KeyError: 'table'`.
  That is a TiPToP bug.
- The third v5 trial (093) planned and succeeded.

### B7. Not established (1), infrastructure (1, excluded)
- 143: the plan finished with 2 of 3 blocks in the bowl; not reviewed.
- 123: Gemini returned 503.

---

## Part C. Results by scene

### Scene 1: cube → bowl. 18 / 30
| cause | runs |
|---|---|
| steep side grasp missed | 000, 002, 006, 056, 100, 106 |
| side grasp held, then slipped | 005 |
| cube behind the gripper (v9, every trial) | 009, 058, 107 |
| top-down miss, unexplained | 049, 104 |

The **grasp** scene. 9 of 12 failures are grasp failures that a post-lift checkpoint can detect: the main
target for checkpoint recovery. Variant 9 will repeat on retry: the cube is hidden from the same pose.

### Scene 2: can → mug. 29 / 30
One failure (109): the can was seen only from the top and modeled as a disc; the placement tipped the mug.

### Scene 3: banana → bin. 11 / 29 (+1 infrastructure)
| cause | runs |
|---|---|
| swung in the hand, caught the rim | 028, 069, 070, 076, 077, 118, 120, 121, 122, 126 |
| arm or gripper hit the bin | 020, 022, 024, 029, 125 |
| modeled too thin, caught the rim | 026, 073 |
| slipped out after the lift | 025 |

The **carry and place** scene: 26 of 29 lifts succeed, and **17 of 18 failures come after a good grasp**,
where a post-grasp checkpoint sees "held" and can't help.

### Scene 4: cube → mug, cans → bowl. 0 / 20 (+7 no plan)
**Unsolvable from this observation pose.** The tomato soup can is out of frame in 27/27; the cube is out of
frame or behind the gripper in 18/27, and the mug in 13/27. The goal is built only from visible objects. The
7 no-plan trials are the same problem. This scene measures camera coverage, not Gemini.

### Scene 5: 3 blocks → bowl. 26 / 28 (+2 no plan)
With 150 s, plans take 92–115 s and nearly all succeed. Every 90 s "failure" was the time limit (14 of 15
replays succeed). Remaining: 144 (arm hit the bowl), 143 (not established), and the two variant-5 no-plan
trials.

---

## Part D. Implications
1. **Checkpoint recovery can address about 10 failures** (unstable grasps, mostly scene 1). Partial
   observability recurs on retry, because retries return to the same observation pose.
2. **It can't address** scene 4 (never seen), scene 3's 17 carry failures, or collisions. State this up front.
3. **The observation pose matters most here:** a view that keeps the gripper out of the image and covers
   the table would remove scene 4's failures and most partial observability.
4. **Our partial-observability share (53%) far exceeds the paper's (13 of 173 trials)**, largely from scene
   4's layout and the gripper in the wrist view. That is a property of this setup.
5. **Report upstream:** the `KeyError: 'table'` crash (142), and that 90 s episodes are too short for
   3-object tasks.

## Files
| file | content |
|---|---|
| `failure_analysis.md` | this document |
| `manual_labels.csv` | the applied labels (copy of the baseline's) |
| `first_pass.md` | the first pass (historical; superseded) |
| baseline (pod and `results/pod_backup/.../baseline_panda_20261106/`) | `summary.md`, `review.html` (videos), `manual_labels.csv`, `*_auto_20261106.*` (automatic originals) |
| `results/baseline_s5_replay150/` (pod) | the 15 scene-5 replays at 150 s |
