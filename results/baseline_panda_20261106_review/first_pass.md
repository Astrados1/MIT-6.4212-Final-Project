> **Superseded.** This is the first pass, kept for the record. The final labels, numbers and method are in
> [failure_analysis.md](failure_analysis.md). Changes since: the scene 5 replays at 150 s (14/15 succeed, so
> scene 5 is 26/28, not 19/28); 27 human video reviews; scene 4 relabeled from Gemini detection to partial
> observability (objects out of view); corrected total 84/137, not 77/137.

# Baseline first-pass review (`baseline_panda_20261106`)

Done November 6, 2026 from end and grasp frames, the simulator traces and the planner outputs. No video was
watched in real time. Proposed label changes are in `manual_labels_proposed.csv` (same columns as
`manual_labels.csv`). **Nothing has been applied.** A person should confirm the items marked for review, then
copy the accepted rows into the baseline's `manual_labels.csv`.

## Two scoring bugs found

### 1. Scene 5: episodes that hit the 90 s limit were scored on the reset scene
15 scene-5 trials hit the 90 s episode limit before their plan finished. At that moment the simulator
reset the scene, and the recorded "final" state is the **reset** state, with every block back where it
started. All 15 were scored as failures with wrong causes (object detection, partial observability,
needs review).

Re-scored from the last state before the reset:

| | trials | runs |
|---|---|---|
| all 3 blocks in the bowl (the arm was only returning home) | 7 | 040, 047, 088, 089, 090, 093, 138 |
| 2 of 3 placed, or still holding the third: a timeout failure | 8 | 041, 091, 092, 094, 097, 137, 143, 146 |

Two separate issues follow from this:
- **Code:** the baseline evaluator must score the state before the reset. The recovery evaluator already
  scores at its own 90 s snapshot.
- **A method decision:** is 90 s the right limit for a 3-block task? The plan takes 80-100 s of
  simulated time. Since replays are exact, the 15 plans could be replayed with a longer episode at no
  Gemini cost.

### 2. Scene 3: the "in the bin" check used world-axis boxes
The goal check builds a box along the **world** axes around each object.
- A banana lying tilted inside the bin gets a box that dips below the table, so it is rejected.
- A bin turned on the table gets a widened box that includes a banana lying beside it.

A check in the **bin's own frame** (banana center inside the bin's walls, above its floor) agrees with
every end frame where the banana is visible (7 in, 14 out). It also resolves the 6 where the wall hides it.

| | automatic | corrected |
|---|---|---|
| scene 3 successes | 6/29 | **11/29** |
| false successes | 020, 024, 029 (banana beside the bin) | |
| false failures | 021, 023, 027, 071, 074, 078, 119, 127 (banana tilted inside) | |

The same check is too strict for the round bowl and mug (a cube or can sits above a shallow rim). The
original check is correct there, since rotation doesn't change a round container's box. I confirmed six
bowl and mug successes on end frames.

## Corrected baseline numbers (pending human confirmation)

| scene | automatic task success | corrected |
|---|---|---|
| 1 cube → bowl | 18/30 | 18/30 |
| 2 can → mug | 29/30 | 29/30 |
| 3 banana → bin | 6/29 | **11/29** |
| 4 cube → mug, cans → bowl | 0/20 | 0/20 |
| 5 3 blocks → bowl (at 90 s) | 12/28 | **19/28** |
| **overall** | **65/137 (47%)** | **77/137 (56%)** |

## Cause labels

- **Scene 4 (27 trials):** the tomato soup can was **never detected** in any trial, though it is plainly
  visible. The cube was missed in 18 and the mug in 13. The automatic label is goal-translation failure for
  24 of them, but a goal can't name an object that was never detected. Proposed label: object detection.
  (In the 3 variant-0 trials Gemini did ground cube → mug, and only the soup can was missing.)
- **Scene 5 timeouts (8):** proposed `timeout/episode_time_limit`. This is not one of the paper's
  categories, so decide how to report it.
- **Scene 1 missed grasps (8):** 000, 002, 006, 049, 056, 100, 104, 106. The fingers closed 5-16 cm from
  the cube center, and the frames show side or edge contact. "M2T2 unstable grasp" looks right. These are
  the trials checkpoint recovery targets.
- **Scene 3 real failures:** the banana bounced off the rim or landed beside the bin. "Partial
  observability (shape under-approximated)" is plausible but unproven. A placement or collision cause is
  equally possible.

## Still needs a person (about 12 videos, ~20 min)

| run | why |
|---|---|
| 009, 058, 107 (scene 1 v9) | grasp looks centered (0.3-2.5 cm) yet missed; labeled partial observability |
| 005 (scene 1 v5) | cube dropped during carry |
| 109 (scene 2 v1) | lifted and carried, task failed; labeled partial observability |
| 076, 125 (scene 3 v7) | drop and missed grasp |
| 071, 075, 127 (scene 3) | banana hidden by the bin wall; bin-frame check says inside |
| 144 (scene 5 v7) | plan finished, 4 blocks moved, still failed |
| 069, 122 (scene 3) | banana behind the bin; bin-frame check says outside |

## Effect on earlier smoke-test claims
The smoke test reported that scene 5 v1 (040) "flipped from failure to success without a retry". That was
this bug: the baseline trial had already succeeded at 90 s. The checkpoint motion did not change its
outcome there.
