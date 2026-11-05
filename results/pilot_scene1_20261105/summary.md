# Summary: pilot_scene1_20261105

- Scheduled trials: 15 (records written: 15)
- Plan returned by server: 15/15 (100%)
- Valid executed plans: 15/15 (100%)
- **Stable lift, overall (per scheduled trial): 11/15 (73%), 95% CI [48-89%]**
- **Stable lift, conditional (per valid executed plan): 11/15 (73%), 95% CI [48-89%]**
- Final task success (runs with an automatic check or manual label): 9/15 (60%), 95% CI [36-80%]; unchecked: 0
- Planning time (s): median 10.0, min 8.8, max 20.8, n=15
- Runs with manual label overrides: 15

## By variant

| variant | scheduled | plan returned | valid executed | stable lift |
|---|---|---|---|---|
| 0 | 3 | 3 | 3 | 2 |
| 1 | 3 | 3 | 3 | 3 |
| 2 | 3 | 3 | 3 | 3 |
| 3 | 3 | 3 | 3 | 3 |
| 4 | 3 | 3 | 3 | 0 |

## Failure categories (denominator: 15 scheduled trials)

| category | count |
|---|---|
| missed_grasp | 4 |
| (stable lift) | 11 |

## Stage attribution (valid executed runs with attribution)

| metric | median, stable lift | median, failed |
|---|---|---|
| perceived vs true object xy (m) | 0.002 (n=11) | 0.003 (n=4) |
| gripper vs object xy at close (m) | 0.050 (n=11) | 0.028 (n=4) |
| object pushed before close (m) | 0.000 (n=11) | 0.011 (n=4) |
| joint tracking error at close (rad) | 0.001 (n=11) | 0.032 (n=4) |

Heuristic stage hints for failed runs (confirm on video):

- approach_contact: 3
- grasp_pose_or_frame: 1

- Plan picked a different object than the scored target: 0
- Plan trajectory dt [0.02] s; executed/planned duration [1.111] (1.0 means the client's 50 Hz assumption holds)

## Paper failure taxonomy (TiPToP failure-analysis figure)

Denominator: 15 paper-comparable trials = 15 scheduled minus 0 excluded (infrastructure or needs_review). Automatic labels are first passes: confirm on video and in the server log, and override with `paper_category` in manual_labels.csv. The paper's 173 trials ran on a real robot, so proportions in simulation are not expected to match. Reviewed, confirmed task failures whose cause remains uncertain are retained as Unattributed Task Failure, outside the paper's module categories. With a few dozen trials the 95% intervals are wide: "paper % inside our CI" means only that this run can't tell the two apart.

| node | ours | % of our trials | % of our failures | paper (of 173) | in paper figure | our 95% CI | paper % inside our CI |
|---|---|---|---|---|---|---|---|
| Success | 9 | 60% |  | 118 | yes | [36-80%] | yes |
| Failure | 6 | 40% |  | 55 | yes | [20-64%] | yes |
| &nbsp;&nbsp;Grasp Failure | 4 | 27% | 67% | 31 | yes | [11-52%] | yes |
| &nbsp;&nbsp;&nbsp;&nbsp;M2T2 Unstable Grasp | 4 | 27% | 67% | 20 | yes | [11-52%] | yes |
| &nbsp;&nbsp;cuTAMP Failure | 1 | 7% | 17% | 5 | yes | [1-30%] | yes |
| &nbsp;&nbsp;&nbsp;&nbsp;Rigid Grasp Assumption | 1 | 7% | 17% | 2 | yes | [1-30%] | no |
| &nbsp;&nbsp;Execution Failure (not in paper) | 1 | 7% | 17% |  | no | [1-30%] |  |
| &nbsp;&nbsp;&nbsp;&nbsp;Tracking or Collision | 1 | 7% | 17% |  | no | [1-30%] |  |

- Paper labels set manually: 15
- Task success source: automatic 0, manual 15, unchecked 0

| run | paper category | source | evidence |
|---|---|---|---|
| 002_s1v2_baseline_t0 | execution_failure/tracking_or_collision | manual | lifted and carried without slip, but the goal state was not reached |
| 004_s1v4_baseline_t0 | grasp_failure/m2t2_unstable_grasp | manual | no stable lift (missed_grasp); grasp source: m2t2 |
| 007_s1v2_baseline_t1 | cutamp_failure/rigid_grasp_assumption | manual | object moved in hand: 0.008 m, 11.45 deg |
| 009_s1v4_baseline_t1 | grasp_failure/m2t2_unstable_grasp | manual | no stable lift (missed_grasp); grasp source: m2t2 |
| 010_s1v0_baseline_t2 | grasp_failure/m2t2_unstable_grasp | manual | no stable lift (missed_grasp); grasp source: m2t2 |
| 014_s1v4_baseline_t2 | grasp_failure/m2t2_unstable_grasp | manual | no stable lift (missed_grasp); grasp source: m2t2 |

Per-run videos with their causes: review.html (open in a browser).
