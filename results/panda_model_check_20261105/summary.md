# Summary: panda_model_check_20261105

- Scheduled trials: 1 (records written: 1)
- Plan returned by server: 1/1 (100%)
- Valid executed plans: 1/1 (100%)
- **Stable lift, overall (per scheduled trial): 0/1 (0%), 95% CI [0-79%]**
- **Stable lift, conditional (per valid executed plan): 0/1 (0%), 95% CI [0-79%]**
- Final task success (runs with an automatic check or manual label): 0/1 (0%), 95% CI [0-79%]; unchecked: 0
- Runs with manual label overrides: 1

## By variant

| variant | scheduled | plan returned | valid executed | stable lift |
|---|---|---|---|---|
| 2 | 1 | 1 | 1 | 0 |

## Failure categories (denominator: 1 scheduled trials)

| category | count |
|---|---|
| missed_grasp | 1 |
| (stable lift) | 0 |

## Stage attribution (valid executed runs with attribution)

| metric | median, stable lift | median, failed |
|---|---|---|
| perceived vs true object xy (m) | n/a | 0.003 (n=1) |
| gripper vs object xy at close (m) | n/a | 0.069 (n=1) |
| object pushed before close (m) | n/a | 0.000 (n=1) |
| joint tracking error at close (rad) | n/a | 0.002 (n=1) |

Heuristic stage hints for failed runs (confirm on video):

- grasp_pose_or_frame: 1

- Plan picked a different object than the scored target: 0
- Plan trajectory dt [0.02] s; executed/planned duration [1.111] (1.0 means the client's 50 Hz assumption holds)

## Paper failure taxonomy (TiPToP failure-analysis figure)

Denominator: 1 paper-comparable trials = 1 scheduled minus 0 excluded (infrastructure or needs_review). Automatic labels are first passes: confirm on video and in the server log, and override with `paper_category` in manual_labels.csv. The paper's 173 trials ran on a real robot, so proportions in simulation are not expected to match. Reviewed, confirmed task failures whose cause remains uncertain are retained as Unattributed Task Failure, outside the paper's module categories. With a few dozen trials the 95% intervals are wide: "paper % inside our CI" means only that this run can't tell the two apart.

| node | ours | % of our trials | % of our failures | paper (of 173) | in paper figure | our 95% CI | paper % inside our CI |
|---|---|---|---|---|---|---|---|
| Success | 0 | 0% |  | 118 | yes | [0-79%] | yes |
| Failure | 1 | 100% |  | 55 | yes | [21-100%] | yes |
| &nbsp;&nbsp;Grasp Failure | 1 | 100% | 100% | 31 | yes | [21-100%] | no |
| &nbsp;&nbsp;&nbsp;&nbsp;M2T2 Unstable Grasp | 1 | 100% | 100% | 20 | yes | [21-100%] | no |

- Paper labels set manually: 1
- Task success source: automatic 0, manual 1, unchecked 0

| run | paper category | source | evidence |
|---|---|---|---|
| 000_s1v2_scene1_t0 | grasp_failure/m2t2_unstable_grasp | manual | no stable lift (missed_grasp); grasp source: m2t2 |

Per-run videos with their causes: review.html (open in a browser).
