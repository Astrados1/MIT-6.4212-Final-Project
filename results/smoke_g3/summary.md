# Summary: smoke_g3

- Scheduled trials: 1 (records written: 1)
- Plan returned by server: 1/1 (100%)
- Valid executed plans: 1/1 (100%)
- **Stable lift, overall (per scheduled trial): 1/1 (100%), 95% CI [21-100%]**
- **Stable lift, conditional (per valid executed plan): 1/1 (100%), 95% CI [21-100%]**
- Final task success (runs with an automatic check or manual label): 1/1 (100%), 95% CI [21-100%]; unchecked: 0
- Planning time (s): median 27.2, min 27.2, max 27.2, n=1
- Runs with manual label overrides: 0

## By variant

| variant | scheduled | plan returned | valid executed | stable lift |
|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 1 |

## Failure categories (denominator: 1 scheduled trials)

| category | count |
|---|---|
| (stable lift) | 1 |

## Stage attribution (valid executed runs with attribution)

| metric | median, stable lift | median, failed |
|---|---|---|
| perceived vs true object xy (m) | 0.002 (n=1) | n/a |
| gripper vs object xy at close (m) | 0.079 (n=1) | n/a |
| object pushed before close (m) | 0.000 (n=1) | n/a |
| joint tracking error at close (rad) | 0.001 (n=1) | n/a |

- Plan picked a different object than the scored target: 0
- Plan trajectory dt [0.02] s; executed/planned duration [1.111] (1.0 means the client's 50 Hz assumption holds)

## Paper failure taxonomy (TiPToP failure-analysis figure)

Denominator: 1 paper-comparable trials = 1 scheduled minus 0 excluded (infrastructure or needs_review). Automatic labels are first passes: confirm on video and in the server log, and override with `paper_category` in manual_labels.csv. The paper's 173 trials ran on a real robot, so proportions in simulation are not expected to match. Reviewed, confirmed task failures whose cause remains uncertain are retained as Unattributed Task Failure, outside the paper's module categories. With a few dozen trials the 95% intervals are wide: "paper % inside our CI" means only that this run can't tell the two apart.

| node | ours | % of our trials | % of our failures | paper (of 173) | in paper figure | our 95% CI | paper % inside our CI |
|---|---|---|---|---|---|---|---|
| Success | 1 | 100% |  | 118 | yes | [21-100%] | yes |
| Failure | 0 | 0% |  | 55 | yes | [0-79%] | yes |

- Paper labels set manually: 0
- Task success source: automatic 1, manual 0, unchecked 0

| run | paper category | source | evidence |
|---|---|---|---|

Per-run videos with their causes: review.html (open in a browser).
