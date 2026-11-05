# Summary: paper_20261105

- Scheduled trials: 40 (records written: 40)
- Plan returned by server: 5/40 (12%)
- Valid executed plans: 5/40 (12%)
- **Stable lift, overall (per scheduled trial): 3/40 (8%), 95% CI [3-20%]**
- **Stable lift, conditional (per valid executed plan): 3/5 (60%), 95% CI [23-88%]**
- Final task success (runs with an automatic check or manual label): 3/5 (60%), 95% CI [23-88%]; unchecked: 35
- Planning time (s): median 2.0, min 1.7, max 13.2, n=40
- Runs with manual label overrides: 40

## By variant

| variant | scheduled | plan returned | valid executed | stable lift |
|---|---|---|---|---|
| 0 | 10 | 1 | 1 | 0 |
| 1 | 10 | 1 | 1 | 1 |
| 2 | 10 | 2 | 2 | 1 |
| 3 | 10 | 1 | 1 | 1 |

## By condition

| condition | scheduled | valid executed | stable lift (overall) | stable lift (conditional) |
|---|---|---|---|---|
| scene1 | 8 | 4 | 2/8 (25%) | 2/4 (50%) |
| scene2 | 8 | 1 | 1/8 (12%) | 1/1 (100%) |
| scene3 | 8 | 0 | 0/8 (0%) | 0/0 (n/a) |
| scene4 | 8 | 0 | 0/8 (0%) | 0/0 (n/a) |
| scene5 | 8 | 0 | 0/8 (0%) | 0/0 (n/a) |

## Failure categories (denominator: 40 scheduled trials)

| category | count |
|---|---|
| missed_grasp | 2 |
| infrastructure | 35 |
| (stable lift) | 3 |

## Warnings

- 16 run(s) scored a target not named by the plan (nearest object or unresolved); review: 012_s4v0_scene4_t0, 013_s4v1_scene4_t0, 014_s4v2_scene4_t0, 015_s4v3_scene4_t0, 016_s5v0_scene5_t0, 017_s5v1_scene5_t0, 018_s5v2_scene5_t0, 019_s5v3_scene5_t0, 032_s4v0_scene4_t1, 033_s4v1_scene4_t1, 034_s4v2_scene4_t1, 035_s4v3_scene4_t1, 036_s5v0_scene5_t1, 037_s5v1_scene5_t1, 038_s5v2_scene5_t1, 039_s5v3_scene5_t1

## Stage attribution (valid executed runs with attribution)

| metric | median, stable lift | median, failed |
|---|---|---|
| perceived vs true object xy (m) | 0.002 (n=3) | 0.003 (n=2) |
| gripper vs object xy at close (m) | 0.025 (n=3) | 0.116 (n=2) |
| object pushed before close (m) | 0.000 (n=3) | 0.000 (n=2) |
| joint tracking error at close (rad) | 0.002 (n=3) | 0.001 (n=2) |

Heuristic stage hints for failed runs (confirm on video):

- grasp_pose_or_frame: 2

- Plan picked a different object than the scored target: 0
- Plan trajectory dt [0.02] s; executed/planned duration [1.111] (1.0 means the client's 50 Hz assumption holds)

## Paper failure taxonomy (TiPToP failure-analysis figure)

Denominator: 5 paper-comparable trials = 40 scheduled minus 35 excluded (infrastructure or needs_review). Automatic labels are first passes: confirm on video and in the server log, and override with `paper_category` in manual_labels.csv. The paper's 173 trials ran on a real robot, so proportions in simulation are not expected to match. Reviewed, confirmed task failures whose cause remains uncertain are retained as Unattributed Task Failure, outside the paper's module categories. With a few dozen trials the 95% intervals are wide: "paper % inside our CI" means only that this run can't tell the two apart.

| node | ours | % of our trials | % of our failures | paper (of 173) | in paper figure | our 95% CI | paper % inside our CI |
|---|---|---|---|---|---|---|---|
| Success | 3 | 60% |  | 118 | yes | [23-88%] | yes |
| Failure | 2 | 40% |  | 55 | yes | [12-77%] | yes |
| &nbsp;&nbsp;Grasp Failure | 2 | 40% | 100% | 31 | yes | [12-77%] | yes |
| &nbsp;&nbsp;&nbsp;&nbsp;M2T2 Unstable Grasp | 2 | 40% | 100% | 20 | yes | [12-77%] | no |

Excluded from the denominator: infrastructure 35

- Paper labels set manually: 40
- Task success source: automatic 0, manual 5, unchecked 35

| run | paper category | source | evidence |
|---|---|---|---|
| 000_s1v0_scene1_t0 | grasp_failure/m2t2_unstable_grasp | manual | no stable lift (missed_grasp); grasp source: m2t2 |
| 002_s1v2_scene1_t0 | grasp_failure/m2t2_unstable_grasp | manual | no stable lift (missed_grasp); grasp source: m2t2 |
| 004_s2v0_scene2_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 005_s2v1_scene2_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 006_s2v2_scene2_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 007_s2v3_scene2_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 008_s3v0_scene3_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 009_s3v1_scene3_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 010_s3v2_scene3_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 011_s3v3_scene3_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 012_s4v0_scene4_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 013_s4v1_scene4_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 014_s4v2_scene4_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 015_s4v3_scene4_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 016_s5v0_scene5_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 017_s5v1_scene5_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 018_s5v2_scene5_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 019_s5v3_scene5_t0 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 020_s1v0_scene1_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 021_s1v1_scene1_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 022_s1v2_scene1_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 023_s1v3_scene1_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 024_s2v0_scene2_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 025_s2v1_scene2_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 027_s2v3_scene2_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 028_s3v0_scene3_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 029_s3v1_scene3_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 030_s3v2_scene3_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 031_s3v3_scene3_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 032_s4v0_scene4_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 033_s4v1_scene4_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 034_s4v2_scene4_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 035_s4v3_scene4_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 036_s5v0_scene5_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 037_s5v1_scene5_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 038_s5v2_scene5_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| 039_s5v3_scene5_t1 | infrastructure | manual | server error looks like infrastructure: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |

Per-run videos with their causes: review.html (open in a browser).
