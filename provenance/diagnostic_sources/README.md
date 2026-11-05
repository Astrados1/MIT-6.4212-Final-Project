# Sources used for the November 5 diagnostics

These are unchanged copies of the diagnostic scripts actually executed. They are read-only with
respect to robot state and policy. They write their result JSON under `/workspace`.

Run the three FK scripts in the pinned TiPToP pixi environment. Their temporary input mappings were:

| script | expected input | immutable result source |
|---|---|---|
| `fk-check.py` | `/workspace/smoke-trace.npz` | `results/preflight_smoke_20261105T045535Z/runs/000_s1v0_scene1_t0/trace.npz` |
| `placement-fk.py` | `/workspace/placement-failure-trace.npz` | `results/pilot_scene1_20261105/runs/002_s1v2_baseline_t0/trace.npz` |
| `acquisition-fk.py` | `/workspace/acquisition-trace.npz`, `/workspace/acquisition-record.json` | trace and record in `results/pilot_scene1_20261105/runs/007_s1v2_baseline_t1/` |

`joint-limits.py` was run in an isolated CPU environment with `usd-core`, outside the simulator
environment. `mesh-bounds-actual.py` used the working original Isaac Sim environment on Blackwell.
Both read the pinned simulator assets under `/opt/tiptop-project/droid-sim-evals/assets`.
The projection diagnostics are calculations using observed transforms, not counterfactual executions.
