# MIT 6.4212 Final Project: reactive grasping for TiPToP

The goal is to improve [TiPToP](https://github.com/tiptop-robot/tiptop)'s grasp success by replacing its open-loop
approach–close–lift segment with a learned, goal-conditioned visuomotor policy. TiPToP's perception, grasp generation
and task/motion planning stay as they are. The full plan is in [docs/project_plan.md](docs/project_plan.md).

## Week one: upstream baseline

Week one reproduces the upstream TiPToP baseline in the DROID/Franka simulator
([droid-sim-evals](https://github.com/tiptop-robot/droid-sim-evals)) and measures grasp outcomes with an
observation-only evaluator. No learning, retries or IIWA work happens this week.

| what | where |
|---|---|
| Step-by-step runbook (Linux + NVIDIA GPU) | [docs/setup.md](docs/setup.md) |
| Success definition, failure labels, denominators | [docs/scoring.md](docs/scoring.md) |
| Findings from reading the pinned upstream code | [docs/upstream_notes.md](docs/upstream_notes.md) |
| Results write-up (template) | [docs/baseline_report.md](docs/baseline_report.md) |
| Upstream pins | [provenance/pins.env](provenance/pins.env) |
| Clone upstream at pins / record machine / record provenance | `scripts/setup_upstream.sh`, `scripts/collect_env_info.sh`, `scripts/record_provenance.sh` |
| Observed episode (runs in the simulator env) | [grasp_eval/observed_eval.py](grasp_eval/observed_eval.py) |
| Pilot runner (scene 1, variants 0–4, 3 trials) | `scripts/run_pilot.sh` (wraps `python3 -m grasp_eval.run_suite pilot`) |
| Failure-mode probes: stage attribution, object displacement, sensor corruption, scene survey | [docs/probes.md](docs/probes.md), `python3 -m grasp_eval.run_suite <suite>` |
| Scoring / records / attribution / summary | [grasp_eval/](grasp_eval/) |

Upstream repos are not vendored. `setup_upstream.sh` clones them next to this repo at the pinned SHAs, and the
evaluator wraps the simulator from outside. The only change to upstream code is one patch in [patches/](patches/README.md):
the pinned TiPToP calls `gemini-robotics-er-1.6-preview`, which Google shut down on Aug 31, 2026, so we switch it
to `gemini-robotics-er-2-preview`. Results are therefore "upstream TiPToP with Gemini ER 2", not an exact reproduction.

## Status

- [x] Harness, scoring, run records, summary, runbook (unit tests pass, see below)
- [ ] Execution machine documented: **needs a Linux NVIDIA GPU machine (teammate)**
- [x] Patch for the retired Gemini model (ER 1.6 → ER 2); applies cleanly to pinned TiPToP, not yet tried against the live API
- [ ] Gemini API key with access to `gemini-robotics-er-2-preview`: **not obtained yet**
- [ ] Install + provenance, geometry check, replay smoke test
- [ ] Fresh end-to-end run, target mapping verified, thresholds frozen
- [ ] 15-run pilot with all failures recorded, report written
- [x] Probe suites implemented (stage attribution, displacement sweep with plan replay, sensor corruption, scene survey); unit-tested with stand-ins, not run on a GPU
- [ ] Probe suites run: displacement first, then sensor, then scenes

`observed_eval.py`, the shell scripts and the runbook have not been run on a GPU machine yet. Expect small fixes on first contact.

## Tests

The scorer and summarizer need only Python 3.10+ and numpy:

```bash
python -m unittest discover -s tests -t .
```
