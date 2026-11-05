# Patches to upstream code

Each change we make to an upstream repository lives here as a patch, and nowhere else.
`scripts/setup_upstream.sh` applies them to fresh checkouts. `scripts/record_provenance.sh` and every run
record show whether they are applied. Each patch is a deviation from the upstream baseline and must also be
listed under "Behavior-changing modifications" in `docs/baseline_report.md`.

| patch | repo | why | behavior change |
|---|---|---|---|
| `tiptop/0001-gemini-robotics-er-2.patch` | tiptop @ `d8f5afd` | `gemini-robotics-er-1.6-preview`, the default in the pinned code, was shut down by Google on Aug 31, 2026. Google's documented replacement is `gemini-robotics-er-2-preview`. | Yes: a different perception model. The thinking setting also changes from `thinking_budget=0` (thinking off, documented for ER 1.6) to `thinking_level=MINIMAL`, the lowest level documented for ER 2. Google does not document a budget range for ER 2, and a request cannot combine budget and level. |

| `tiptop/0002-panda-robot-model.patch` | tiptop @ `d8f5afd` | `config/tiptop.yml` selects `robot.type: fr3_robotiq`, but droid-sim-evals simulates a Franka **Panda** (`franka_robotiq_2f_85_flattened.usd`). The two arms have the same kinematics but different joint limits (joint 6: FR3 4.5169 rad, Panda 3.7525 rad; joint 7: 3.0159 vs 2.8973). In baseline pilot run 002 an FR3 plan commanded joint 6 to 4.219 rad, the simulator stopped it at 3.7525, and the cube was released 12.5 cm off target. TiPToP already supports `panda_robotiq` (cuTAMP's `load_panda_robotiq_container`). | Yes: the planner's robot model, including its joint and velocity limits. Decided after the Nov-5 baseline, which used FR3, so later results are a separate condition. Same change as `configs/tiptop_panda_sim.yml`, which Raj's cached-scene diagnostic loaded successfully. To reproduce the FR3 baseline, reverse it: `git -C $TIPTOP_DIR/tiptop apply -R patches/tiptop/0002-panda-robot-model.patch`. |

The 0002 patch's context was rebuilt from `configs/tiptop_panda_sim.yml` (a copy of the upstream file). It applies
cleanly to that reconstruction; `setup_upstream.sh` and preflight report whether it applied to the real checkout.

Sources: [Gemini deprecations](https://ai.google.dev/gemini-api/docs/deprecations),
[Robotics ER overview](https://ai.google.dev/gemini-api/docs/robotics-overview),
[thinking levels](https://ai.google.dev/gemini-api/docs/thinking). The locked `google-genai` 1.70.0 has
`types.ThinkingLevel.MINIMAL`.

Not verified against the live API yet (no key). If ER 2 rejects `MINIMAL`, record the error and the setting
actually used, update the patch, and note the change in `docs/baseline_report.md`.
