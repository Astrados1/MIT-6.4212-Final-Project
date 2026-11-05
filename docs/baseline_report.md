# Week-one baseline report

_Template. Fill in as the runbook (`docs/setup.md`) progresses. Leave a section marked "not done" rather than deleting it._

## Environment

- On November 4, 2026, the RunPod Secure Cloud node reported Ubuntu 22.04.5 LTS, an NVIDIA RTX A6000
  with 46,068 MiB visible, compute capability 8.6, and driver 595.91.07. The pod had a 150 GB root
  disk and a 100 GB volume mounted at `/workspace`. The container image was
  `nvidia/cuda:12.6.3-devel-ubuntu22.04`, launched with `NVIDIA_DRIVER_CAPABILITIES=all`.
  [Machine record](../provenance/machine_8fa2f45dbe02.txt).
- The GPU node received HTTP 404 at the Gemini API root, HTTP 200 at the PyPI index, and HTTP 200
  for the SAM2.1 large checkpoint. The first network check required installing `curl` in the base image.
- The initial Vulkan check listed only the CPU renderer and reported
  `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0`.
  Installing the Ubuntu packages `libegl1` and `libgl1` in the container let `vulkaninfo --summary`
  enumerate the RTX A6000 with the NVIDIA proprietary driver. Isaac Sim camera rendering still needs the replay test.
- The runbook calls for verifying access to `gemini-robotics-er-2-preview` before installing the pinned
  stack. As of this machine check, that credential check, installation provenance, and all simulation
  runs were pending. The pod was released at 22:52 UTC after preserving its machine record. The
  next allocation should repeat the node checklist and verify model access before dependency setup.
- Differences from the paper's release: current upstream code at the pins, not the original paper release, plus the
  Gemini model substitution below.

### November 4 installation and simulator startup

- After the user clarified credential authorization, both a model lookup and a small generation request
  succeeded for `gemini-robotics-er-2-preview` with `thinkingLevel=MINIMAL`. The credential also passed
  preflight from the GPU node.
- A second RTX A6000 pod ran Ubuntu 22.04.5 with driver 595.91.07, a 350 GB root disk, and a 40 GB
  workspace volume. Upstream repositories and environments were installed under `/opt/tiptop-project`.
  [Completed machine record](../provenance/machine_9d2f7b9e8b33_after_setup.txt).
- Installation succeeded for TiPToP, cuRobo, cuTAMP 0.0.6, M2T2 and its checksum-verified weights,
  Isaac Sim 5.0.0.0, and Isaac Lab 2.2.0. TiPToP used PyTorch 2.7.1/CUDA 12.9, M2T2 used
  PyTorch 2.4.1/CUDA 12.0, and the simulator retained PyTorch 2.7.0. Both server health endpoints
  passed. The M2T2 and cuRobo revisions are recorded in `provenance/pins.env`. The upstream cuRobo
  installer fetches main, so its revision requires checking after any reinstall.
  [Installation provenance](../provenance/provenance_20261105T034937Z.txt).
- The first asset URL supplied 50 scenes and lacked scene 3 variants 10 and 11. The second documented
  URL, `https://pi-sim-assets.s3.us-east-1.amazonaws.com/assets.zip`, supplied all 51 scene files
  referenced by the config. Its SHA-256 is
  `b791aef3585fea0b56605e9ac847970d399866e909f8349f358749321fb9567d`.
  The first archive and extracted directory were retained during diagnosis.
- Stage 0 passed the GPU, OS, network, tools, layout, pins, patch, servers, credential, and disk checks
  after switching archives. Goal names remained pending. The diagnostic bootstrap used
  `--no-require-goals` to obtain the scene inspection needed to populate those names.
  [Bootstrap report](../results/preflight_20261105T034346Z.json).
- The first inspection exited 139. The native log identified missing `libXt.so.6` and `libGLU.so.1`.
  Installing Ubuntu packages `libxt6` and `libglu1-mesa` resolved those loading errors. A subsequent
  scene-1 inspection still crashed in `librtx.scenedb.plugin.so!carbOnPluginStartup`, before scene
  creation. A minimal `SimulationApp({"headless": True})` launch, with `CUDA_VISIBLE_DEVICES`
  unset, reproduced exit 139. The full diagnostic log is `minimal-sim.log` in the separately saved
  `output/tiptop-baseline-20261104/sanitized-logs` folder.
- The stack matches NVIDIA's reported R595/older-Kit startup failure. NVIDIA staff recommended R580
  for the reported case. Host-driver incompatibility is the leading diagnosis for this A6000,
  pending verification on a compatible host. [NVIDIA issue and resolution](https://github.com/isaac-sim/IsaacSim/issues/619).
- Requests for an A6000 host advertising CUDA 13.0, intended to select an R580 host, failed in both
  RunPod pools with `HTTP 500: create pod: There are no instances currently available`. Actual driver
  identity must be verified on any future allocation.
- The preflight wrapper also raised `UnicodeDecodeError` while decoding native crash output.
  Its subprocess decoder now replaces malformed bytes, preserving the exit code and readable
  diagnostics. The regression test and the other 63 harness tests pass.
- The second pod was released at 10:51 p.m. Eastern on November 4 after hash-verifying 28 downloaded
  diagnostic files. Diagnostic text copies redact credential-like vendor build metadata, and memory
  dumps were excluded. Geometry findings, replay video, scoring calibration, trial memory peaks,
  run timings, pilot results, and paper-suite results remain pending.

## Geometry check (runbook C1)

- Planner robot model vs simulator asset (fr3_robotiq vs Panda + Robotiq 2F-85): result, how checked
- TCP / grasp frame:
- Camera transform (incl. the 1.5 cm client offset):
- cuRobo interpolation rate vs the client's assumed 50 Hz:
- Gripper reference bodies used by the evaluator:
- Target object mapping (instruction → rigid body name), per scene:

## Smoke tests

- Fixed-plan replay (scene 1 / variant 0): pass/fail, video path, notes
- M2T2 / planner smoke tests:
- Fresh end-to-end run: record path, server output dir, video path

## Frozen scoring thresholds

| parameter | value | evidence |
|---|---|---|
| min_rise_m | | |
| min_hold_s | | |
| max_obj_to_gripper_m | | |

Frozen on: (date)

## Pilot results

Paste `results/pilot_scene1/summary.md` here, or link it, along with where videos and traces are stored.

## Failure modes observed

## Behavior-changing modifications

List every change with its reason and the results directories it affects.

| change | reason | affects |
|---|---|---|
| Gemini perception model `gemini-robotics-er-1.6-preview` → `gemini-robotics-er-2-preview`; thinking `thinking_budget=0` → `thinking_level=MINIMAL` (`patches/tiptop/0001-gemini-robotics-er-2.patch`) | ER 1.6 was shut down Aug 31, 2026, so the unmodified baseline cannot run. ER 2 is Google's documented replacement and has no zero-thinking budget documented. | every run (all results directories) |

## What the learned skill should replace

Which part of the executed plan (approach → close → lift) failed, and the plan step indices that bound it.

## November 5 recovery and first completed task

The user authorized changes to the GPU type and simulation software to obtain results. The working
configuration separates the original planner environments on an RTX A6000 (driver 595.91.07) from
Isaac Sim 5.0.0.0 / Isaac Lab 2.2.0 on an RTX PRO 6000 Blackwell Max-Q (driver 590.48.01).
The simulator's Torch and torchvision versions remain 2.7.0 and 0.22.0, but their CUDA wheels are
`2.7.0+cu128` and `0.22.0+cu128`, enabling sm_120. `UV_NO_SYNC=1` prevents restoration of cu126.
Two loopback-only SSH forwards connect the simulator to TiPToP port 8765 and M2T2 port 8123.
The Google credential is held only on the planner host. The simulator's credential-presence WARN
is expected; model lookup, a small generation, and the full planning request passed on the planner.

A6000 driver 595 crashed the original renderer before scene loading. Trials with Isaac Sim 6.x
required substantial Isaac Lab API changes and did not establish a valid camera pipeline. Those
experimental environments are excluded from the baseline. On Blackwell/R590, the original runtime
renders actual images and all five robot scenes complete inspection. Cold shader initialization
can exceed 100 seconds; subsequent launches are much faster. A failed synthetic cube test initially
placed the camera inside the cube; a correctly scaled scene produced a verified image.

- M2T2 sample: 515 predicted grasps; cuTAMP demo: a successful three-object plan.
- Original saved-plan replay: 450 frames / 30 seconds, exit 0, complete video. The cube remains on
  the table in sampled frames, so this is an execution check, not a successful task trial.
- Fresh smoke: `results/preflight_smoke_20261105T045535Z`; 467 control steps, plan complete,
  22.63 seconds planning, 176.91 seconds recorded wall time (excluding app launch), 9 plan steps.
  Video confirms the cube is picked up and placed in the bowl. Automatic whole-task success is true.
- Perception localized the cube within 2.3 mm in xy and associated 394 M2T2 grasps with it.
- The original 10 cm proximity threshold falsely reports no stable lift: while visibly carrying the
  cube above 5 cm, the inner-finger-link origin midpoint is 15.467–15.479 cm from the cube.
  At a provisional 20 cm limit, the same immutable trace has a 10.855 cm maximum rise and a
  6.867 second continuous hold. The smoke correction is recorded in `manual_labels.csv`.
  Pilot runs will use 5 cm rise, 0.5 second hold, and a provisional 20 cm proximity limit; freeze
  only after reviewing pilot successes and failures. Raw smoke records remain unchanged.

### Geometry findings and measurement corrections

The cube mesh is about 7.21 cm across. Both `left_inner_finger` and `right_inner_finger` exist.
Scene 4 contains two cans; scene 5 contains five candidate blocks, of which the goal requires three.
All object/container names in `configs/scenes.json` come from completed inspections.

Direct mesh-vertex checks confirm substantial mesh-center offsets from rigid-body origins in
scenes 4 and 5 (for example the scene-4 cube has a local z offset about 20 cm). These are real asset
transforms, not erroneous bounding-box metadata. The evaluator now retains raw body poses and also
records geometry-box centers, using the centers for grasp proximity, lift height, nearest-object
matching and perception comparisons. Whole-task checks continue to transform the full geometry box.
The bounding-box reader explicitly selects the PhysX body prim, supporting payloaded child bodies.
These are measurement changes only; no actions, physics, assets, retries or planning policies change.
A regression test covers a rotated object with its mesh 20 cm from its body origin.

The recorded trajectory dt is 0.02 seconds. Stride 3 at 15 Hz executes trajectories at about 1.111x
planned duration; the original timing is preserved. Panda-versus-FR3 FK comparison and the complete
pilot/paper summaries are still pending at this checkpoint.
