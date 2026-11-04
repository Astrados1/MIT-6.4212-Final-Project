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
