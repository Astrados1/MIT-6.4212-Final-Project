# Staged GPU testing

Check one layer at a time on the GPU, and only move on when the layer below passes. Stages 0–3 never
call TiPToP, M2T2 or Gemini, so they use none of your API quota. If something breaks, you know which
layer broke.

| stage | command | proves | typical cost |
|---|---|---|---|
| 0. Pod | `python3 -m grasp_eval.stages pod` | the host's driver, GPU, Vulkan and graphics libraries can run Isaac Sim | seconds, before any install |
| 1. Simulator | `python3 -m grasp_eval.stages sim` | Isaac Sim starts, builds a scene and renders real camera images | a few minutes (the first start compiles shaders) |
| 2. Scenes | `python3 -m grasp_eval.stages scenes` | every scene/variant you will use loads with its objects on the table and still | one simulator start per variant |
| 3. Arm | `python3 -m grasp_eval.stages arm` | the arm reaches commanded poses, the gripper opens and closes, and where the joint limits really are | one simulator start, about 3 minutes of simulated motion |
| 4. Trials | `preflight --smoke-run`, then `run_suite` / `recovery_suite` | the full pipeline, with Gemini | Gemini calls and trial time |

Each stage writes `results/stages/<stage>_<utc>/report.md` (for you to read) and `report.json`, prints one
`[PASS|WARN|FAIL]` line per check, and exits 1 if anything failed. **FAIL** means stop and fix it. **WARN**
means look at it before going on. Stages 1–3 also save images or a video in the same folder: look at them.
A passing number is not the same as a correct-looking scene.

The rules behind every PASS/FAIL are in `grasp_eval/stage_checks.py`. The simulator-side measurements are in
`grasp_eval/stage_worker.py`, which you can also run by hand (see its docstring).

## Picking a pod

- **The driver matters more than the GPU model.** Isaac Sim 5.0 crashes at startup on NVIDIA driver 595
  ([IsaacSim issue 619](https://github.com/isaac-sim/IsaacSim/issues/619)). The November 5 baseline ran
  the simulator on driver 590.48.01. The pod has no control over the host's driver, so check it first (stage 0)
  and stop the pod right away if it says 595.
- **Needs RT cores.** RTX A6000, L40S, RTX 6000 Ada and similar work. A100/H100/H200 have no RT cores and
  cannot render the cameras.
- **One pod is enough** if its driver is compatible: the baseline's simulator and planner peaks add up to about
  15 GB. The November 5 run used two pods only because no A6000 with an older driver was available
  (`docs/runpod_reproduction.md`).
- **Blackwell GPUs** (RTX PRO 6000 Blackwell, compute capability 12) need the CUDA 12.8 Torch override in the
  simulator environment, and M2T2 does not support them, so they only work as a simulator host in a two-pod setup.
- Container image `nvidia/cuda:12.8.1-devel-ubuntu22.04` (or 12.6.3), with `NVIDIA_DRIVER_CAPABILITIES=all`.
  About 60 GB free disk.

Between stages you can stop the pod and keep its volume. A stopped pod bills only for storage.

## Stage 0: the pod

Right after the pod starts, before installing anything heavy:

```bash
cd /workspace && git clone https://github.com/Astrados1/MIT-6.4212-Final-Project.git
cd MIT-6.4212-Final-Project
apt-get update && apt-get install -y libegl1 libgl1 libxt6 libglu1-mesa vulkan-tools
python3 -m grasp_eval.stages pod
```

What it checks: the driver version of every GPU (595 is a FAIL), RT cores, whether Vulkan sees the NVIDIA GPU
rather than only a CPU renderer, the four graphics libraries whose absence crashed the November 4 container,
the OS, free disk, and two environment variables. `OMNI_KIT_ACCEPT_EULA` is a WARN until you set it: setting it
to `YES` accepts NVIDIA's Omniverse license, so do that yourself once you have read it.

## Stage 1: the simulator

Install the simulator first (`docs/setup.md` section B, and `docs/runpod_reproduction.md` for the asset
archive and, on Blackwell only, the Torch override). Then:

```bash
export TIPTOP_DIR=/opt/tiptop-project      # parent of droid-sim-evals/
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export XDG_RUNTIME_DIR=/tmp/xdg-tiptop && mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
python3 -m grasp_eval.stages sim
```

Three steps, each in its own process, stopping at the first failure:

1. **cuda**: the simulator environment's PyTorch can multiply matrices on the GPU.
2. **launch**: Isaac Sim starts headless with cameras enabled, and nothing else. The R595 crash happened here,
   as exit 139. The report then points you back to stage 0.
3. **render**: scene 1 variant 0 is built, reset and settled exactly as the trials do, and both cameras are
   saved as PNGs. A blank or uniform image is a FAIL: the renderer is not drawing.

Open the two PNGs. You should see the table, the cube and the bowl from the external camera, and the view
from the wrist camera.

The runner sets `UV_NO_SYNC=1` for every simulator command, so `uv run` never re-installs the lockfile's
Torch over an override.

## Stage 2: every scene you will use

```bash
python3 -m grasp_eval.stages scenes                    # scenes 1-5, first 5 variants: 25 starts
python3 -m grasp_eval.stages scenes --scenes 1         # just scene 1 first, if you prefer
python3 -m grasp_eval.stages scenes --all-variants     # all 51 configured variants
```

For each scene/variant: every object has a sensible geometry box, the goal and target names exist, every
object starts on the table (geometry center between 0 and 0.6 m high, inside the workspace), and nothing is
still moving after the robot holds still for 1 s. Objects that differ from the scene's first variant are a
WARN. Each variant's folder has both camera images and a `scene_inspection.json`.

The November 5 baseline only inspected variant 0 of each scene. Later its records showed the scene-4 cube
at z ≈ −218 m in variants 1–3. That is exactly what this stage catches. The report ends with
"Do not run trials on: …" for every failed variant. Leave those out of stage 4.

An interrupted run continues with `--out-dir results/stages/scenes_<utc> --resume`.

## Stage 3: the arm

```bash
python3 -m grasp_eval.stages arm
```

No planner. The arm is driven directly through a fixed sequence in scene 1 variant 0, at most 0.5 rad/s,
holding 1.5 s at each target:

- TiPToP's home pose, then each joint moved +0.25 and −0.25 rad from home and back (free space above the table);
- TiPToP's capture pose and back;
- gripper close, then open, with nothing in it;
- **limit probes**: joint 6 commanded to 4.219 rad and joint 7 to 3.0 rad. The planner's FR3 model allows
  these, the simulated Panda does not (its limits are 3.7525 and 2.8973 rad). 4.219 rad is what baseline pilot
  run 002 commanded just before it dropped the cube beside the bowl.

PASS needs every normal move to end within 0.02 rad of its target, the gripper to visibly change state,
and no joint to leave its limits. The probes are reported as information: they show where the
simulator really stops each joint. That is the arm mismatch measured directly. If a joint goes past its
recorded limit, or stops short of it, the report WARNs. An object moving more than 1 cm means the arm hit
something.

Watch `arm_check.mp4`. `arm_trace.npz` has every commanded and measured joint angle. Skip the probes with
`--no-limit-probe`.

## Stage 4: trials

Only now start the planner services and spend Gemini calls (`docs/setup.md` section C, with your key exported
in the TiPToP server's terminal only):

```bash
python3 -m grasp_eval.preflight --tiptop-dir $TIPTOP_DIR --smoke-run       # one full trial
python3 -m grasp_eval.run_suite pilot --tiptop-dir $TIPTOP_DIR --target-object rubiks_cube \
  --eval-args --max-obj-to-gripper-m 0.20                                    # baseline pilot
python3 -m grasp_eval.recovery_suite --tiptop-dir $TIPTOP_DIR --target-object rubiks_cube   # week two
```

Both suites stop by themselves if Gemini returns a quota error. Fix the quota, then rerun with `--resume`.
